import re
import asyncio
from typing import Dict, Optional
from ..base import Strategy
from ..random_strategy import RandomStrategy
from .validator import GameContextValidator
from .llm_exceptions import (
    LLMInitializationError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMResponseError,
    LLMValidationError,
)
from .config import LLMConfig
from .prompt import create_prompt
from .utils import LLMClient, CircuitBreaker, RESPONSE_PATTERNS


class LLMStrategy(Strategy):
    """
    Async LLM Strategy that uses Language Models for decision making.
    Supports both Ollama (local) and Groq (cloud) models via Langchain.
    Falls back to random strategy if LLM parsing fails.
    """

    def __init__(
        self, config: Optional[LLMConfig] = None, llm_client: Optional[LLMClient] = None
    ):
        """Initialize LLM strategy with dependency injection."""
        self.config = config or LLMConfig.from_env()

        provider = self.config.provider
        model = (
            self.config.ollama_model if provider == "ollama" else self.config.groq_model
        )

        super().__init__(
            f"LLM-{provider.title()}",
            f"Uses {provider} {model} model for strategic decisions",
        )

        self.fallback_strategy = RandomStrategy()
        self.llm = llm_client  # Allow injection for testing
        self.circuit_breaker = CircuitBreaker(
            self.config.circuit_breaker_threshold, self.config.circuit_breaker_timeout
        )

        if not self.llm:
            self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LLM based on configuration with proper error handling."""
        try:
            if self.config.provider == "ollama":
                self.llm = self._create_ollama_client()
            elif self.config.provider == "groq":
                self.llm = self._create_groq_client()
            else:
                raise LLMInitializationError(
                    f"Unsupported LLM provider: {self.config.provider}"
                )

        except Exception as e:
            if not self.config.use_fallback:
                raise LLMInitializationError(
                    f"Failed to initialize LLM {self.config.provider}: {e}"
                ) from e
            self.llm = None

    def _create_ollama_client(self):
        """Create Ollama client with error handling."""
        try:
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=self.config.ollama_model,
                temperature=self.config.temperature,
                timeout=self.config.timeout,
            )
        except ImportError as e:
            raise LLMInitializationError("langchain_ollama not installed") from e

    def _create_groq_client(self):
        """Create Groq client with error handling."""
        try:
            from langchain_groq import ChatGroq

            if not self.config.groq_api_key:
                raise LLMInitializationError("GROQ_API_KEY not provided")

            return ChatGroq(
                groq_api_key=self.config.groq_api_key,
                model_name=self.config.groq_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
            )
        except ImportError as e:
            raise LLMInitializationError("langchain_groq not installed") from e

    def decide(self, game_context: Dict) -> int:
        """
        Make a strategic decision using LLM (sync wrapper for async).
        Falls back to random strategy if LLM fails.
        """
        # For backwards compatibility, provide sync interface
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we can't use asyncio.run()
                # This is a limitation - ideally the entire game should be async
                return asyncio.create_task(self.adecide(game_context)).result()
            else:
                return asyncio.run(self.adecide(game_context))
        except Exception:
            return self.fallback_strategy.decide(game_context)

    async def adecide(self, game_context: Dict) -> int:
        """
        Async version: Make a strategic decision using LLM.
        Falls back to random strategy if LLM fails.
        """
        if not self.llm:
            return self.fallback_strategy.decide(game_context)

        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            return self.fallback_strategy.decide(game_context)

        # Validate input to prevent injection and ensure data integrity
        try:
            GameContextValidator.validate(game_context)
        except LLMValidationError:
            return self.fallback_strategy.decide(game_context)

        # Retry with exponential backoff (async)
        for attempt in range(self.config.retry_attempts):
            try:
                llm_response = await self._aget_llm_decision(game_context)

                token_id = self._parse_llm_response(llm_response, game_context)
                if token_id is not None:
                    self.circuit_breaker.record_success()
                    return token_id

                raise LLMResponseError("Failed to parse valid token ID from response")

            except (LLMNetworkError, LLMRateLimitError):
                # Exponential backoff for retries (non-blocking)
                if attempt < self.config.retry_attempts - 1:
                    delay = self.config.retry_delay * (2**attempt)
                    await asyncio.sleep(delay)

            except (LLMAuthenticationError, LLMValidationError):
                # Non-retryable errors
                self.circuit_breaker.record_failure()
                break

            except Exception:
                # Unknown errors - treat as non-retryable
                break

        # All attempts failed, record failure and use fallback
        self.circuit_breaker.record_failure()
        return self.fallback_strategy.decide(game_context)

    async def _aget_llm_decision(self, game_context: Dict) -> str:
        """Async version: Get decision from LLM using structured prompt with proper error handling."""
        prompt = self._create_prompt(game_context)

        try:
            if self.config.provider == "groq":
                # For chat models - use proper message format
                try:
                    from langchain_core.messages import HumanMessage

                    if hasattr(self.llm, "ainvoke"):
                        response = await self.llm.ainvoke(
                            [HumanMessage(content=prompt)]
                        )
                    else:
                        # Fallback to sync if async not available
                        response = self.llm.invoke([HumanMessage(content=prompt)])
                except ImportError:
                    # Fallback without HumanMessage
                    if hasattr(self.llm, "ainvoke"):
                        response = await self.llm.ainvoke(prompt)
                    else:
                        response = self.llm.invoke(prompt)

                return (
                    response.content if hasattr(response, "content") else str(response)
                )
            else:
                # For completion models (Ollama)
                if hasattr(self.llm, "ainvoke"):
                    response = await self.llm.ainvoke(prompt)
                else:
                    # Fallback to sync if async not available
                    response = self.llm.invoke(prompt)
                return (
                    response.content if hasattr(response, "content") else str(response)
                )

        except asyncio.TimeoutError as e:
            raise LLMNetworkError(f"LLM call timed out: {e}") from e
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "quota" in error_str:
                raise LLMRateLimitError(f"Rate limit exceeded: {e}") from e
            elif (
                "auth" in error_str
                or "unauthorized" in error_str
                or "forbidden" in error_str
            ):
                raise LLMAuthenticationError(f"Authentication failed: {e}") from e
            elif (
                "network" in error_str
                or "connection" in error_str
                or "timeout" in error_str
            ):
                raise LLMNetworkError(f"Network error: {e}") from e
            else:
                raise LLMResponseError(f"LLM API call failed: {e}") from e

    def _create_prompt(self, game_context: Dict) -> str:
        """Create structured prompt for LLM decision making with sanitized data."""
        valid_moves = self._get_valid_moves(game_context)
        return create_prompt(game_context, valid_moves)

    def _parse_llm_response(self, response: str, game_context: Dict) -> Optional[int]:
        """
        Parse LLM response to extract token ID with enhanced error handling.
        Uses pre-compiled regex patterns for performance.
        """
        if not response:
            raise LLMResponseError("Empty response from LLM")
        # Remove any 'think' tags or similar artifacts from the response
        response = re.sub(
            r"<\s*think\s*>.*?<\s*/\s*think\s*>",
            "",
            response,
            flags=re.DOTALL | re.IGNORECASE,
        )

        valid_moves = self._get_valid_moves(game_context)
        valid_token_ids = [move["token_id"] for move in valid_moves]

        # Clean response
        response = response.strip().lower()
        if len(response) > 200:  # Limit response length for processing
            response = response[:200]

        # Try structured parsing first (most reliable)
        try:
            # Look for JSON-like structures first
            json_match = re.search(
                r'\{[^}]*"(?:token_id|token|move)"\s*:\s*([0-3])[^}]*\}', response
            )
            if json_match:
                token_id = int(json_match.group(1))
                if token_id in valid_token_ids:
                    return token_id
        except (ValueError, AttributeError):
            pass

        # Try pre-compiled regex patterns (performance optimized)
        for pattern in RESPONSE_PATTERNS:
            matches = pattern.findall(response)
            for match in matches:
                try:
                    token_id = int(match)
                    if token_id in valid_token_ids:
                        return token_id
                except ValueError:
                    continue

        # Last resort: look for any valid token ID mentioned
        for token_id in valid_token_ids:
            if str(token_id) in response:
                return token_id

        raise LLMResponseError(
            f"Could not parse valid token ID from response: {response[:100]}"
        )
