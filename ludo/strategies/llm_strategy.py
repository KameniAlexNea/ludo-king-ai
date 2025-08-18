"""
LLM Strategy - Uses Language Models (Ollama/Groq) for decision making.
"""

import json
import re
import os
from typing import Dict, Optional
from .base import Strategy
from .random_strategy import RandomStrategy


class DefaultConfig:
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # Default to Ollama
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
    GROQ_API = os.getenv("GROQ_API", "")
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 50))
    TIMEOUT = int(os.getenv("TIMEOUT", 30))
    USE_FALLBACK = bool(os.getenv("USE_FALLBACK", 1))
    VERBOSE_ERRORS = bool(os.getenv("VERBOSE_ERRORS", 0))
    RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", 2))


config = DefaultConfig()


class LLMStrategy(Strategy):
    """
    Strategy that uses LLM (Large Language Model) to make decisions.
    Supports both Ollama (local) and Groq (cloud) models via Langchain.
    Falls back to random strategy if LLM parsing fails.
    """

    def __init__(self):
        """Initialize LLM strategy using configuration."""
        provider = config.LLM_PROVIDER
        model = config.OLLAMA_MODEL if provider == "ollama" else config.GROQ_MODEL

        super().__init__(
            f"LLM-{provider.title()}",
            f"Uses {provider} {model} model for strategic decisions",
        )

        self.config = config
        self.fallback_strategy = RandomStrategy()
        self.llm = None
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LLM based on configuration."""
        try:
            if self.config.LLM_PROVIDER == "ollama":
                from langchain_ollama import ChatOllama

                self.llm = ChatOllama(
                    model=self.config.OLLAMA_MODEL,
                    temperature=self.config.TEMPERATURE,
                    timeout=self.config.TIMEOUT,
                )
            elif self.config.LLM_PROVIDER == "groq":
                from langchain_groq import ChatGroq

                self.llm = ChatGroq(
                    groq_api_key=self.config.GROQ_API,
                    model_name=self.config.GROQ_MODEL,
                    temperature=self.config.TEMPERATURE,
                    max_tokens=self.config.MAX_TOKENS,
                    timeout=self.config.TIMEOUT,
                )
            else:
                raise ValueError(
                    f"Unsupported LLM provider: {self.config.LLM_PROVIDER}"
                )

        except Exception as e:
            if self.config.VERBOSE_ERRORS:
                print(
                    f"Warning: Failed to initialize LLM {self.config.LLM_PROVIDER}: {e}"
                )
                print("Will use fallback random strategy")
            self.llm = None

    def decide(self, game_context: Dict) -> int:
        """
        Make a strategic decision using LLM.
        Falls back to random strategy if LLM fails.
        """
        if not self.llm:
            return self.fallback_strategy.decide(game_context)

        # Try multiple times with retry logic
        for attempt in range(self.config.RETRY_ATTEMPTS + 1):
            try:
                # Get LLM decision
                llm_response = self._get_llm_decision(game_context)
                token_id = self._parse_llm_response(llm_response, game_context)

                if token_id is not None:
                    return token_id

            except Exception as e:
                if self.config.VERBOSE_ERRORS:
                    print(f"LLM decision attempt {attempt + 1} failed: {e}")

        # Fallback to random strategy
        if self.config.VERBOSE_ERRORS:
            print("All LLM attempts failed, using fallback strategy")
        return self.fallback_strategy.decide(game_context)

    def _get_llm_decision(self, game_context: Dict) -> str:
        """Get decision from LLM using structured prompt."""
        prompt = self._create_prompt(game_context)

        if self.config.LLM_PROVIDER == "groq":
            # For chat models
            response = self.llm.invoke([{"type": "user", "content": prompt}])
            return response.content
        else:
            # For completion models (Ollama)
            return self.llm.invoke(prompt)

    def _create_prompt(self, game_context: Dict) -> str:
        """Create structured prompt for LLM decision making."""
        valid_moves = self._get_valid_moves(game_context)
        player_state = game_context.get("player_state", {})
        opponents = game_context.get("opponents", [])

        # Format moves information
        moves_info = []
        for i, move in enumerate(valid_moves):
            move_desc = f"Token {move['token_id']}: "
            move_desc += f"{move['move_type']} (value: {move['strategic_value']:.2f})"

            if move.get("captures_opponent"):
                move_desc += " [CAPTURES OPPONENT]"
            if move.get("is_safe_move"):
                move_desc += " [SAFE]"
            else:
                move_desc += " [RISKY]"

            moves_info.append(move_desc)

        # Format game state
        my_progress = player_state.get("finished_tokens", 0)
        my_home_tokens = player_state.get("home_tokens", 0)
        my_active_tokens = 4 - my_home_tokens - my_progress

        opponent_progress = [opp.get("tokens_finished", 0) for opp in opponents]
        max_opponent_progress = max(opponent_progress, default=0)

        prompt = f"""You are playing Ludo. Analyze the game situation and choose the best move.

GAME SITUATION:
- My progress: {my_progress}/4 tokens finished, {my_home_tokens} at home, {my_active_tokens} active
- Opponents' progress: {opponent_progress} (max: {max_opponent_progress}/4)
- Game phase: {"Early" if my_progress == 0 else "Mid" if my_progress < 3 else "End"}

AVAILABLE MOVES:
{chr(10).join(f"{i + 1}. {move}" for i, move in enumerate(moves_info))}

STRATEGY GUIDELINES:
1. Finishing tokens (reaching home) is highest priority
2. Capturing opponents sends them back and slows them down
3. Safe moves avoid being captured by opponents
4. Risky moves might lead to capture but can advance position
5. Early game: Focus on getting tokens out of home
6. Mid game: Balance offense (capturing) and defense (safety)
7. End game: Prioritize finishing over capturing

Choose the token ID (0-3) for your move. Respond with ONLY the token number.

DECISION: """

        return prompt

    def _parse_llm_response(self, response: str, game_context: Dict) -> Optional[int]:
        """
        Parse LLM response to extract token ID.
        Returns None if parsing fails.
        """
        if not response:
            return None

        valid_moves = self._get_valid_moves(game_context)
        valid_token_ids = [move["token_id"] for move in valid_moves]

        # Clean response
        response = response.strip().lower()

        # Try to extract number from response
        patterns = [
            r"\b([0-3])\b",  # Single digit 0-3
            r"token\s*([0-3])",  # "token 2"
            r"id\s*([0-3])",  # "id 1"
            r"move\s*([0-3])",  # "move 3"
            r"choose\s*([0-3])",  # "choose 0"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                try:
                    token_id = int(match)
                    if token_id in valid_token_ids:
                        return token_id
                except ValueError:
                    continue

        # Try parsing as JSON
        try:
            # Look for JSON-like structures
            json_match = re.search(r"\{.*\}", response)
            if json_match:
                data = json.loads(json_match.group())
                token_id = data.get("token_id") or data.get("token") or data.get("move")
                if token_id is not None and int(token_id) in valid_token_ids:
                    return int(token_id)
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Last resort: look for first valid token ID mentioned
        for token_id in valid_token_ids:
            if str(token_id) in response:
                return token_id

        return None


class OllamaStrategy(LLMStrategy):
    """Convenience class for Ollama-based strategy."""

    def __init__(self, model_name: str = "llama2"):
        super().__init__(model_provider="ollama", model_name=model_name)


class GroqStrategy(LLMStrategy):
    """Convenience class for Groq-based strategy."""

    def __init__(self, model_name: str = "mixtral-8x7b-32768"):
        super().__init__(model_provider="groq", model_name=model_name)
