"""
Simple LLM Strategy - Just call the LLM and parse the response.
"""

import os
import re
from typing import Dict, Optional

from ..base import Strategy
from ..random_strategy import RandomStrategy
from .prompt import create_prompt

RESPONSE_PATTERNS = [
    re.compile(r"(?:^|\s)([0-3])(?:\s|$)"),  # Standalone digit
    re.compile(r"token\s*(?:id\s*)?(?:is\s*)?([0-3])"),  # "token 2" or "token id 2"
    re.compile(r"(?:choose|select|pick)\s*(?:token\s*)?([0-3])"),  # "choose 2"
    re.compile(r"decision\s*(?:is\s*)?(?:token\s*)?([0-3])"),  # "decision is 2"
    re.compile(r"move\s*(?:token\s*)?([0-3])"),  # "move token 2"
]


class LLMStrategy(Strategy):
    """
    Simple LLM strategy that calls an LLM and parses the token ID.
    Falls back to random strategy if anything goes wrong.
    """

    def __init__(self, provider: str = None, model: str = None):
        """Initialize LLM strategy."""
        if provider is None:
            provider = os.getenv("LLM_PROVIDER", "ollama")
        if model is None:
            model = os.getenv("LLM_MODEL", "gpt-oss")
        super().__init__(
            f"LLM-{provider.title()}",
            f"Uses {provider} {model} model for decisions",
        )

        self.provider = provider
        self.model = model
        self.fallback_strategy = RandomStrategy()
        self.llm = None

        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LLM client."""
        if self.provider == "ollama":
            from langchain_ollama import ChatOllama

            self.llm = ChatOllama(model=self.model, temperature=0.1)
        elif self.provider == "groq":
            from langchain_groq import ChatGroq

            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                print("GROQ_API_KEY not found, falling back to random strategy")
                return
            self.llm = ChatGroq(
                groq_api_key=api_key, model_name=self.model, temperature=0.1
            )

    def decide(self, game_context: Dict) -> int:
        """Make a decision using the LLM."""
        if not self.llm:
            return self.fallback_strategy.decide(game_context)

        try:
            # Create prompt
            valid_moves = self._get_valid_moves(game_context)
            prompt = create_prompt(game_context, valid_moves)

            # Call LLM
            if self.provider == "groq":
                from langchain_core.messages import HumanMessage

                response = self.llm.invoke([HumanMessage(content=prompt)])
                response_text = response.content
            else:
                response = self.llm.invoke(prompt)
                response_text = (
                    response.content if hasattr(response, "content") else str(response)
                )

            # Parse response
            token_id = self._parse_response(response_text, game_context)
            if token_id is not None:
                return token_id

        except Exception:
            pass

        # Fallback to random strategy
        return self.fallback_strategy.decide(game_context)

    def _parse_response(self, response: str, game_context: Dict) -> Optional[int]:
        """Parse the LLM response to extract token ID."""
        if not response:
            return None
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
