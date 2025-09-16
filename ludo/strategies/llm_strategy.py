"""
LLM Strategy - Uses Language Models (Ollama/Groq) for decision making.
"""

import os

from ludo_engine.strategies.llm import LLMStrategy


class OllamaStrategy(LLMStrategy):
    """Convenience class for Ollama-based strategy."""

    def __init__(self, model_name: str = os.getenv("LLM_MODEL", "gpt-oss")):
        super().__init__(provider="ollama", model=model_name)


class GroqStrategy(LLMStrategy):
    """Convenience class for Groq-based strategy."""

    def __init__(self, model_name: str = os.getenv("LLM_MODEL", "gpt-oss")):
        super().__init__(provider="groq", model=model_name)
