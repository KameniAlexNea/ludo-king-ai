"""
LLM Strategy - Uses Language Models (Ollama/Groq) for decision making.
"""

import os
from .llm import LLMConfig, LLMStrategy


class OllamaStrategy(LLMStrategy):
    """Convenience class for Ollama-based strategy."""

    def __init__(self, model_name: str = "llama2"):
        config = LLMConfig(provider="ollama", ollama_model=model_name)
        super().__init__(config)


class GroqStrategy(LLMStrategy):
    """Convenience class for Groq-based strategy."""

    def __init__(self, model_name: str = "mixtral-8x7b-32768", api_key: str = ""):
        config = LLMConfig(
            provider="groq",
            groq_model=model_name,
            groq_api_key=api_key or os.getenv("GROQ_API", ""),
        )
        super().__init__(config)
