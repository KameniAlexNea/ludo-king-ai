from dataclasses import dataclass
import os


@dataclass
class LLMConfig:
    """Configuration for LLM strategy with proper type safety."""

    provider: str = "ollama"
    ollama_model: str = "llama2"
    groq_model: str = "mixtral-8x7b-32768"
    groq_api_key: str = ""
    temperature: float = 0.3
    max_tokens: int = 50
    timeout: int = 10  # Reduced for real-time games
    use_fallback: bool = True
    verbose_errors: bool = False
    retry_attempts: int = 2
    retry_delay: float = 0.5  # Reduced for games
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create config from environment variables with proper type conversion."""
        return cls(
            provider=os.getenv("LLM_PROVIDER", "ollama"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama2"),
            groq_model=os.getenv("GROQ_MODEL", "mixtral-8x7b-32768"),
            groq_api_key=os.getenv("GROQ_API", ""),
            temperature=float(os.getenv("TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("MAX_TOKENS", "50")),
            timeout=int(os.getenv("TIMEOUT", "10")),
            use_fallback=os.getenv("USE_FALLBACK", "true").lower()
            in ("true", "1", "yes"),
            verbose_errors=os.getenv("VERBOSE_ERRORS", "false").lower()
            in ("true", "1", "yes"),
            retry_attempts=max(
                1, int(os.getenv("RETRY_ATTEMPTS", "2"))
            ),  # Ensure at least 1 attempt
            retry_delay=float(os.getenv("RETRY_DELAY", "0.5")),
        )
