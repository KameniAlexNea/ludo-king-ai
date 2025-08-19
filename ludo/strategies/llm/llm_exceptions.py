# Specific exception types for different error conditions
class LLMError(Exception):
    """Base exception for LLM-related errors."""

    pass


class LLMInitializationError(LLMError):
    """Raised when LLM initialization fails."""

    pass


class LLMNetworkError(LLMError):
    """Raised when network-related LLM errors occur (retryable)."""

    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limit is exceeded (retryable with longer delay)."""

    pass


class LLMAuthenticationError(LLMError):
    """Raised when authentication fails (non-retryable)."""

    pass


class LLMResponseError(LLMError):
    """Raised when LLM response cannot be parsed."""

    pass


class LLMValidationError(LLMError):
    """Raised when input validation fails."""

    pass
