import re
import asyncio
from typing import Union, Any, Protocol
from abc import abstractmethod


class LLMClient(Protocol):
    """Protocol for LLM clients to enable dependency injection and testing."""

    @abstractmethod
    async def ainvoke(self, prompt: Union[str, list]) -> Any:
        """Async invoke method for LLM clients."""
        pass


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if asyncio.get_event_loop().time() - self.last_failure_time >= self.timeout:
                self.state = "half-open"
                return True
            return False
        else:  # half-open
            return True

    def record_success(self):
        """Record successful execution."""
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = asyncio.get_event_loop().time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


# Compiled regex patterns (performance optimization)
RESPONSE_PATTERNS = [
    re.compile(r"(?:^|\s)([0-3])(?:\s|$)"),  # Standalone digit
    re.compile(r"token\s*(?:id\s*)?(?:is\s*)?([0-3])"),  # "token 2" or "token id 2"
    re.compile(r"(?:choose|select|pick)\s*(?:token\s*)?([0-3])"),  # "choose 2"
    re.compile(r"decision\s*(?:is\s*)?(?:token\s*)?([0-3])"),  # "decision is 2"
    re.compile(r"move\s*(?:token\s*)?([0-3])"),  # "move token 2"
]
