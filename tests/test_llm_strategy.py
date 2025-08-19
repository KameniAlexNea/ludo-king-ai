"""
Unit tests for LLM Strategy functionality.
"""

import unittest
import asyncio
import os
from unittest.mock import patch

# Import the LLM strategy components
from ludo.strategies.llm.config import LLMConfig
from ludo.strategies.llm.strategy import LLMStrategy
from ludo.strategies.llm.validator import GameContextValidator
from ludo.strategies.llm.llm_exceptions import (
    LLMValidationError,
)
from ludo.strategies.llm.utils import CircuitBreaker
from ludo.strategies.llm_strategy import OllamaStrategy, GroqStrategy


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, responses=None, should_fail=False, fail_type="network"):
        self.responses = responses or ["2", "1", "0", "3"]
        self.call_count = 0
        self.should_fail = should_fail
        self.fail_type = fail_type

    async def ainvoke(self, prompt):
        """Mock async invoke."""
        if self.should_fail:
            if self.fail_type == "network":
                raise Exception("Network error")
            elif self.fail_type == "timeout":
                raise asyncio.TimeoutError("Request timeout")
            elif self.fail_type == "auth":
                raise Exception("Authentication failed")
            elif self.fail_type == "rate_limit":
                raise Exception("Rate limit exceeded")

        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1

        class MockResponse:
            def __init__(self, content):
                self.content = content

        return MockResponse(response)


class TestLLMConfig(unittest.TestCase):
    """Test LLM configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LLMConfig()
        self.assertEqual(config.provider, "ollama")
        self.assertEqual(config.ollama_model, "llama2")
        self.assertEqual(config.timeout, 10)
        self.assertTrue(config.use_fallback)
        self.assertEqual(config.retry_attempts, 2)

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "groq",
                "GROQ_MODEL": "test-model",
                "TIMEOUT": "20",
                "USE_FALLBACK": "false",
                "RETRY_ATTEMPTS": "3",
            },
        ):
            config = LLMConfig.from_env()
            self.assertEqual(config.provider, "groq")
            self.assertEqual(config.groq_model, "test-model")
            self.assertEqual(config.timeout, 20)
            self.assertFalse(config.use_fallback)
            self.assertEqual(config.retry_attempts, 3)

    def test_boolean_parsing(self):
        """Test proper boolean parsing."""
        test_cases = [("true", True), ("false", False), ("1", True), ("0", False)]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"USE_FALLBACK": env_value}):
                config = LLMConfig.from_env()
                self.assertEqual(config.use_fallback, expected)


class TestGameContextValidator(unittest.TestCase):
    """Test game context validation."""

    def setUp(self):
        self.valid_context = {
            "player_state": {"finished_tokens": 1, "home_tokens": 2},
            "opponents": [{"tokens_finished": 0}, {"tokens_finished": 2}],
            "valid_moves": [
                {"token_id": 0, "move_type": "move_forward"},
                {"token_id": 1, "move_type": "capture"},
            ],
        }

    def test_valid_context(self):
        """Test validation of valid context."""
        GameContextValidator.validate(self.valid_context)  # Should not raise

    def test_missing_fields(self):
        """Test validation with missing fields."""
        with self.assertRaises(LLMValidationError):
            GameContextValidator.validate({})

        with self.assertRaises(LLMValidationError):
            GameContextValidator.validate({"player_state": {}})

    def test_invalid_player_state(self):
        """Test validation with invalid player state."""
        invalid_context = self.valid_context.copy()
        invalid_context["player_state"] = {"finished_tokens": -1}

        with self.assertRaises(LLMValidationError):
            GameContextValidator.validate(invalid_context)

    def test_invalid_moves(self):
        """Test validation with invalid moves."""
        invalid_context = self.valid_context.copy()
        invalid_context["valid_moves"] = []  # Empty moves

        with self.assertRaises(LLMValidationError):
            GameContextValidator.validate(invalid_context)


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker functionality."""

    def test_initial_state(self):
        """Test initial state."""
        cb = CircuitBreaker(failure_threshold=3, timeout=60)
        self.assertEqual(cb.state, "closed")
        self.assertTrue(cb.can_execute())
        self.assertEqual(cb.failure_count, 0)

    def test_failure_counting(self):
        """Test failure counting."""
        cb = CircuitBreaker(failure_threshold=2, timeout=1)

        cb.record_failure()
        self.assertEqual(cb.failure_count, 1)
        self.assertEqual(cb.state, "closed")

        cb.record_failure()
        self.assertEqual(cb.state, "open")
        self.assertFalse(cb.can_execute())

    def test_success_reset(self):
        """Test success resets circuit breaker."""
        cb = CircuitBreaker(failure_threshold=2, timeout=1)
        cb.record_failure()
        self.assertEqual(cb.failure_count, 1)

        cb.record_success()
        self.assertEqual(cb.failure_count, 0)
        self.assertEqual(cb.state, "closed")


class TestLLMStrategy(unittest.TestCase):
    """Test LLM strategy functionality."""

    def setUp(self):
        self.config = LLMConfig(
            provider="ollama",
            timeout=5,
            retry_attempts=2,
            use_fallback=True,
            verbose_errors=False,
        )

        self.valid_context = {
            "player_state": {"finished_tokens": 1, "home_tokens": 2},
            "opponents": [{"tokens_finished": 0}],
            "valid_moves": [
                {"token_id": 0, "move_type": "move_forward", "strategic_value": 0.7},
                {"token_id": 1, "move_type": "capture", "strategic_value": 0.9},
                {"token_id": 2, "move_type": "enter_home", "strategic_value": 1.0},
            ],
        }

    def test_initialization(self):
        """Test strategy initialization."""
        mock_client = MockLLMClient()
        strategy = LLMStrategy(config=self.config, llm_client=mock_client)

        self.assertEqual(strategy.config, self.config)
        self.assertEqual(strategy.llm, mock_client)
        self.assertEqual(strategy.name, "LLM-Ollama")

    def test_successful_decision(self):
        """Test successful decision making."""
        mock_client = MockLLMClient(responses=["2"])
        strategy = LLMStrategy(config=self.config, llm_client=mock_client)

        # Run async test
        async def run_test():
            decision = await strategy.adecide(self.valid_context)
            self.assertEqual(decision, 2)
            self.assertEqual(mock_client.call_count, 1)

        asyncio.run(run_test())

    def test_fallback_on_failure(self):
        """Test fallback when LLM fails."""
        mock_client = MockLLMClient(should_fail=True)
        strategy = LLMStrategy(config=self.config, llm_client=mock_client)

        async def run_test():
            decision = await strategy.adecide(self.valid_context)
            self.assertIn(decision, [0, 1, 2])  # Valid fallback

        asyncio.run(run_test())

    def test_invalid_context_handling(self):
        """Test handling of invalid context."""
        mock_client = MockLLMClient()
        strategy = LLMStrategy(config=self.config, llm_client=mock_client)

        async def run_test():
            decision = await strategy.adecide({"invalid": "context"})
            self.assertIsInstance(decision, int)
            self.assertEqual(mock_client.call_count, 0)  # No LLM calls

        asyncio.run(run_test())

    def test_response_parsing(self):
        """Test response parsing."""
        test_cases = [
            ("2", 2),
            ("Token 1", 1),
            ("I choose 0", 0),
            ("Decision: 1", 1),
            ('{"token_id": 2}', 2),
        ]

        for response, expected in test_cases:
            mock_client = MockLLMClient(responses=[response])
            strategy = LLMStrategy(config=self.config, llm_client=mock_client)

            async def run_test():
                decision = await strategy.adecide(self.valid_context)
                self.assertEqual(decision, expected)

            asyncio.run(run_test())

    def test_sync_wrapper(self):
        """Test synchronous wrapper."""
        mock_client = MockLLMClient(responses=["1"])
        strategy = LLMStrategy(config=self.config, llm_client=mock_client)

        decision = strategy.decide(self.valid_context)
        self.assertEqual(decision, 1)

    def test_prompt_creation(self):
        """Test prompt creation."""
        mock_client = MockLLMClient()
        strategy = LLMStrategy(config=self.config, llm_client=mock_client)

        prompt = strategy._create_prompt(self.valid_context)

        self.assertIn("GAME SITUATION:", prompt)
        self.assertIn("AVAILABLE MOVES:", prompt)
        self.assertIn("My progress: 1/4", prompt)
        self.assertIn("Token 0: move_forward", prompt)
        self.assertIn("Token 1: capture", prompt)
        self.assertIn("Token 2: enter_home", prompt)


class TestConvenienceClasses(unittest.TestCase):
    """Test convenience classes."""

    def test_ollama_strategy(self):
        """Test OllamaStrategy creation."""
        with patch("ludo.strategies.llm.strategy.LLMStrategy._initialize_llm"):
            strategy = OllamaStrategy("test-model")
            self.assertEqual(strategy.config.provider, "ollama")
            self.assertEqual(strategy.config.ollama_model, "test-model")

    def test_groq_strategy(self):
        """Test GroqStrategy creation."""
        with patch("ludo.strategies.llm.strategy.LLMStrategy._initialize_llm"):
            strategy = GroqStrategy("test-model", "test-key")
            self.assertEqual(strategy.config.provider, "groq")
            self.assertEqual(strategy.config.groq_model, "test-model")
            self.assertEqual(strategy.config.groq_api_key, "test-key")


class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios."""

    def setUp(self):
        self.config = LLMConfig(retry_attempts=1, use_fallback=True)
        self.valid_context = {
            "player_state": {"finished_tokens": 1, "home_tokens": 2},
            "opponents": [{"tokens_finished": 0}],
            "valid_moves": [{"token_id": 0, "move_type": "move_forward"}],
        }

    def test_network_error(self):
        """Test network error handling."""
        mock_client = MockLLMClient(should_fail=True, fail_type="network")
        strategy = LLMStrategy(config=self.config, llm_client=mock_client)

        async def run_test():
            decision = await strategy.adecide(self.valid_context)
            self.assertIn(decision, [0])  # Should fallback

        asyncio.run(run_test())

    def test_auth_error(self):
        """Test authentication error handling."""
        mock_client = MockLLMClient(should_fail=True, fail_type="auth")
        strategy = LLMStrategy(config=self.config, llm_client=mock_client)

        async def run_test():
            decision = await strategy.adecide(self.valid_context)
            self.assertIn(decision, [0])  # Should fallback

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
