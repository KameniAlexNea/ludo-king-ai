"""
Integration tests for LLM Strategy with real game scenarios.
"""

import unittest
import asyncio

from ludo.strategies.llm.config import LLMConfig
from ludo.strategies.llm.strategy import LLMStrategy


class MockGameLLMClient:
    """Mock LLM client that makes realistic game decisions."""

    def __init__(self):
        self.call_count = 0

    async def ainvoke(self, prompt):
        """Mock that analyzes prompt and makes reasonable decisions."""
        self.call_count += 1

        if "CAPTURES OPPONENT" in prompt and "Token 1:" in prompt:
            return MockResponse("1")  # Prefer capturing

        if "enter_home" in prompt:
            lines = prompt.split("\n")
            for line in lines:
                if "enter_home" in line and "Token" in line:
                    token_id = line.split("Token ")[1][0]
                    return MockResponse(token_id)

        return MockResponse("0")  # Default


class MockResponse:
    def __init__(self, content):
        self.content = content


class TestLLMStrategyGameScenarios(unittest.TestCase):
    """Test LLM strategy in various game scenarios."""

    def setUp(self):
        self.config = LLMConfig(
            provider="ollama",
            timeout=2,
            retry_attempts=1,
            use_fallback=True,
            verbose_errors=False,
        )

    def test_early_game_scenario(self):
        """Test strategy in early game."""
        context = {
            "player_state": {"finished_tokens": 0, "home_tokens": 4},
            "opponents": [{"tokens_finished": 0}],
            "valid_moves": [
                {
                    "token_id": 0,
                    "move_type": "start_token",
                    "strategic_value": 0.8,
                    "captures_opponent": False,
                    "is_safe_move": True,
                }
            ],
        }

        mock_client = MockGameLLMClient()
        strategy = LLMStrategy(config=self.config, llm_client=mock_client)

        async def run_test():
            decision = await strategy.adecide(context)
            self.assertEqual(decision, 0)
            self.assertEqual(mock_client.call_count, 1)

        asyncio.run(run_test())

    def test_capture_opportunity(self):
        """Test strategy with capture opportunity."""
        context = {
            "player_state": {"finished_tokens": 1, "home_tokens": 1},
            "opponents": [{"tokens_finished": 0}],
            "valid_moves": [
                {
                    "token_id": 0,
                    "move_type": "move_forward",
                    "strategic_value": 0.5,
                    "captures_opponent": False,
                    "is_safe_move": True,
                },
                {
                    "token_id": 1,
                    "move_type": "capture",
                    "strategic_value": 0.9,
                    "captures_opponent": True,
                    "is_safe_move": False,
                },
            ],
        }

        mock_client = MockGameLLMClient()
        strategy = LLMStrategy(config=self.config, llm_client=mock_client)

        async def run_test():
            decision = await strategy.adecide(context)
            self.assertEqual(decision, 1)  # Should prefer capture
            self.assertEqual(mock_client.call_count, 1)

        asyncio.run(run_test())

    def test_end_game_scenario(self):
        """Test strategy in end game."""
        context = {
            "player_state": {"finished_tokens": 3, "home_tokens": 0},
            "opponents": [{"tokens_finished": 2}],
            "valid_moves": [
                {
                    "token_id": 0,
                    "move_type": "move_forward",
                    "strategic_value": 0.6,
                    "captures_opponent": False,
                    "is_safe_move": True,
                },
                {
                    "token_id": 3,
                    "move_type": "enter_home",
                    "strategic_value": 1.0,
                    "captures_opponent": False,
                    "is_safe_move": True,
                },
            ],
        }

        mock_client = MockGameLLMClient()
        strategy = LLMStrategy(config=self.config, llm_client=mock_client)

        async def run_test():
            decision = await strategy.adecide(context)
            self.assertEqual(decision, 3)  # Should prefer finishing
            self.assertEqual(mock_client.call_count, 1)

        asyncio.run(run_test())

    def test_fallback_with_invalid_response(self):
        """Test fallback when LLM gives invalid response."""
        context = {
            "player_state": {"finished_tokens": 1, "home_tokens": 2},
            "opponents": [{"tokens_finished": 1}],
            "valid_moves": [
                {
                    "token_id": 2,
                    "move_type": "move_forward",
                    "strategic_value": 0.7,
                    "captures_opponent": False,
                    "is_safe_move": True,
                }
            ],
        }

        class InvalidResponseClient:
            def __init__(self):
                self.call_count = 0

            async def ainvoke(self, prompt):
                self.call_count += 1
                return MockResponse("invalid token 99")

        mock_client = InvalidResponseClient()
        strategy = LLMStrategy(config=self.config, llm_client=mock_client)

        async def run_test():
            decision = await strategy.adecide(context)
            self.assertEqual(decision, 2)  # Should fallback to valid choice
            self.assertGreaterEqual(mock_client.call_count, 1)

        asyncio.run(run_test())

    def test_consistency(self):
        """Test strategy consistency."""
        context = {
            "player_state": {"finished_tokens": 1, "home_tokens": 2},
            "opponents": [{"tokens_finished": 1}],
            "valid_moves": [
                {
                    "token_id": 0,
                    "move_type": "move_forward",
                    "strategic_value": 0.7,
                    "captures_opponent": False,
                    "is_safe_move": True,
                }
            ],
        }

        class DeterministicClient:
            def __init__(self):
                self.call_count = 0

            async def ainvoke(self, prompt):
                self.call_count += 1
                return MockResponse("0")

        mock_client = DeterministicClient()
        strategy = LLMStrategy(config=self.config, llm_client=mock_client)

        async def run_test():
            decisions = []
            for _ in range(3):
                decision = await strategy.adecide(context)
                decisions.append(decision)

            # All decisions should be the same
            self.assertTrue(all(d == decisions[0] for d in decisions))
            self.assertTrue(all(d == 0 for d in decisions))
            self.assertEqual(mock_client.call_count, 3)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
