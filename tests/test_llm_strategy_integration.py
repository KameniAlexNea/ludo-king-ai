"""
Integration tests for LLM Strategy with real game scenarios.
"""

import unittest

from ludo.strategies.llm.strategy import LLMStrategy


class MockGameLLMClient:
    """Mock LLM client that makes realistic game decisions."""

    def __init__(self):
        self.call_count = 0

    def invoke(self, prompt):
        self.call_count += 1
        # Prefer capture if available
        if "CAPTURES OPPONENT" in prompt and "Token 1:" in prompt:
            return type("R", (), {"content": "1"})
        # Prefer entering home
        if "enter_home" in prompt:
            for line in prompt.split("\n"):
                if "enter_home" in line and "Token" in line:
                    tid = line.split("Token ")[1].split(":")[0]
                    return type("R", (), {"content": tid})
        # Default
        return type("R", (), {"content": "0"})

    # No separate MockResponse needed


class TestLLMStrategyGameScenarios(unittest.TestCase):
    """Test LLM strategy in various game scenarios."""

    def setUp(self):
        # Use Ollama provider by default
        self.provider = "ollama"

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
        strategy = LLMStrategy(provider=self.provider)
        strategy.llm = mock_client
        decision = strategy.decide(context)
        self.assertEqual(decision, 0)
        self.assertEqual(mock_client.call_count, 1)

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
        strategy = LLMStrategy(provider=self.provider)
        strategy.llm = mock_client
        decision = strategy.decide(context)
        self.assertEqual(decision, 1)
        self.assertEqual(mock_client.call_count, 1)

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
        strategy = LLMStrategy(provider=self.provider)
        strategy.llm = mock_client
        decision = strategy.decide(context)
        self.assertEqual(decision, 3)
        self.assertEqual(mock_client.call_count, 1)

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

            def invoke(self, prompt):
                self.call_count += 1
                return type("R", (), {"content": "invalid token 99"})

        mock_client = InvalidResponseClient()
        strategy = LLMStrategy(provider=self.provider)
        strategy.llm = mock_client
        decision = strategy.decide(context)
        self.assertIn(decision, [move["token_id"] for move in context["valid_moves"]])
        self.assertEqual(mock_client.call_count, 1)

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

            def invoke(self, prompt):
                self.call_count += 1
                return type("R", (), {"content": "0"})

        mock_client = DeterministicClient()
        strategy = LLMStrategy(provider=self.provider)
        strategy.llm = mock_client
        decisions = [strategy.decide(context) for _ in range(3)]
        self.assertTrue(all(d == decisions[0] for d in decisions))
        self.assertEqual(decisions[0], 0)
        self.assertEqual(mock_client.call_count, 3)


if __name__ == "__main__":
    unittest.main()
