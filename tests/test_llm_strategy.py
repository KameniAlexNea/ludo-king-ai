"""
Unit tests for simplified LLM Strategy functionality.
"""

import unittest
from ludo.strategies.llm.strategy import LLMStrategy
from ludo.strategies.llm.prompt import create_prompt
from ludo.strategies.llm_strategy import OllamaStrategy, GroqStrategy


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, responses=None, should_fail=False):
        self.responses = responses or ["0", "1", "2", "3"]
        self.call_count = 0
        self.should_fail = should_fail

    def invoke(self, prompt):
        if self.should_fail:
            self.call_count += 1
            raise Exception("LLM failure")
        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1

        class R:
            def __init__(self, content):
                self.content = content

        return R(resp)


class TestLLMStrategy(unittest.TestCase):
    """Test simplified LLM strategy synchronous API."""

    def setUp(self):
        self.context = {
            "player_state": {"finished_tokens": 1, "home_tokens": 2},
            "opponents": [{"tokens_finished": 0}],
            "valid_moves": [
                {"token_id": 0, "move_type": "move_forward"},
                {"token_id": 1, "move_type": "capture"},
                {"token_id": 2, "move_type": "enter_home"},
            ],
        }

    def test_initialization(self):
        client = MockLLMClient(responses=["2"])
        strat = LLMStrategy(provider="ollama", model="test-model")
        strat.llm = client
        self.assertEqual(strat.provider, "ollama")
        self.assertEqual(strat.model, "test-model")
        self.assertEqual(strat.name, "LLM-Ollama")

    def test_successful_decide(self):
        client = MockLLMClient(responses=["1"])
        strat = LLMStrategy(provider="ollama")
        strat.llm = client
        decision = strat.decide(self.context)
        self.assertEqual(decision, 1)
        self.assertEqual(client.call_count, 1)

    def test_fallback_on_failure(self):
        client = MockLLMClient(should_fail=True)
        strat = LLMStrategy(provider="ollama")
        strat.llm = client
        decision = strat.decide(self.context)
        # fallback to random: should be one of valid ids
        self.assertIn(decision, [0, 1, 2])
        self.assertEqual(client.call_count, 1)

    def test_parse_response(self):
        strat = LLMStrategy(provider="ollama")
        cases = [
            ("2", 2),
            ("Token 1 is best", 1),
            ("I choose 0", 0),
            ("Decision: 2", 2),
            ('{"token_id": 2}', 2),
        ]
        for resp, exp in cases:
            token = strat._parse_response(resp, self.context)
            self.assertEqual(token, exp)

    def test_prompt_creation(self):
        prompt = create_prompt(self.context, self.context["valid_moves"])
        self.assertIn("GAME SITUATION:", prompt)
        self.assertIn("AVAILABLE MOVES:", prompt)


class TestConvenienceStrategies(unittest.TestCase):
    """Test Ollama and Groq strategy classes."""

    def test_ollama(self):
        strat = OllamaStrategy("mymodel")
        self.assertEqual(strat.provider, "ollama")
        self.assertEqual(strat.model, "mymodel")

    def test_groq(self):
        strat = GroqStrategy("gmodel")
        self.assertEqual(strat.provider, "groq")
        self.assertEqual(strat.model, "gmodel")


if __name__ == "__main__":
    unittest.main()
