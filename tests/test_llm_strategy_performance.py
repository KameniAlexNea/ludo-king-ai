"""
Performance tests for LLM Strategy using unittest.
"""

import unittest
import time
from statistics import mean
from ludo.strategies.llm.strategy import LLMStrategy
from ludo.strategies.llm.prompt import create_prompt


class BenchmarkLLMClient:
    """Mock LLM client for benchmarking."""

    def __init__(self, response_delay=0.01, responses=None):
        self.response_delay = response_delay
        self.responses = responses or ["0", "1", "2", "3"]
        self.call_count = 0
        self.total_time = 0

    def invoke(self, prompt):
        start = time.time()
        time.sleep(self.response_delay)
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        self.total_time += time.time() - start

        class MockResponse:
            def __init__(self, content):
                self.content = content

        return MockResponse(response)


class TestLLMStrategyPerformance(unittest.TestCase):
    """Performance tests for LLM strategy."""

    def setUp(self):
        self.context = {
            "player_state": {"finished_tokens": 1, "home_tokens": 2},
            "opponents": [{"tokens_finished": 0}, {"tokens_finished": 1}],
            "valid_moves": [
                {"token_id": 0, "move_type": "move_forward", "strategic_value": 0.6},
                {"token_id": 1, "move_type": "capture", "strategic_value": 0.8},
                {"token_id": 2, "move_type": "enter_home", "strategic_value": 0.9},
            ],
        }

    def test_single_decision_timing(self):
        """Test timing of single decision."""
        mock_client = BenchmarkLLMClient(response_delay=0.01)
        strategy = LLMStrategy()
        strategy.llm = mock_client

        start = time.time()
        decision = strategy.decide(self.context)
        elapsed = time.time() - start

        self.assertIn(decision, [0, 1, 2])
        self.assertLess(elapsed, 0.1)
        self.assertEqual(mock_client.call_count, 1)

    def test_multiple_decisions_timing(self):
        """Test timing of multiple decisions."""
        mock_client = BenchmarkLLMClient(response_delay=0.005)
        strategy = LLMStrategy()
        strategy.llm = mock_client

        times = []
        for _ in range(10):
            start = time.time()
            decision = strategy.decide(self.context)
            times.append(time.time() - start)
            self.assertIn(decision, [0, 1, 2])

        self.assertLess(mean(times), 0.05)
        self.assertLess(max(times), 0.1)
        self.assertEqual(mock_client.call_count, 10)

    def test_concurrent_decisions(self):
        """Test sequential performance of multiple decisions."""
        mock_client = BenchmarkLLMClient(response_delay=0.01)
        strategy = LLMStrategy()
        strategy.llm = mock_client

        start = time.time()
        results = [strategy.decide(self.context) for _ in range(5)]
        elapsed = time.time() - start

        self.assertEqual(len(results), 5)
        self.assertTrue(all(r in [0, 1, 2] for r in results))
        self.assertLess(elapsed, 0.5)
        self.assertEqual(mock_client.call_count, 5)

    def test_fallback_performance(self):
        """Test performance when falling back to random."""

        class FailingClient:
            def __init__(self):
                self.call_count = 0

            def invoke(self, prompt):
                self.call_count += 1
                time.sleep(0.01)
                raise Exception("Always fails")

        mock_client = FailingClient()
        strategy = LLMStrategy()
        strategy.llm = mock_client

        start = time.time()
        decision = strategy.decide(self.context)
        elapsed = time.time() - start

        self.assertIn(decision, [0, 1, 2])
        self.assertLess(elapsed, 0.1)

    def test_prompt_creation_performance(self):
        """Test prompt creation performance."""
        context = self.context
        moves = context["valid_moves"]

        times = []
        for _ in range(100):
            start = time.time()
            prompt = create_prompt(context, moves)
            times.append(time.time() - start)
            self.assertTrue(prompt)
            self.assertIn("GAME SITUATION:", prompt)

        self.assertLess(mean(times), 0.001)

    def test_response_parsing_performance(self):
        """Test response parsing performance."""
        strategy = LLMStrategy()
        test_responses = [
            "2",
            "Token 1 is best",
            "I choose 0",
            "Decision: 3",
            '{"token_id": 2}',
            "Move token 1 forward",
        ]

        total = 0
        start = time.time()
        for _ in range(100):
            for resp in test_responses:
                tid = strategy._parse_response(resp, self.context)
                if tid is not None:
                    self.assertIn(tid, [0, 1, 2])
                total += 1
        elapsed = time.time() - start
        self.assertLess(elapsed / total, 0.001)


if __name__ == "__main__":
    unittest.main()
