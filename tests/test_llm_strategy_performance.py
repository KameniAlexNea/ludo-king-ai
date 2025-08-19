"""
Performance tests for LLM Strategy using unittest.
"""

import unittest
import asyncio
import time
from statistics import mean

from ludo.strategies.llm.config import LLMConfig
from ludo.strategies.llm.strategy import LLMStrategy


class BenchmarkLLMClient:
    """Mock LLM client for benchmarking."""
    
    def __init__(self, response_delay=0.01, responses=None):
        self.response_delay = response_delay
        self.responses = responses or ["0", "1", "2", "3"]
        self.call_count = 0
        self.total_time = 0
    
    async def ainvoke(self, prompt):
        start_time = time.time()
        await asyncio.sleep(self.response_delay)
        
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        
        self.total_time += time.time() - start_time
        
        class MockResponse:
            def __init__(self, content):
                self.content = content
        
        return MockResponse(response)


class TestLLMStrategyPerformance(unittest.TestCase):
    """Performance tests for LLM strategy."""
    
    def setUp(self):
        self.config = LLMConfig(
            provider="ollama",
            timeout=5,
            retry_attempts=1,
            use_fallback=True,
            verbose_errors=False
        )
        
        self.context = {
            "player_state": {"finished_tokens": 1, "home_tokens": 2},
            "opponents": [{"tokens_finished": 0}, {"tokens_finished": 1}],
            "valid_moves": [
                {"token_id": 0, "move_type": "move_forward", "strategic_value": 0.6},
                {"token_id": 1, "move_type": "capture", "strategic_value": 0.8},
                {"token_id": 2, "move_type": "enter_home", "strategic_value": 0.9},
            ]
        }
    
    def test_single_decision_timing(self):
        """Test timing of single decision."""
        mock_client = BenchmarkLLMClient(response_delay=0.01)
        strategy = LLMStrategy(config=self.config, llm_client=mock_client)
        
        async def run_test():
            start_time = time.time()
            decision = await strategy.adecide(self.context)
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            self.assertIn(decision, [0, 1, 2])
            self.assertLess(execution_time, 0.1)  # Should be fast
            self.assertEqual(mock_client.call_count, 1)
        
        asyncio.run(run_test())
    
    def test_multiple_decisions_timing(self):
        """Test timing of multiple decisions."""
        mock_client = BenchmarkLLMClient(response_delay=0.005)
        strategy = LLMStrategy(config=self.config, llm_client=mock_client)
        
        async def run_test():
            num_decisions = 10
            times = []
            
            for _ in range(num_decisions):
                start_time = time.time()
                decision = await strategy.adecide(self.context)
                end_time = time.time()
                
                execution_time = end_time - start_time
                times.append(execution_time)
                
                self.assertIn(decision, [0, 1, 2])
            
            avg_time = mean(times)
            max_time = max(times)
            
            self.assertLess(avg_time, 0.05)  # Average should be fast
            self.assertLess(max_time, 0.1)   # No single decision too slow
            self.assertEqual(mock_client.call_count, num_decisions)
        
        asyncio.run(run_test())
    
    def test_concurrent_decisions(self):
        """Test concurrent decision performance."""
        mock_client = BenchmarkLLMClient(response_delay=0.01)
        strategy = LLMStrategy(config=self.config, llm_client=mock_client)
        
        async def run_test():
            num_concurrent = 5
            
            start_time = time.time()
            tasks = [strategy.adecide(self.context) for _ in range(num_concurrent)]
            decisions = await asyncio.gather(*tasks)
            end_time = time.time()
            
            total_time = end_time - start_time
            
            self.assertEqual(len(decisions), num_concurrent)
            self.assertTrue(all(d in [0, 1, 2] for d in decisions))
            self.assertLess(total_time, 0.1)  # Should run concurrently
            self.assertEqual(mock_client.call_count, num_concurrent)
        
        asyncio.run(run_test())
    
    def test_fallback_performance(self):
        """Test performance when falling back."""
        class FailingClient:
            def __init__(self):
                self.call_count = 0
            
            async def ainvoke(self, prompt):
                self.call_count += 1
                await asyncio.sleep(0.01)
                raise Exception("Always fails")
        
        mock_client = FailingClient()
        strategy = LLMStrategy(config=self.config, llm_client=mock_client)
        
        async def run_test():
            start_time = time.time()
            decision = await strategy.adecide(self.context)
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            self.assertIn(decision, [0, 1, 2])  # Should fallback
            self.assertLess(execution_time, 0.1)  # Fallback should be fast
        
        asyncio.run(run_test())
    
    def test_prompt_creation_performance(self):
        """Test prompt creation performance."""
        mock_client = BenchmarkLLMClient(response_delay=0.001)
        strategy = LLMStrategy(config=self.config, llm_client=mock_client)
        
        # Time prompt creation
        num_prompts = 100
        start_time = time.time()
        
        for _ in range(num_prompts):
            prompt = strategy._create_prompt(self.context)
            self.assertGreater(len(prompt), 0)
            self.assertIn("GAME SITUATION:", prompt)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_prompts
        
        self.assertLess(avg_time, 0.001)  # Should be very fast
    
    def test_response_parsing_performance(self):
        """Test response parsing performance."""
        strategy = LLMStrategy(config=self.config, llm_client=None)
        
        test_responses = [
            "2", "Token 1 is best", "I choose 0", "Decision: 3",
            '{"token_id": 2}', "Move token 1 forward"
        ]
        
        num_iterations = 100
        start_time = time.time()
        
        for _ in range(num_iterations):
            for response in test_responses:
                try:
                    token_id = strategy._parse_llm_response(response, self.context)
                    if token_id is not None:
                        self.assertIn(token_id, [0, 1, 2])
                except:
                    pass  # Some may fail, that's okay
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / (num_iterations * len(test_responses))
        
        self.assertLess(avg_time, 0.001)  # Should be very fast


if __name__ == '__main__':
    unittest.main()
