import unittest

from ludo.strategies.weighted_random import WeightedRandomStrategy


class TestWeightedRandomStrategyBehavior(unittest.TestCase):
    def setUp(self):
        self.strategy = WeightedRandomStrategy()

    def test_always_finish_if_available(self):
        finish = {"token_id": 2, "move_type": "finish", "current_position": 104, "target_position": 105, "captures_opponent": False, "captured_tokens": [], "is_safe_move": True, "strategic_value": 50}
        other = {"token_id": 0, "move_type": "normal", "current_position": 5, "target_position": 8, "captures_opponent": False, "captured_tokens": [], "is_safe_move": True, "strategic_value": 200}
        ctx = {"valid_moves": [finish, other], "player_state": {"finished_tokens": 0, "color": "red"}}
        # Run multiple iterations to ensure deterministic finish choice
        for _ in range(30):
            self.assertEqual(self.strategy.decide(ctx), 2)

    def test_sampling_distribution_not_degenerate(self):
        # Without finish, ensure all moves can appear
        m1 = {"token_id": 0, "move_type": "normal", "current_position": 5, "target_position": 8, "captures_opponent": False, "captured_tokens": [], "is_safe_move": True, "strategic_value": 10}
        m2 = {"token_id": 1, "move_type": "normal", "current_position": 10, "target_position": 12, "captures_opponent": False, "captured_tokens": [], "is_safe_move": True, "strategic_value": 15}
        m3 = {"token_id": 2, "move_type": "normal", "current_position": 15, "target_position": 18, "captures_opponent": False, "captured_tokens": [], "is_safe_move": False, "strategic_value": 5}
        ctx = {"valid_moves": [m1, m2, m3], "player_state": {"finished_tokens": 0, "color": "red"}}
        seen = set()
        for _ in range(120):
            seen.add(self.strategy.decide(ctx))
        # Expect at least two different tokens sampled (0 or 2 plus 1 safe progress)
        self.assertGreaterEqual(len(seen), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
