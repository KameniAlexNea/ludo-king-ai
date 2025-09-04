import unittest

from ludo.strategies.probabilistic import ProbabilisticStrategy


class TestProbabilisticStrategyBehavior(unittest.TestCase):
    def setUp(self):
        self.strategy = ProbabilisticStrategy()

    def test_finish_short_circuit(self):
        finish = {"token_id": 2, "move_type": "finish", "current_position": 104, "target_position": 105, "captures_opponent": False, "captured_tokens": [], "is_safe_move": True, "strategic_value": 50}
        other = {"token_id": 0, "move_type": "normal", "current_position": 5, "target_position": 8, "captures_opponent": False, "captured_tokens": [], "is_safe_move": True, "strategic_value": 90}
        ctx = {"valid_moves": [other, finish], "player_state": {"finished_tokens": 0, "color": "red"}, "opponents": [], "board": {"board_positions": {}}, "current_situation": {"player_color": "red"}, "players": []}
        self.assertEqual(self.strategy.decide(ctx), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
