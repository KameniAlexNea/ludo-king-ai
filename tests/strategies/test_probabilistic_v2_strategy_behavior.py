import unittest

from ludo.strategies.probabilistic_v2 import ProbabilisticV2Strategy


class TestProbabilisticV2StrategyBehavior(unittest.TestCase):
    def setUp(self):
        self.strategy = ProbabilisticV2Strategy()

    def test_finish_short_circuit(self):
        finish = {
            "token_id": 3,
            "move_type": "finish",
            "current_position": 104,
            "target_position": 105,
            "captures_opponent": False,
            "captured_tokens": [],
            "is_safe_move": True,
            "strategic_value": 40,
        }
        mid = {
            "token_id": 0,
            "move_type": "normal",
            "current_position": 5,
            "target_position": 9,
            "captures_opponent": False,
            "captured_tokens": [],
            "is_safe_move": True,
            "strategic_value": 60,
        }
        ctx = {
            "valid_moves": [mid, finish],
            "player_state": {"finished_tokens": 0, "color": "red"},
            "opponents": [],
            "board": {"board_positions": {}},
            "current_situation": {"player_color": "red"},
            "players": [],
        }
        self.assertEqual(self.strategy.decide(ctx), 3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
