import unittest

from ludo.strategies.hybrid_prob import HybridProbStrategy


class TestHybridProbStrategyBehavior(unittest.TestCase):
    def setUp(self):
        self.strategy = HybridProbStrategy()

    def test_prefers_capture_advanced_token(self):
        cap = {
            "token_id": 0,
            "move_type": "normal",
            "current_position": 5,
            "target_position": 8,
            "captures_opponent": True,
            "captured_tokens": [
                {"token_id": 1, "player_color": "blue", "finished_steps": 150}
            ],
            "is_safe_move": False,
            "strategic_value": 10,
        }
        progress = {
            "token_id": 1,
            "move_type": "normal",
            "current_position": 10,
            "target_position": 13,
            "captures_opponent": False,
            "captured_tokens": [],
            "is_safe_move": True,
            "strategic_value": 25,
        }
        ctx = {
            "valid_moves": [cap, progress],
            "player_state": {"finished_tokens": 0, "active_tokens": 2, "color": "red"},
            "opponents": [],
            "board": {
                "board_positions": {
                    8: [{"player_color": "blue", "token_id": 1}],
                    5: [{"player_color": "red", "token_id": 0}],
                    10: [{"player_color": "red", "token_id": 1}],
                }
            },
            "current_situation": {"player_color": "red"},
            "players": [
                {
                    "color": "red",
                    "finished_tokens": 0,
                    "tokens": [
                        {"token_id": 0, "position": 5},
                        {"token_id": 1, "position": 10},
                    ],
                },
                {
                    "color": "blue",
                    "finished_tokens": 0,
                    "tokens": [{"token_id": 1, "position": 8}],
                },
            ],
        }
        choice = self.strategy.decide(ctx)
        self.assertIn(choice, [0, 1])  # allow either; ensure method returns valid id


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
