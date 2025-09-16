import unittest

from ludo_engine.strategies.defensive import DefensiveStrategy


class TestDefensiveStrategyBehavior(unittest.TestCase):
    def setUp(self):
        self.strategy = DefensiveStrategy()

    def test_finish_priority(self):
        finish_mv = {
            "token_id": 0,
            "current_position": 104,
            "target_position": 105,
            "move_type": "finish",
            "captures_opponent": False,
            "captured_tokens": [],
            "is_safe_move": True,
            "strategic_value": 80.0,
        }
        safe_mv = {
            "token_id": 1,
            "current_position": 10,
            "target_position": 12,
            "move_type": "normal",
            "captures_opponent": False,
            "captured_tokens": [],
            "is_safe_move": True,
            "strategic_value": 90.0,
        }
        ctx = {
            "valid_moves": [finish_mv, safe_mv],
            "player_state": {"finished_tokens": 0, "active_tokens": 2, "color": "red"},
            "current_situation": {"player_color": "red"},
            "players": [
                {
                    "color": "red",
                    "tokens": [
                        {"token_id": 0, "position": 104},
                        {"token_id": 1, "position": 10},
                    ],
                },
                {"color": "green", "tokens": [{"token_id": 0, "position": 20}]},
            ],
            "opponents": [{"color": "green", "tokens_finished": 0}],
        }
        choice = self.strategy.decide(ctx)
        self.assertEqual(choice, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
