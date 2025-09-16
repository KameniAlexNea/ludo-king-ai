import unittest

from ludo_engine.strategies.optimist import OptimistStrategy


class TestOptimistStrategyBehavior(unittest.TestCase):
    def setUp(self):
        self.strategy = OptimistStrategy()

    def test_prefers_capture_over_safe(self):
        cap = {
            "token_id": 0,
            "move_type": "normal",
            "current_position": 5,
            "target_position": 8,
            "captures_opponent": True,
            "captured_tokens": [{"token_id": 1, "player_color": "blue"}],
            "is_safe_move": False,
            "strategic_value": 15,
        }
        safe = {
            "token_id": 1,
            "move_type": "normal",
            "current_position": 10,
            "target_position": 12,
            "captures_opponent": False,
            "captured_tokens": [],
            "is_safe_move": True,
            "strategic_value": 25,
        }
        ctx = {
            "valid_moves": [cap, safe],
            "player_state": {"finished_tokens": 0, "active_tokens": 2, "color": "red"},
            "opponents": [],
        }
        ctx["current_situation"] = {"player_color": "red"}
        ctx["players"] = [
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
                "tokens": [{"token_id": 0, "position": 8}],
            },
        ]
        self.assertEqual(self.strategy.decide(ctx), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
