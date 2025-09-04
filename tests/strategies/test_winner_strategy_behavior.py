import unittest

from ludo.strategies.winner import WinnerStrategy


class TestWinnerStrategyBehavior(unittest.TestCase):
    def setUp(self):
        self.strategy = WinnerStrategy()

    def test_finish_then_home_depth(self):
        finish_mv = {
            "token_id": 2,
            "current_position": 104,
            "target_position": 105,
            "move_type": "finish",
            "captures_opponent": False,
            "captured_tokens": [],
            "is_safe_move": True,
            "strategic_value": 50,
        }
        deep_home = {
            "token_id": 0,
            "current_position": 101,
            "target_position": 103,
            "move_type": "advance_home_column",
            "captures_opponent": False,
            "captured_tokens": [],
            "is_safe_move": True,
            "strategic_value": 60,
        }
        shallow_home = {
            "token_id": 1,
            "current_position": 100,
            "target_position": 101,
            "move_type": "advance_home_column",
            "captures_opponent": False,
            "captured_tokens": [],
            "is_safe_move": True,
            "strategic_value": 70,
        }
        ctx = {
            "valid_moves": [deep_home, shallow_home, finish_mv],
            "player_state": {"finished_tokens": 0, "active_tokens": 3, "color": "red"},
            "current_situation": {"player_color": "red"},
            "players": [
                {"color": "red", "finished_tokens": 0, "tokens": [{"token_id": 0, "position": 101}, {"token_id": 1, "position": 100}, {"token_id": 2, "position": 104}]}
            ],
            "opponents": [
                {"color": "green", "tokens_finished": 0}
            ],
        }
        self.assertEqual(self.strategy.decide(ctx), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
