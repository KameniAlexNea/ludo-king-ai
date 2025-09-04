import unittest

from ludo.strategies.cautious import CautiousStrategy


def players_for_threat(current_color: str, threatening_pos: int):
    return [
        {
            "color": current_color,
            "tokens": [
                {"token_id": 0, "position": 5},
                {"token_id": 1, "position": 12},
            ],
        },
        {
            "color": "blue" if current_color != "blue" else "green",
            "tokens": [
                {"token_id": 0, "position": threatening_pos},
            ],
        },
    ]


class TestCautiousStrategyBehavior(unittest.TestCase):
    def setUp(self):
        self.strategy = CautiousStrategy()

    def test_prefers_deeper_home_column(self):
        # Two home column advances with different depth
        mv_shallow = {
            "token_id": 0,
            "current_position": 100,
            "target_position": 101,
            "move_type": "advance_home_column",
            "captures_opponent": False,
            "captured_tokens": [],
            "is_safe_move": True,
            "strategic_value": 10.0,
        }
        mv_deep = {
            "token_id": 1,
            "current_position": 102,
            "target_position": 104,
            "move_type": "advance_home_column",
            "captures_opponent": False,
            "captured_tokens": [],
            "is_safe_move": True,
            "strategic_value": 11.0,
        }
        ctx = {
            "valid_moves": [mv_shallow, mv_deep],
            "player_state": {"finished_tokens": 0, "active_tokens": 2, "color": "red"},
            "current_situation": {"player_color": "red"},
            "players": [
                {
                    "color": "red",
                    "finished_tokens": 0,
                    "tokens": [
                        {"token_id": 0, "position": 100},
                        {"token_id": 1, "position": 102},
                    ],
                },
                {
                    "color": "blue",
                    "finished_tokens": 0,
                    "tokens": [{"token_id": 0, "position": 20}],
                },
            ],
            "opponents": [{"color": "blue", "tokens_finished": 0}],
        }
        choice = self.strategy.decide(ctx)
        self.assertEqual(choice, 1, "Cautious should choose deeper home column move")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
