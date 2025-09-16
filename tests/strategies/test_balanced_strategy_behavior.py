import unittest

from ludo_engine.strategies.balanced import BalancedStrategy


class TestBalancedStrategyBehavior(unittest.TestCase):
    def setUp(self):
        self.strategy = BalancedStrategy()

    def test_capture_when_behind(self):
        capture_move = {
            "token_id": 0,
            "move_type": "normal",
            "current_position": 5,
            "target_position": 8,
            "captures_opponent": True,
            "captured_tokens": [{"token_id": 1, "player_color": "green"}],
            "is_safe_move": False,
            "strategic_value": 10,
        }
        safe_move = {
            "token_id": 1,
            "move_type": "normal",
            "current_position": 10,
            "target_position": 12,
            "captures_opponent": False,
            "captured_tokens": [],
            "is_safe_move": True,
            "strategic_value": 20,
        }
        ctx = {
            "valid_moves": [capture_move, safe_move],
            "player_state": {"finished_tokens": 0, "active_tokens": 2, "color": "red"},
            "opponents": [{"color": "green", "tokens_finished": 2}],
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
                    "color": "green",
                    "finished_tokens": 2,
                    "tokens": [{"token_id": 1, "position": 8}],
                },
            ],
        }
        self.assertEqual(self.strategy.decide(ctx), 0)

    def test_safe_when_ahead(self):
        # Mark capture as unsafe and highly threatened (simulate multiple attackers)
        risky_capture = {
            "token_id": 0,
            "move_type": "normal",
            "current_position": 5,
            "target_position": 8,
            "captures_opponent": True,
            "captured_tokens": [{"token_id": 1, "player_color": "green"}],
            "is_safe_move": False,
            "strategic_value": 25,
        }
        safe_move = {
            "token_id": 1,
            "move_type": "normal",
            "current_position": 10,
            "target_position": 12,
            "captures_opponent": False,
            "captured_tokens": [],
            "is_safe_move": True,
            "strategic_value": 20,
        }
        ctx = {
            "valid_moves": [risky_capture, safe_move],
            "player_state": {"finished_tokens": 3, "active_tokens": 2, "color": "red"},
            "opponents": [{"color": "green", "tokens_finished": 1}],
            "current_situation": {"player_color": "red"},
            "players": [
                {
                    "color": "red",
                    "finished_tokens": 3,
                    "tokens": [
                        {"token_id": 0, "position": 5},
                        {"token_id": 1, "position": 10},
                    ],
                },
                # Simulate two green tokens that could threaten landing square soon -> raise threat
                # Opponent tokens positioned so landing at 8 is threatened by >1 attackers (positions 2,3,4 all within 6 behind 8)
                {
                    "color": "green",
                    "finished_tokens": 1,
                    "tokens": [
                        {"token_id": 1, "position": 2},
                        {"token_id": 2, "position": 3},
                        {"token_id": 3, "position": 4},
                    ],
                },
            ],
        }
        # ahead path chooses defensive path -> safe move
        self.assertEqual(self.strategy.decide(ctx), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
