import unittest

from ludo_engine.strategies.killer import KillerStrategy


def base_players(current_color: str, opponent_pos: int):
    opp_color = "green" if current_color != "green" else "red"
    return [
        {
            "color": current_color,
            "finished_tokens": 0,
            "tokens": [
                {"token_id": 0, "position": 5},
                {"token_id": 1, "position": 10},
            ],
        },
        {
            "color": opp_color,
            "finished_tokens": 0,
            "tokens": [
                {"token_id": 0, "position": opponent_pos},
            ],
        },
    ]


class TestKillerStrategyBehavior(unittest.TestCase):
    def setUp(self):
        self.strategy = KillerStrategy()

    def test_prefers_capture_when_available_and_safe(self):
        current_color = "red"
        capture_move = {
            "token_id": 0,
            "current_position": 5,
            "target_position": 8,  # suppose opponent at 8
            "move_type": "normal",
            "captures_opponent": True,
            "captured_tokens": [
                {"token_id": 0, "player_color": "green", "position": 8}
            ],
            "is_safe_move": True,  # landing square is star (8) safe
            "strategic_value": 10.0,
        }
        quiet_move = {
            "token_id": 1,
            "current_position": 10,
            "target_position": 13,
            "move_type": "normal",
            "captures_opponent": False,
            "captured_tokens": [],
            "is_safe_move": True,
            "strategic_value": 20.0,  # higher base value than capture move
        }
        ctx = {
            "valid_moves": [capture_move, quiet_move],
            "player_state": {
                "finished_tokens": 0,
                "active_tokens": 2,
                "color": current_color,
            },
            "current_situation": {"player_color": current_color},
            "players": base_players(current_color, 8),
            "opponents": [
                {"color": "green", "tokens_finished": 0, "threat_level": 0.3}
            ],
            "board": {
                "board_positions": {
                    5: [{"player_color": current_color, "token_id": 0}],
                    8: [{"player_color": "green", "token_id": 0}],
                    10: [{"player_color": current_color, "token_id": 1}],
                }
            },
        }
        choice = self.strategy.decide(ctx)
        self.assertEqual(
            choice,
            0,
            "KillerStrategy should prioritize safe capture even if strategic_value lower",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
