from __future__ import annotations

import random
import unittest
from types import SimpleNamespace

import numpy as np

from ludo_rl.ludo.config import config
from ludo_rl.ludo.player import Player
from ludo_rl.ludo_env import format_env_state
from ludo_rl.strategy import available, create
from ludo_rl.strategy.registry import STRATEGY_REGISTRY
from ludo_rl.strategy.rusher import RusherStrategy


class StrategyRegistryTests(unittest.TestCase):
    def test_available_matches_registry(self) -> None:
        expected = set(STRATEGY_REGISTRY.keys())
        self.assertEqual(set(available().keys()), expected)

    def test_create_returns_strategy_instance(self) -> None:
        strategy = create("rusher")
        self.assertIsInstance(strategy, RusherStrategy)


class PlayerDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(0)

    def _board_stack(self) -> np.ndarray:
        return np.zeros((10, config.PATH_LENGTH), dtype=float)

    def test_player_decide_uses_configured_strategy(self) -> None:
        player = Player(color=0)
        player.strategy_name = "rusher"

        piece_a = player.pieces[0]
        piece_b = player.pieces[1]
        piece_a.position = 0
        piece_b.position = 10

        valid_moves = [
            {"piece": piece_a, "new_pos": 11},
            {"piece": piece_b, "new_pos": 12},
        ]

        decision = player.decide(self._board_stack(), 6, valid_moves)
        self.assertIsNotNone(decision)
        self.assertIs(decision["piece"], piece_a)
        self.assertEqual(decision["new_pos"], 11)

    def test_player_unknown_strategy_falls_back_to_random(self) -> None:
        player = Player(color=0)
        player.strategy_name = "unknown"

        piece_a = player.pieces[0]
        piece_b = player.pieces[1]
        piece_a.position = 0
        piece_b.position = 10

        valid_moves = [
            {"piece": piece_a, "new_pos": 6},
            {"piece": piece_b, "new_pos": 16},
        ]

        decision = player.decide(self._board_stack(), 6, valid_moves)
        self.assertEqual(player.strategy_name, "random")
        self.assertIsNotNone(decision)
        self.assertIn(decision["piece"], (piece_a, piece_b))


class FormatEnvStateTests(unittest.TestCase):
    def test_format_env_state_outputs_turn_information(self) -> None:
        class DummyPiece:
            def __init__(self, position: int) -> None:
                self.position = position

        class DummyPlayer:
            def __init__(self, positions: list[int]) -> None:
                self.pieces = [DummyPiece(pos) for pos in positions]

            def has_won(self) -> bool:
                return False

        dummy_env = SimpleNamespace(
            current_turn=5,
            max_game_turns=200,
            agent_index=0,
            simulator=SimpleNamespace(
                game=SimpleNamespace(
                    players=[
                        DummyPlayer([0, 0, 0, 0]),
                        DummyPlayer([1, 2, 3, 4]),
                        DummyPlayer([5, 6, 7, 8]),
                        DummyPlayer([9, 10, 11, 12]),
                    ]
                )
            ),
        )

        snapshot = format_env_state(dummy_env)
        self.assertIn("Turn 5/200", snapshot)
        self.assertIn("AGENT (P0)", snapshot)
        self.assertIn("Opponent (P1)", snapshot)


if __name__ == "__main__":
    unittest.main()
