from __future__ import annotations

import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium import spaces

from ludo_rl.extractor import (
    LudoCnnExtractor,
    LudoTransformerExtractor,
    _extract_piece_positions,
)
from ludo_rl.ludo_env import format_env_state
from ludo_rl.ludo_king.config import config
from ludo_rl.ludo_king.player import Player
from ludo_rl.strategy import available, create
from ludo_rl.strategy.registry import STRATEGY_REGISTRY
from ludo_rl.strategy.rusher import RusherStrategy


class StrategyRegistryTests(unittest.TestCase):
    def test_available_matches_registry(self) -> None:
        expected = set(STRATEGY_REGISTRY.keys())
        self.assertEqual(set(available(False).keys()), expected)

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
        player.strategy = None  # ensure strategy is built from name

        from ludo_rl.ludo_king.types import Move

        piece_a = player.pieces[0]
        piece_b = player.pieces[1]
        piece_a.position = 0
        piece_b.position = 10

        valid_moves = [
            Move(player_index=0, piece_id=piece_a.piece_id, new_pos=11, dice_roll=6),
            Move(player_index=0, piece_id=piece_b.piece_id, new_pos=12, dice_roll=6),
        ]

        decision = player.choose(self._board_stack(), 6, valid_moves)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.piece_id, piece_a.piece_id)
        self.assertEqual(decision.new_pos, 11)

    def test_player_unknown_strategy_falls_back_to_random(self) -> None:
        player = Player(color=0)
        player.strategy_name = "unknown"
        player.strategy = None  # trigger unknown-name handling

        from ludo_rl.ludo_king.types import Move

        piece_a = player.pieces[0]
        piece_b = player.pieces[1]
        piece_a.position = 0
        piece_b.position = 10

        valid_moves = [
            Move(player_index=0, piece_id=piece_a.piece_id, new_pos=6, dice_roll=6),
            Move(player_index=0, piece_id=piece_b.piece_id, new_pos=16, dice_roll=6),
        ]

        decision = player.choose(self._board_stack(), 6, valid_moves)
        self.assertEqual(player.strategy_name, "random")
        self.assertIsNotNone(decision)
        self.assertIn(decision.piece_id, (piece_a.piece_id, piece_b.piece_id))


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
            game=SimpleNamespace(
                players=[
                    DummyPlayer([0, 0, 0, 0]),
                    DummyPlayer([1, 2, 3, 4]),
                    DummyPlayer([5, 6, 7, 8]),
                    DummyPlayer([9, 10, 11, 12]),
                ]
            ),
        )

        snapshot = format_env_state(dummy_env)
        self.assertIn("Turn 5/200", snapshot)
        self.assertIn("AGENT (P0)", snapshot)
        self.assertIn("Opponent (P1)", snapshot)


class LudoTransformerExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features_dim = 128
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=-50.0,
                    high=50.0,
                    shape=(10, config.PATH_LENGTH),
                    dtype=np.float32,
                ),
                "dice_roll": spaces.Box(low=0, high=5, shape=(1,), dtype=np.int64),
            }
        )
        self.extractor = LudoTransformerExtractor(
            self.observation_space, features_dim=self.features_dim
        )
        self.extractor.eval()

    def test_forward_outputs_expected_shape(self) -> None:
        batch_size = 3
        board = torch.randn(batch_size, 10, config.PATH_LENGTH, dtype=torch.float32)
        board[:, 0, :] = 0.0
        piece_indices = torch.tensor([0, 10, 20, 30], dtype=torch.long)
        for batch_idx in range(batch_size):
            board[batch_idx, 0, piece_indices] = 1.0
        observations = {
            "board": board,
            "dice_roll": torch.full((batch_size, 1), 9, dtype=torch.long),
        }
        with torch.no_grad():
            output = self.extractor(observations)

        self.assertEqual(output.shape, (batch_size, self.features_dim))
        self.assertFalse(torch.isnan(output).any().item())

    def test_extract_piece_positions_handles_stacks_and_padding(self) -> None:
        my_channel = torch.zeros(2, config.PATH_LENGTH, dtype=torch.float32)
        my_channel[0, 4] = 2  # Two pieces stacked at square 4
        my_channel[0, 10] = 1
        my_channel[0, 22] = 1

        my_channel[1, 0] = 1  # Piece still in yard
        my_channel[1, 3] = 1
        my_channel[1, 15] = 2

        positions = _extract_piece_positions(my_channel, self.extractor.board_length)

        self.assertEqual(positions.shape, (2, config.PIECES_PER_PLAYER))
        self.assertListEqual(positions[0].tolist(), [4, 4, 10, 22])
        self.assertListEqual(positions[1].tolist(), [0, 3, 15, 15])


class LudoCnnExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features_dim = 128
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=-50.0,
                    high=50.0,
                    shape=(10, config.PATH_LENGTH),
                    dtype=np.float32,
                ),
                "dice_roll": spaces.Box(low=0, high=5, shape=(1,), dtype=np.int64),
            }
        )
        self.extractor = LudoCnnExtractor(
            self.observation_space, features_dim=self.features_dim
        )
        self.extractor.eval()

    def test_forward_outputs_expected_shape(self) -> None:
        batch_size = 3
        board = torch.randn(batch_size, 10, config.PATH_LENGTH, dtype=torch.float32)
        board[:, 0, :] = 0.0
        piece_indices = torch.tensor([0, 10, 20, 30], dtype=torch.long)
        for batch_idx in range(batch_size):
            board[batch_idx, 0, piece_indices] = 1.0
        observations = {
            "board": board,
            "dice_roll": torch.full((batch_size, 1), 9, dtype=torch.long),
        }
        with torch.no_grad():
            output = self.extractor(observations)

        self.assertEqual(output.shape, (batch_size, self.features_dim))
        self.assertFalse(torch.isnan(output).any().item())

    def test_extract_piece_positions_handles_stacks_and_padding(self) -> None:
        my_channel = torch.zeros(2, config.PATH_LENGTH, dtype=torch.float32)
        my_channel[0, 4] = 2  # Two pieces stacked at square 4
        my_channel[0, 10] = 1
        my_channel[0, 22] = 1

        my_channel[1, 0] = 1  # Piece still in yard
        my_channel[1, 3] = 1
        my_channel[1, 15] = 2

        positions = _extract_piece_positions(my_channel, self.extractor.board_length)

        self.assertEqual(positions.shape, (2, config.PIECES_PER_PLAYER))
        self.assertListEqual(positions[0].tolist(), [4, 4, 10, 22])
        self.assertListEqual(positions[1].tolist(), [0, 3, 15, 15])


if __name__ == "__main__":
    unittest.main()
