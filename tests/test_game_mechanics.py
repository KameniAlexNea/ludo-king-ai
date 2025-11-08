from __future__ import annotations

import random
import unittest
from unittest import mock

import numpy as np

from ludo_rl.ludo.board import LudoBoard
from ludo_rl.ludo.config import config
from ludo_rl.ludo.game import LudoGame
from ludo_rl.ludo.moves import MoveManagement
from ludo_rl.ludo.piece import Piece
from ludo_rl.ludo.player import Player
from ludo_rl.ludo.reward import compute_move_rewards, reward_config


def make_players() -> list[Player]:
    return [Player(i) for i in range(config.NUM_PLAYERS)]


class PieceModelTests(unittest.TestCase):
    def test_is_safe_and_finished(self) -> None:
        piece = Piece(color=0, piece_id=0)
        safe_square = config.SAFE_SQUARES_ABS[0]
        self.assertTrue(piece.is_safe(safe_square))
        self.assertFalse(piece.is_safe(safe_square - 1))

        self.assertFalse(piece.is_finished())
        piece.position = 57
        self.assertTrue(piece.is_finished())

    def test_piece_move_helpers(self) -> None:
        piece = Piece(color=1, piece_id=2)
        self.assertTrue(piece.in_yard())
        self.assertIsNone(piece.destination_for_roll(3))
        self.assertEqual(piece.destination_for_roll(6), 1)
        piece.move_to(10)
        self.assertEqual(piece.position, 10)
        self.assertFalse(piece.in_yard())
        self.assertEqual(piece.destination_for_roll(2), 12)
        piece.move_to(52)
        self.assertTrue(piece.in_home_column())
        self.assertEqual(piece.destination_for_roll(5), 57)
        self.assertIsNone(piece.destination_for_roll(6))
        piece.move_to(51)
        self.assertEqual(piece.destination_for_roll(6), 57)
        piece.move_to(57)
        self.assertIsNone(piece.destination_for_roll(1))
        piece.send_home()
        self.assertTrue(piece.in_yard())


class BoardAndPlayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.players = make_players()
        self.board = LudoBoard(self.players)
        self.player = self.players[0]

    def test_absolute_and_relative_position_conversion(self) -> None:
        abs_pos = self.board.absolute_position(0, 10)
        self.assertEqual(abs_pos, 10)
        self.assertEqual(self.board.relative_position(0, abs_pos), 10)
        self.assertEqual(self.board.absolute_position(0, 0), -1)
        self.assertEqual(self.board.relative_position(0, 0), -1)

    def test_valid_moves_from_yard_and_blockade(self) -> None:
        moves = self.player.get_valid_moves(self.board, 6)
        self.assertEqual(len(moves), 4)  # all pieces can enter
        # Create a blockade at position 5 and ensure third piece cannot join
        self.player.pieces[0].position = 4
        self.player.pieces[1].position = 5
        self.player.pieces[2].position = 5
        moves = self.player.get_valid_moves(self.board, 1)
        self.assertFalse(any(move["new_pos"] == 5 for move in moves))

    def test_valid_moves_home_column_limits(self) -> None:
        self.player.pieces[0].position = 52
        moves = self.player.get_valid_moves(self.board, 3)
        self.assertTrue(
            any(
                move["piece"] is self.player.pieces[0] and move["new_pos"] == 55
                for move in moves
            )
        )
        moves = self.player.get_valid_moves(self.board, 6)
        self.assertFalse(any(move["piece"] is self.player.pieces[0] for move in moves))

    def test_make_move_capture_and_extra_turn(self) -> None:
        mover = self.player.pieces[0]
        mover.position = 10
        target_rel = 11
        target_abs = self.board.absolute_position(0, target_rel)
        opponent_piece = self.players[1].pieces[0]
        opponent_rel = self.board.relative_position(1, target_abs)
        opponent_piece.position = opponent_rel

        resolution = self.player.move_piece(self.board, mover, target_rel, dice_roll=1)
        rewards = compute_move_rewards(
            len(self.players),
            0,
            resolution.old_position,
            resolution.new_position,
            resolution.events,
        )
        self.assertTrue(resolution.events["knockouts"])
        self.assertEqual(opponent_piece.position, 0)
        self.assertGreater(rewards[0], 0)
        self.assertTrue(resolution.extra_turn)

    def test_make_move_hits_blockade(self) -> None:
        mover = self.player.pieces[0]
        new_position = 12
        abs_target = self.board.absolute_position(0, new_position)
        while self.board.is_safe_square(abs_target) or new_position <= 1:
            new_position += 1
            abs_target = self.board.absolute_position(0, new_position)
        opponent_rel = self.board.relative_position(1, abs_target)
        self.assertNotEqual(opponent_rel, -1)
        self.players[1].pieces[0].position = opponent_rel
        self.players[1].pieces[1].position = opponent_rel

        mover.position = new_position - 1
        resolution = self.player.move_piece(
            self.board, mover, new_position, dice_roll=1
        )
        rewards = compute_move_rewards(
            len(self.players),
            0,
            resolution.old_position,
            resolution.new_position,
            resolution.events,
        )
        self.assertTrue(resolution.events["hit_blockade"])
        self.assertFalse(resolution.events["move_resolved"])
        self.assertEqual(mover.position, new_position - 1)
        self.assertEqual(rewards[0], reward_config.hit_blockade)

    def test_make_move_forms_blockade(self) -> None:
        mover = self.player.pieces[0]
        mover.position = 8
        self.player.pieces[1].position = 10
        resolution = self.player.move_piece(self.board, mover, 10, dice_roll=2)
        rewards = compute_move_rewards(
            len(self.players),
            0,
            resolution.old_position,
            resolution.new_position,
            resolution.events,
        )
        self.assertTrue(resolution.events["blockades"])
        self.assertIn({"player": 0, "relative_pos": 10}, resolution.events["blockades"])
        self.assertAlmostEqual(
            rewards[0],
            reward_config.progress + reward_config.blockade,
        )


class MoveManagementCompatibilityTests(unittest.TestCase):
    def test_wrapper_delegates_to_board(self) -> None:
        players = make_players()
        manager = MoveManagement(players)
        moves = manager.get_valid_moves(0, 6)
        self.assertEqual(len(moves), 4)
        piece = players[0].pieces[0]
        outcome = manager.make_move(0, piece, 1, dice_roll=6)
        self.assertIn("events", outcome)
        self.assertTrue(outcome["extra_turn"])


class LudoGameTests(unittest.TestCase):
    def test_get_board_state_safe_zones_and_counts(self) -> None:
        game = LudoGame()
        state = game.get_board_state(agent_index=0)
        for idx in range(52, 57):
            self.assertEqual(state["safe_zones"][idx], 1)
        # All opponents start in yard
        self.assertEqual(state["opp1_pieces"][0], 4)

        # Move an agent piece to ensure my channel increments
        game.players[0].pieces[0].position = 10
        state = game.get_board_state(agent_index=0)
        self.assertEqual(state["my_pieces"][10], 1)

    def test_roll_dice_range(self) -> None:
        game = LudoGame()
        rolls = {game.roll_dice() for _ in range(50)}
        self.assertTrue(all(1 <= roll <= 6 for roll in rolls))
        self.assertEqual(len(rolls), 6)

    def test_make_move_delegation(self) -> None:
        game = LudoGame()
        piece = game.players[0].pieces[0]
        piece.position = 0
        result = game.make_move(0, piece, 1, dice_roll=6)
        self.assertIn("reward", result)
        self.assertIn("events", result)

    def test_take_turn_random_fallback(self) -> None:
        game = LudoGame()
        player = game.players[0]

        with (
            mock.patch(
                "ludo_rl.ludo.player.Player.decide",
                autospec=True,
                return_value=None,
            ),
            mock.patch.object(game.rng, "choice", wraps=game.rng.choice) as choice_mock,
        ):
            outcome = game.take_turn(0, dice_roll=6)

        self.assertFalse(outcome.skipped)
        self.assertIsNotNone(outcome.move)
        self.assertIn(outcome.move["piece"], player.pieces)
        choice_mock.assert_called()

    def test_take_turn_invalid_move_skips(self) -> None:
        game = LudoGame()
        piece = game.players[0].pieces[0]
        piece.position = 0
        invalid_move = {"piece": piece, "new_pos": 3, "dice_roll": 6}

        outcome = game.take_turn(0, dice_roll=6, move=invalid_move)

        self.assertTrue(outcome.skipped)
        self.assertIsNone(outcome.result)


class RewardComputationTests(unittest.TestCase):
    def test_reward_components_accumulation(self) -> None:
        events = {
            "move_resolved": True,
            "exited_home": True,
            "finished": True,
            "knockouts": [{"player": 1, "piece_id": 0, "abs_pos": 5}],
            "blockades": [{"player": 0, "relative_pos": 12}],
        }
        rewards = compute_move_rewards(2, 0, 0, 1, events)
        expected = (
            reward_config.progress
            + reward_config.exit_home
            + reward_config.finish
            + reward_config.capture
            + reward_config.blockade
        )
        self.assertAlmostEqual(rewards[0], expected)
        self.assertAlmostEqual(rewards[1], reward_config.got_capture)

    def test_reward_penalties_on_failed_move(self) -> None:
        events = {
            "move_resolved": False,
            "hit_blockade": True,
            "knockouts": [],
        }
        rewards = compute_move_rewards(2, 0, 5, 5, events)
        self.assertAlmostEqual(rewards[0], reward_config.hit_blockade)
        self.assertEqual(rewards[1], 0.0)


class PlayerBehaviourTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(1)
        self.player = Player(color=0)

    def test_has_won_states(self) -> None:
        self.assertFalse(self.player.has_won())
        for piece in self.player.pieces:
            piece.position = 57
        self.assertTrue(self.player.has_won())
        self.assertTrue(self.player.has_won())  # cached flag path

    def test_decide_no_valid_moves_returns_none(self) -> None:
        board = np.zeros((10, config.PATH_LENGTH), dtype=float)
        self.assertIsNone(self.player.decide(board, 3, []))

    def test_decide_resets_strategy_cache(self) -> None:
        board = np.zeros((10, config.PATH_LENGTH), dtype=float)
        piece = self.player.pieces[0]
        self.player._strategy = None
        self.player.strategy_name = "rusher"
        decision_first = self.player.decide(
            board,
            6,
            [{"piece": piece, "new_pos": 1}],
        )
        self.assertIsNotNone(decision_first)
        old_strategy = self.player._strategy
        self.player.strategy_name = "cautious"
        decision_second = self.player.decide(
            board,
            6,
            [{"piece": piece, "new_pos": 1}],
        )
        self.assertIsNotNone(decision_second)
        self.assertEqual(getattr(self.player._strategy, "name", None), "cautious")
        self.assertIsNot(self.player._strategy, old_strategy)

    def test_decide_strategy_returns_none_fallback(self) -> None:
        class NullStrategy:
            name = "null"

            def select_move(self, ctx):  # noqa: D401 - simple stub
                return None

        board = np.zeros((10, config.PATH_LENGTH), dtype=float)
        piece = self.player.pieces[0]
        valid_moves = [{"piece": piece, "new_pos": 1}]
        self.player.strategy_name = "null"
        self.player._strategy = NullStrategy()
        choice = self.player.decide(board, 6, valid_moves)
        self.assertIn(choice, valid_moves)

    def test_decide_invalid_selected_move_fallback(self) -> None:
        class InvalidStrategy:
            name = "invalid"

            def select_move(self, ctx):
                from ludo_rl.strategy.types import MoveOption

                return MoveOption(
                    piece_id=3,
                    current_pos=0,
                    new_pos=10,
                    dice_roll=6,
                    progress=0,
                    distance_to_goal=0,
                    can_capture=False,
                    capture_count=0,
                    enters_home=False,
                    enters_safe_zone=False,
                    forms_blockade=False,
                    extra_turn=False,
                    risk=0.0,
                    leaving_safe_zone=False,
                )

        board = np.zeros((10, config.PATH_LENGTH), dtype=float)
        piece = self.player.pieces[0]
        valid_moves = [{"piece": piece, "new_pos": 1}]
        self.player.strategy_name = "invalid"
        self.player._strategy = InvalidStrategy()
        choice = self.player.decide(board, 6, valid_moves)
        self.assertIn(choice, valid_moves)


if __name__ == "__main__":
    unittest.main()
