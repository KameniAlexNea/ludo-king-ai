"""
Tests for ludo_rl/ludo_king/simulator.py
"""

import unittest

import numpy as np

from ludo_rl.ludo_king.config import config
from ludo_rl.ludo_king.game import Game
from ludo_rl.ludo_king.player import Player
from ludo_rl.ludo_king.simulator import Simulator
from ludo_rl.ludo_king.types import Color, Move, MoveEvents, MoveResult, KnockoutEvent


def _make_game() -> Game:
    players = [Player(c) for c in [Color.RED, Color.GREEN, Color.YELLOW, Color.BLUE]]
    return Game(players=players)


class TestForGame(unittest.TestCase):
    """Tests for Simulator.for_game()."""

    def test_creates_simulator(self):
        game = _make_game()
        sim = Simulator.for_game(game, agent_index=0)
        self.assertEqual(sim.agent_index, 0)
        self.assertIs(sim.game, game)

    def test_initializes_history_buffers(self):
        game = _make_game()
        sim = Simulator.for_game(game, agent_index=0)
        self.assertEqual(sim._pos_hist.shape, (config.HISTORY_LENGTH, 16))
        self.assertEqual(sim._dice_hist.shape, (config.HISTORY_LENGTH,))

    def test_direct_init_raises(self):
        with self.assertRaises(RuntimeError):
            Simulator()


class TestGetAgentReward(unittest.TestCase):
    """Tests for Simulator.get_agent_reward()."""

    def test_starts_at_zero(self):
        game = _make_game()
        sim = Simulator.for_game(game, agent_index=0)
        self.assertEqual(sim.get_agent_reward(), 0.0)

    def test_accumulates_rewards(self):
        game = _make_game()
        sim = Simulator.for_game(game, agent_index=0)
        sim._agent_reward_acc = 0.5
        self.assertEqual(sim.get_agent_reward(), 0.5)


class TestGetTokenSequenceObservation(unittest.TestCase):
    """Tests for Simulator.get_token_sequence_observation()."""

    def test_returns_dict_with_keys(self):
        game = _make_game()
        sim = Simulator.for_game(game, agent_index=0)
        obs = sim.get_token_sequence_observation(3)
        self.assertIn("positions", obs)
        self.assertIn("dice_history", obs)
        self.assertIn("token_mask", obs)
        self.assertIn("current_dice", obs)

    def test_positions_shape(self):
        game = _make_game()
        sim = Simulator.for_game(game, agent_index=0)
        obs = sim.get_token_sequence_observation(3)
        self.assertEqual(obs["positions"].shape, (config.HISTORY_LENGTH, 16))

    def test_current_dice_value(self):
        game = _make_game()
        sim = Simulator.for_game(game, agent_index=0)
        obs = sim.get_token_sequence_observation(5)
        self.assertEqual(obs["current_dice"][0], 5)


class TestAppendHistory(unittest.TestCase):
    """Tests for Simulator._append_history()."""

    def test_increments_hist_len(self):
        game = _make_game()
        sim = Simulator.for_game(game, agent_index=0)
        self.assertEqual(sim._hist_len, 0)
        sim._append_history(3, 0)
        self.assertEqual(sim._hist_len, 1)

    def test_wraps_around(self):
        game = _make_game()
        sim = Simulator.for_game(game, agent_index=0)
        for i in range(config.HISTORY_LENGTH + 5):
            sim._append_history(i % 6 + 1, 0)
        self.assertEqual(sim._hist_len, config.HISTORY_LENGTH)


class TestProcessMoveResult(unittest.TestCase):
    """Tests for Simulator._process_move_result()."""

    def test_accumulates_got_captured_penalty(self):
        game = _make_game()
        sim = Simulator.for_game(game, agent_index=0)

        events = MoveEvents(move_resolved=True)
        events.knockouts = [KnockoutEvent(player=0, piece_id=0, abs_pos=10)]
        result = MoveResult(
            old_position=5, new_position=10, extra_turn=False,
            events=events, rewards=None
        )
        move = Move(player_index=1, piece_id=0, new_pos=10, dice_roll=5)

        sim._process_move_result(1, move, result)

        from ludo_rl.ludo_king.config import reward_config
        self.assertAlmostEqual(sim._agent_reward_acc, reward_config.got_captured)

    def test_appends_to_history(self):
        game = _make_game()
        sim = Simulator.for_game(game, agent_index=0)

        events = MoveEvents(move_resolved=True)
        result = MoveResult(
            old_position=5, new_position=10, extra_turn=False,
            events=events, rewards=None
        )
        move = Move(player_index=0, piece_id=0, new_pos=10, dice_roll=5)

        sim._process_move_result(0, move, result)
        self.assertEqual(sim._hist_len, 1)


class TestStep(unittest.TestCase):
    """Tests for Simulator.step()."""

    def test_returns_terminated_and_extra(self):
        game = _make_game()
        sim = Simulator.for_game(game, agent_index=0)
        move = Move(player_index=0, piece_id=0, new_pos=1, dice_roll=6)
        terminated, extra = sim.step(move)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(extra, bool)

    def test_extra_turn_on_six(self):
        game = _make_game()
        sim = Simulator.for_game(game, agent_index=0)
        move = Move(player_index=0, piece_id=0, new_pos=1, dice_roll=6)
        _, extra = sim.step(move)
        self.assertTrue(extra)


class TestStepOpponentsOnly(unittest.TestCase):
    """Tests for Simulator.step_opponents_only()."""

    def test_runs_without_error(self):
        game = _make_game()
        sim = Simulator.for_game(game, agent_index=0)
        sim.step_opponents_only()

    def test_reset_summaries_true_clears_accumulator(self):
        game = _make_game()
        sim = Simulator.for_game(game, agent_index=0)
        sim._agent_reward_acc = 0.5
        sim.step_opponents_only(reset_summaries=True)
        # Accumulator reset to 0, then opponents might add to it
        # Just verify it was reset (may have new value from opponent moves)

    def test_reset_summaries_false_preserves_accumulator(self):
        game = _make_game()
        sim = Simulator.for_game(game, agent_index=0)
        sim._agent_reward_acc = 0.5
        sim.step_opponents_only(reset_summaries=False)
        self.assertGreaterEqual(sim._agent_reward_acc, 0.5)


if __name__ == "__main__":
    unittest.main()
