"""
Tests for ludo_rl/ludo_king/reward.py

Tests each function in the reward module.
"""

import unittest

from ludo_rl.ludo_king.config import reward_config
from ludo_rl.ludo_king.reward import (
    compute_draw_reward,
    compute_invalid_action_penalty,
    compute_skipped_turn_penalty,
    compute_sparse_rewards,
    compute_terminal_reward,
)
from ludo_rl.ludo_king.types import KnockoutEvent, MoveEvents


class TestComputeSparseRewards(unittest.TestCase):
    """Tests for compute_sparse_rewards()."""

    def test_empty_events_returns_zeros(self):
        events = MoveEvents(move_resolved=True)
        rewards = compute_sparse_rewards(4, 0, events)
        self.assertEqual(rewards, {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0})

    def test_finish_gives_mover_reward(self):
        events = MoveEvents(finished=True, move_resolved=True)
        rewards = compute_sparse_rewards(4, 2, events)
        self.assertEqual(rewards[2], reward_config.finish)
        self.assertEqual(rewards[0], 0.0)

    def test_single_knockout_gives_capture_and_penalty(self):
        events = MoveEvents(move_resolved=True)
        events.knockouts = [KnockoutEvent(player=1, piece_id=0, abs_pos=10)]
        rewards = compute_sparse_rewards(4, 0, events)
        self.assertEqual(rewards[0], reward_config.capture)
        self.assertEqual(rewards[1], reward_config.got_captured)

    def test_multiple_knockouts_multiply_rewards(self):
        events = MoveEvents(move_resolved=True)
        events.knockouts = [
            KnockoutEvent(player=1, piece_id=0, abs_pos=10),
            KnockoutEvent(player=2, piece_id=0, abs_pos=10),
        ]
        rewards = compute_sparse_rewards(4, 0, events)
        self.assertEqual(rewards[0], reward_config.capture * 2)
        self.assertEqual(rewards[1], reward_config.got_captured)
        self.assertEqual(rewards[2], reward_config.got_captured)

    def test_finish_plus_knockout_combines(self):
        events = MoveEvents(finished=True, move_resolved=True)
        events.knockouts = [KnockoutEvent(player=1, piece_id=0, abs_pos=10)]
        rewards = compute_sparse_rewards(4, 0, events)
        expected = reward_config.finish + reward_config.capture
        self.assertEqual(rewards[0], expected)

    def test_two_players(self):
        events = MoveEvents(finished=True, move_resolved=True)
        rewards = compute_sparse_rewards(2, 1, events)
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[1], reward_config.finish)


class TestComputeTerminalReward(unittest.TestCase):
    """Tests for compute_terminal_reward()."""

    def test_rank_1_returns_win(self):
        r = compute_terminal_reward(4, 1)
        self.assertEqual(r, reward_config.win)

    def test_last_rank_returns_full_lose(self):
        r = compute_terminal_reward(4, 4)
        self.assertEqual(r, reward_config.lose)

    def test_rank_2_of_4_returns_scaled_lose(self):
        r = compute_terminal_reward(4, 2)
        expected = reward_config.lose * (1 / 3)
        self.assertAlmostEqual(r, expected, places=6)

    def test_rank_3_of_4_returns_scaled_lose(self):
        r = compute_terminal_reward(4, 3)
        expected = reward_config.lose * (2 / 3)
        self.assertAlmostEqual(r, expected, places=6)

    def test_two_player_lose(self):
        r = compute_terminal_reward(2, 2)
        self.assertEqual(r, reward_config.lose)


class TestComputeInvalidActionPenalty(unittest.TestCase):
    """Tests for compute_invalid_action_penalty()."""

    def test_returns_config_value(self):
        p = compute_invalid_action_penalty()
        self.assertEqual(p, reward_config.invalid_action)

    def test_returns_float(self):
        p = compute_invalid_action_penalty()
        self.assertIsInstance(p, float)


class TestComputeDrawReward(unittest.TestCase):
    """Tests for compute_draw_reward()."""

    def test_returns_config_value(self):
        r = compute_draw_reward()
        self.assertEqual(r, reward_config.draw)

    def test_returns_float(self):
        r = compute_draw_reward()
        self.assertIsInstance(r, float)


class TestComputeSkippedTurnPenalty(unittest.TestCase):
    """Tests for compute_skipped_turn_penalty()."""

    def test_returns_config_value(self):
        p = compute_skipped_turn_penalty()
        self.assertEqual(p, reward_config.skipped_turn)

    def test_returns_float(self):
        p = compute_skipped_turn_penalty()
        self.assertIsInstance(p, float)


if __name__ == "__main__":
    unittest.main()
