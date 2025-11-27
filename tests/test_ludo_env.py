"""
Tests for ludo_rl/ludo_env.py
"""

import unittest
from unittest.mock import patch

import numpy as np

from ludo_rl.ludo_env import LudoEnv
from ludo_rl.ludo_king.config import config, reward_config


class TestReset(unittest.TestCase):
    """Tests for LudoEnv.reset()."""

    def setUp(self):
        self.env = LudoEnv()

    def tearDown(self):
        self.env.close()

    def test_returns_observation_and_info(self):
        obs, info = self.env.reset()
        self.assertIn("positions", obs)
        self.assertIn("action_mask", info)

    def test_observation_shapes(self):
        obs, _ = self.env.reset()
        self.assertEqual(obs["positions"].shape, (config.HISTORY_LENGTH, 16))
        self.assertEqual(obs["current_dice"].shape, (1,))
        self.assertEqual(obs["dice_history"].shape, (config.HISTORY_LENGTH,))

    def test_action_mask_consistent(self):
        _, info = self.env.reset()
        np.testing.assert_array_equal(self.env.action_masks(), info["action_mask"])

    def test_no_valid_moves_flag(self):
        # Force roll of 1 (no exit from yard)
        from ludo_rl.ludo_king.game import Game
        with patch.object(Game, "roll_dice", return_value=1):
            _, info = self.env.reset()
        if not info["action_mask"].any():
            self.assertTrue(info.get("no_valid_moves", False))


class TestStep(unittest.TestCase):
    """Tests for LudoEnv.step()."""

    def setUp(self):
        self.env = LudoEnv()

    def tearDown(self):
        self.env.close()

    def test_invalid_action_returns_penalty(self):
        self.env.reset()
        self.env.move_map = {}  # No valid moves
        with patch("ludo_rl.ludo_env.Simulator.step_opponents_only"):
            _, reward, _, _, _ = self.env.step(0)
        self.assertEqual(reward, reward_config.invalid_action)

    def test_valid_action_returns_observation(self):
        from ludo_rl.ludo_king.game import Game
        with patch.object(Game, "roll_dice", return_value=6):
            _, info = self.env.reset()
            if info["action_mask"].any():
                action = int(np.argmax(info["action_mask"]))
                obs, reward, _, _, _ = self.env.step(action)
                self.assertEqual(obs["positions"].shape, (config.HISTORY_LENGTH, 16))
                self.assertIsInstance(reward, float)

    def test_win_terminates_game(self):
        self.env.reset()
        # Set all pieces to finish position
        for piece in self.env.game.players[0].pieces:
            piece.position = 57
        self.env.game.players[0].pieces[0].position = 56
        self.env.current_dice_roll = 1
        self.env._get_info()

        _, reward, terminated, truncated, _ = self.env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertGreater(reward, 0.0)

    def test_truncation_on_max_turns(self):
        self.env.reset()
        self.env.current_turn = config.MAX_TURNS
        _, _, terminated, truncated, _ = self.env.step(0)
        self.assertTrue(truncated)


class TestActionMasks(unittest.TestCase):
    """Tests for LudoEnv.action_masks()."""

    def setUp(self):
        self.env = LudoEnv()

    def tearDown(self):
        self.env.close()

    def test_returns_boolean_array(self):
        self.env.reset()
        mask = self.env.action_masks()
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(len(mask), 4)

    def test_cached_mask(self):
        self.env.reset()
        mask1 = self.env.action_masks()
        mask2 = self.env.action_masks()
        self.assertIs(mask1, mask2)


class TestRender(unittest.TestCase):
    """Tests for LudoEnv.render()."""

    def setUp(self):
        self.env = LudoEnv()

    def tearDown(self):
        self.env.close()

    def test_returns_string(self):
        self.env.reset()
        output = self.env.render()
        self.assertIsInstance(output, str)
        self.assertIn("Turn", output)


class TestCurriculum(unittest.TestCase):
    """Tests for curriculum-related methods."""

    def setUp(self):
        self.env = LudoEnv()

    def tearDown(self):
        self.env.close()

    def test_set_curriculum_timesteps(self):
        self.env.reset()
        self.env.set_curriculum_timesteps(100000)
        # No error means success


if __name__ == "__main__":
    unittest.main()
