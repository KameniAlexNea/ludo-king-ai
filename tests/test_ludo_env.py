from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from ludo_rl.ludo_env import LudoEnv
from ludo_rl.ludo_king.config import config, reward_config
from ludo_rl.ludo_king.game import Game


class LudoEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = LudoEnv()

    def tearDown(self) -> None:
        self.env.close()

    def test_reset_provides_valid_observation_and_mask(self) -> None:
        """Test reset returns valid observation shape and consistent mask.

        NOTE: With the God Step fix, reset no longer loops until the agent
        has valid moves. The mask may be all-False if the first roll gives
        no valid moves (e.g., rolled 1 when all pieces in yard).
        """
        obs, info = self.env.reset()
        self.assertEqual(obs["positions"].shape, (config.HISTORY_LENGTH, 16))
        self.assertEqual(obs["current_dice"].shape, (1,))
        # Mask should be consistent with env.action_masks()
        np.testing.assert_array_equal(self.env.action_masks(), info["action_mask"])
        # If no valid moves, info should indicate this
        if not info["action_mask"].any():
            self.assertTrue(info.get("no_valid_moves", False))

    def test_reset_no_valid_moves_returns_immediately(self) -> None:
        """Test that reset returns immediately even with no valid moves.

        With the God Step fix, reset no longer loops. If the first roll
        gives no valid moves (e.g., rolled 1 when pieces in yard), it
        returns with an all-False mask and no_valid_moves=True.
        """

        def fixed_roll_one(self):
            return 1  # No valid moves from yard with roll of 1

        with patch.object(Game, "roll_dice", fixed_roll_one):
            _, info = self.env.reset()

        # When rolled 1 with all pieces in yard, there are no valid moves
        self.assertFalse(info["action_mask"].any())
        self.assertTrue(info.get("no_valid_moves", False))

    def test_step_invalid_action_penalises_agent(self) -> None:
        """Test that selecting an invalid action returns the correct penalty.

        When the move_map is empty (no valid actions), selecting any action
        should return the invalid_action penalty.
        """
        self.env.reset()

        def fixed_roll(self):
            return 6

        with (
            patch("ludo_rl.ludo_env.Simulator.step_opponents_only", return_value=None),
            patch.object(Game, "roll_dice", fixed_roll),
        ):
            self.env.move_map = {}
            obs, reward, terminated, truncated, info = self.env.step(0)
        # When move_map is empty, the agent gets invalid_action penalty
        self.assertEqual(reward, reward_config.invalid_action)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(obs["positions"].shape, (config.HISTORY_LENGTH, 16))
        # action_mask should be a sequence of booleans (e.g., list or ndarray)
        self.assertIsInstance(info["action_mask"], (list, np.ndarray))
        self.assertTrue(
            all(isinstance(x, (bool, np.bool_)) for x in info["action_mask"])
        )

    def test_step_valid_action_returns_next_observation(self) -> None:
        """Test that a valid action returns proper observation.

        We use a fixed dice roll of 6 to ensure the agent can exit
        from the yard and have valid moves.
        """

        def fixed_roll_six(self):
            return 6  # Guarantees exit from yard

        with patch.object(Game, "roll_dice", fixed_roll_six):
            _, info = self.env.reset()
            # With roll of 6, there should be valid moves (exit from yard)
            self.assertTrue(
                info["action_mask"].any(), "Should have valid moves with roll of 6"
            )
            action = int(np.where(info["action_mask"])[0][0])
            obs, reward, terminated, truncated, next_info = self.env.step(action)

        self.assertEqual(obs["positions"].shape, (config.HISTORY_LENGTH, 16))
        self.assertIsInstance(reward, float)
        self.assertIn("action_mask", next_info)
        self.assertFalse(terminated and truncated)

    def test_step_handles_win_condition(self) -> None:
        self.env.reset()
        piece = self.env.game.players[0].pieces[0]
        piece.position = 56
        self.env.current_dice_roll = 1
        for other in self.env.game.players[0].pieces[1:]:
            other.position = 57
        self.env._get_info()
        obs, reward, terminated, truncated, info = self.env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        if os.getenv("RANK_ENV") == "1":
            self.assertIn("final_rank", info)
        self.assertGreater(reward, 0.0)
        self.assertEqual(obs["positions"].shape, (config.HISTORY_LENGTH, 16))

    def test_render_returns_summary(self) -> None:
        self.env.reset()
        snapshot = self.env.render()
        self.assertIn("Turn", snapshot)


if __name__ == "__main__":
    unittest.main()
