from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np

from ludo_rl.ludo.config import config
from ludo_rl.ludo.reward import reward_config
from ludo_rl.ludo_env import LudoEnv
from ludo_rl.ludo.simulator import GameSimulator as RealGameSimulator


class LudoEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = LudoEnv()

    def tearDown(self) -> None:
        self.env.close()

    def test_reset_provides_valid_observation_and_mask(self) -> None:
        obs, info = self.env.reset()
        self.assertEqual(obs["board"].shape, (10, config.PATH_LENGTH))
        self.assertEqual(obs["dice_roll"].shape, (1,))
        self.assertTrue(info["action_mask"].any())
        np.testing.assert_array_equal(self.env.action_masks(), info["action_mask"])

    def test_reset_handles_initial_no_moves_loop(self) -> None:
        def factory(agent_index: int):
            simulator = RealGameSimulator(agent_index)
            simulator.game.roll_dice = Mock(side_effect=[1, 6, 6])
            simulator.step_opponents_only = Mock(return_value=[0.0] * config.NUM_PLAYERS)
            return simulator

        with patch("ludo_rl.ludo_env.GameSimulator", side_effect=factory):
            _, info = self.env.reset()

        step_mock = self.env.simulator.step_opponents_only
        self.assertGreaterEqual(step_mock.call_count, 1)
        self.assertTrue(info["action_mask"].any())

    def test_step_invalid_action_penalises_agent(self) -> None:
        self.env.reset()
        with patch.object(
            self.env.simulator,
            "step_opponents_only",
            return_value=[0.0] * config.NUM_PLAYERS,
        ), patch.object(self.env.simulator.game, "roll_dice", return_value=6):
            self.env.move_map = {}
            obs, reward, terminated, truncated, info = self.env.step(0)
        self.assertEqual(reward, reward_config.lose)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(obs["board"].shape, (10, config.PATH_LENGTH))
        self.assertIsInstance(info["action_mask"], bool)

    def test_step_valid_action_returns_next_observation(self) -> None:
        _, info = self.env.reset()
        action = int(np.where(info["action_mask"])[0][0])
        obs, reward, terminated, truncated, next_info = self.env.step(action)
        self.assertEqual(obs["board"].shape, (10, config.PATH_LENGTH))
        self.assertIsInstance(reward, float)
        self.assertIn("action_mask", next_info)
        self.assertFalse(terminated and truncated)

    def test_step_handles_win_condition(self) -> None:
        self.env.reset()
        piece = self.env.simulator.game.players[0].pieces[0]
        piece.position = 56
        self.env.current_dice_roll = 1
        for other in self.env.simulator.game.players[0].pieces[1:]:
            other.position = 57
        self.env._get_info()
        obs, reward, terminated, truncated, info = self.env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertIn("final_rank", info)
        self.assertGreater(reward, 0.0)
        self.assertEqual(obs["board"].shape, (10, config.PATH_LENGTH))

    def test_render_returns_summary(self) -> None:
        self.env.reset()
        snapshot = self.env.render()
        self.assertIn("Turn", snapshot)


if __name__ == "__main__":
    unittest.main()
