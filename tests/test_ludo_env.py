import unittest
from unittest.mock import Mock, patch

import numpy as np
from ludo_engine.core import LudoGame
from ludo_engine.models import ALL_COLORS, PlayerColor

from ludo_rl.config import EnvConfig
from ludo_rl.ludo_env.ludo_env_base import LudoRLEnvBase
from ludo_rl.ludo_env.observation import ObservationBuilder


class MockLudoRLEnvBase(LudoRLEnvBase):
    def attach_opponents(self, options=None):
        pass  # Mock implementation


class TestObservationBuilder(unittest.TestCase):
    def setUp(self):
        self.cfg = EnvConfig()
        self.game = LudoGame(ALL_COLORS)
        self.agent_color = PlayerColor.RED
        self.builder = ObservationBuilder(self.cfg, self.game, self.agent_color)

    def test_compute_size(self):
        size = self.builder.compute_size()
        # 4 agent + 12 opp + 4 progress + 6 dice + 1 turn = 27
        expected = 4 + 12 + 4 + 6 + 1
        self.assertEqual(size, expected)

    def test_normalize_pos_home(self):
        result = self.builder.normalize_pos(-1)  # HOME_POSITION
        self.assertEqual(result, -1.0)

    def test_normalize_pos_main_board(self):
        result = self.builder.normalize_pos(0)
        self.assertIsInstance(result, float)

    def test_token_progress_home(self):
        result = self.builder.token_progress(-1, 0)
        self.assertEqual(result, 0.0)

    def test_build(self):
        obs = self.builder.build(turn_counter=5, dice=3)
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(len(obs), self.builder.size)


class TestLudoRLEnvBase(unittest.TestCase):
    def setUp(self):
        self.cfg = EnvConfig()
        self.env = MockLudoRLEnvBase(self.cfg)

    def test_init(self):
        self.assertEqual(self.env.cfg, self.cfg)
        self.assertEqual(self.env.agent_color, PlayerColor.RED)
        self.assertIsInstance(self.env.game, LudoGame)
        self.assertIsInstance(self.env.obs_builder, ObservationBuilder)

    @patch("ludo_rl.ludo_env.ludo_env_base.LudoRLEnvBase._ensure_agent_turn")
    @patch("ludo_rl.ludo_env.ludo_env_base.LudoRLEnvBase._roll_agent_dice")
    def test_reset(self, mock_roll, mock_ensure):
        mock_roll.return_value = (3, [])
        obs, info = self.env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertIn("episode", info)
        mock_ensure.assert_called_once()
        mock_roll.assert_called_once()

    @patch("ludo_rl.ludo_env.ludo_env_base.LudoRLEnvBase._simulate_single_opponent")
    def test_ensure_agent_turn(self, mock_simulate):
        # Mock game to have agent as current player
        self.env.game.get_current_player = Mock()
        self.env.game.get_current_player.return_value.color = self.env.agent_color
        self.env._ensure_agent_turn()
        # Should not simulate since it's agent's turn
        mock_simulate.assert_not_called()

    def test_step_game_over(self):
        self.env.game.game_over = True
        obs, reward, terminated, truncated, info = self.env.step(0)
        self.assertTrue(terminated)
        self.assertEqual(reward, 0.0)

    @patch("ludo_rl.ludo_env.ludo_env_base.LudoRLEnvBase._roll_agent_dice")
    @patch("ludo_rl.ludo_env.ludo_env_base.LudoRLEnvBase._simulate_single_opponent")
    @patch("ludo_rl.ludo_env.ludo_env_base.LudoRLEnvBase._ensure_agent_turn")
    def test_step_no_valid_moves(self, mock_ensure, mock_simulate, mock_roll):
        self.env._pending_dice = 1
        self.env._pending_valid = []  # No valid moves
        # Mock current player to be agent to prevent opponent simulation loop
        mock_player = Mock()
        mock_player.color = self.env.agent_color
        self.env.game.get_current_player = Mock(return_value=mock_player)
        # Mock roll to prevent calling at end
        mock_roll.return_value = (1, [])
        obs, reward, terminated, truncated, info = self.env.step(0)
        self.assertIsInstance(obs, np.ndarray)
        self.assertIn("illegal_action", info)


if __name__ == "__main__":
    unittest.main()
