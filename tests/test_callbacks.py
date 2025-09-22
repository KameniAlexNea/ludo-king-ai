import unittest
from unittest.mock import Mock, patch

import numpy as np

from ludo_rl.callbacks.annealing import AnnealingCallback
from ludo_rl.callbacks.curriculum import ProgressCallback
from ludo_rl.callbacks.eval_baselines import SimpleBaselineEvalCallback
from ludo_rl.callbacks.hybrid_switch import HybridSwitchCallback
from ludo_rl.config import TrainConfig, EnvConfig


class TestAnnealingCallback(unittest.TestCase):
    def setUp(self):
        self.train_cfg = TrainConfig()
        self.callback = AnnealingCallback(self.train_cfg)

    def test_init(self):
        self.assertEqual(self.callback.train_cfg, self.train_cfg)

    @patch('stable_baselines3.common.callbacks.BaseCallback._on_step')
    def test_on_step_entropy_annealing(self, mock_super_on_step):
        mock_super_on_step.return_value = True
        self.callback.num_timesteps = 1000
        self.callback.model = Mock()
        self.callback.model.ent_coef = 0.1
        self.train_cfg.entropy_anneal_steps = 2000
        self.train_cfg.entropy_coef_initial = 0.1
        self.train_cfg.entropy_coef_final = 0.02

        result = self.callback._on_step()
        self.assertTrue(result)
        # Check if ent_coef was updated
        expected_ent = 0.1 + 0.5 * (0.02 - 0.1)
        self.assertAlmostEqual(self.callback.model.ent_coef, expected_ent)

    @patch('stable_baselines3.common.callbacks.BaseCallback._on_step')
    def test_on_step_capture_scale_annealing(self, mock_super_on_step):
        mock_super_on_step.return_value = True
        self.callback.num_timesteps = 1000
        self.callback.model = Mock()
        self.callback.model.env = Mock()
        self.callback.model.env.envs = [Mock()]
        self.callback.model.env.envs[0].cfg = EnvConfig()
        self.train_cfg.capture_scale_anneal_steps = 2000
        self.train_cfg.capture_scale_initial = 1.3
        self.train_cfg.capture_scale_final = 1.0

        result = self.callback._on_step()
        self.assertTrue(result)
        expected_scale = 1.3 + 0.5 * (1.0 - 1.3)
        self.assertAlmostEqual(self.callback.model.env.envs[0].cfg.reward.capture_reward_scale, expected_scale)


class TestProgressCallback(unittest.TestCase):
    def test_init(self):
        callback = ProgressCallback(total_timesteps=10000, update_freq=1000)
        self.assertEqual(callback.total, 10000)
        self.assertEqual(callback.freq, 1000)

    @patch('stable_baselines3.common.callbacks.BaseCallback._on_step')
    def test_on_step_update_progress(self, mock_super_on_step):
        mock_super_on_step.return_value = True
        callback = ProgressCallback(total_timesteps=10000, update_freq=1000)
        callback.num_timesteps = 5000
        # Just test it doesn't crash
        result = callback._on_step()
        self.assertTrue(result)


class TestSimpleBaselineEvalCallback(unittest.TestCase):
    def setUp(self):
        self.baselines = ["random", "killer"]
        self.callback = SimpleBaselineEvalCallback(self.baselines, n_games=10, eval_freq=1000)

    def test_init(self):
        self.assertEqual(self.callback.baselines, self.baselines)
        self.assertEqual(self.callback.n_games, 10)
        self.assertEqual(self.callback.eval_freq, 1000)

    @patch('stable_baselines3.common.callbacks.BaseCallback._on_step')
    def test_on_step_no_eval(self, mock_super_on_step):
        mock_super_on_step.return_value = True
        self.callback.num_timesteps = 500
        result = self.callback._on_step()
        self.assertTrue(result)

    @patch('ludo_rl.callbacks.eval_baselines.build_opponent_triplets')
    @patch('ludo_rl.callbacks.eval_baselines.MoveUtils.get_action_mask_for_env')
    def test_run_eval(self, mock_get_action_mask, mock_build_triplets):
        # Mock triplets
        mock_build_triplets.return_value = [["opp1", "opp2", "opp3"]]
        # Mock action mask
        mock_get_action_mask.return_value = [1, 0, 0, 0]  # Some mask
        # Mock model predict
        self.callback.model = Mock()
        self.callback.model.predict.return_value = (0, None)  # action 0
        # Mock logger on model
        self.callback.model.logger = Mock()
        # Mock eval_env
        mock_base_env = Mock()
        mock_base_env.reset.return_value = (np.array([0.0]), {})
        mock_base_env.step.return_value = (np.array([0.0]), 1.0, True, False, {"finished_tokens": 4, "captured_opponents": 2, "captured_by_opponents": 1, "episode_capture_ops_available": 5, "episode_capture_ops_taken": 3, "episode_finish_ops_available": 4, "episode_finish_ops_taken": 2, "episode_home_exit_ops_available": 3, "episode_home_exit_ops_taken": 1})
        mock_base_env.game.winner = Mock()
        mock_base_env.game.winner.color = "red"
        mock_base_env.game.game_over = True
        mock_base_env.agent_color = "red"
        self.callback.eval_env.envs = [mock_base_env]
        self.callback.eval_env.normalize_obs = Mock(return_value=np.array([0.0]))
        # Run eval
        self.callback._run_eval()
        # Assert logger was called
        self.callback.model.logger.record.assert_called()


class TestHybridSwitchCallback(unittest.TestCase):
    def test_init(self):
        callback = HybridSwitchCallback(switch_step=5000)
        self.assertEqual(callback.switch_step, 5000)

    @patch('stable_baselines3.common.callbacks.BaseCallback._on_step')
    def test_on_step_before_switch(self, mock_super_on_step):
        mock_super_on_step.return_value = True
        callback = HybridSwitchCallback(switch_step=5000)
        callback.num_timesteps = 3000
        result = callback._on_step()
        self.assertTrue(result)

    @patch('stable_baselines3.common.callbacks.BaseCallback._on_step')
    def test_on_step_after_switch(self, mock_super_on_step):
        mock_super_on_step.return_value = True
        callback = HybridSwitchCallback(switch_step=5000)
        callback.num_timesteps = 6000
        result = callback._on_step()
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()