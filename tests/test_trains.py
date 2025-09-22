import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from ludo_rl.trains.lr_utils import linear_interp, apply_linear_lr
from ludo_rl.trains.training_args import parse_args
from ludo_rl.config import TrainConfig
import argparse


class TestLrUtils(unittest.TestCase):
    def test_linear_interp(self):
        result = linear_interp(0.0, 1.0, 0.5)
        self.assertEqual(result, 0.5)

    def test_linear_interp_bounds(self):
        result = linear_interp(0.0, 1.0, 1.5)
        self.assertEqual(result, 1.0)
        result = linear_interp(0.0, 1.0, -0.5)
        self.assertEqual(result, 0.0)

    @patch('sb3_contrib.MaskablePPO')
    def test_apply_linear_lr(self, mock_ppo):
        model = Mock()
        model.policy.optimizer.param_groups = [{'lr': 0.1}]
        new_lr = apply_linear_lr(model, 0.1, 0.01, 0.5)
        self.assertEqual(new_lr, 0.055)
        self.assertEqual(model.policy.optimizer.param_groups[0]['lr'], 0.055)


class TestTrainingArgs(unittest.TestCase):
    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_args_default(self, mock_parse):
        mock_parse.return_value = argparse.Namespace(
            total_steps=10000,
            n_envs=4,
            logdir="logs",
            model_dir="models",
            eval_freq=1000,
            eval_games=100,
            checkpoint_freq=1000,
            checkpoint_prefix="checkpoint",
            eval_baselines="random,killer",
            learning_rate=0.001,
            n_steps=2048,
            batch_size=64,
            ent_coef=0.01,
            max_turns=1000,
            imitation_enabled=False,
            imitation_strategies="",
            imitation_steps=0,
            imitation_batch_size=64,
            imitation_epochs=1,
            imitation_entropy_boost=0.0,
            entropy_coef_initial=0.01,
            entropy_coef_final=0.001,
            entropy_anneal_steps=10000,
            capture_scale_initial=1.0,
            capture_scale_final=0.5,
            capture_scale_anneal_steps=10000,
            lr_final=0.00025,
            lr_anneal_enabled=False,
            anneal_log_freq=1000,
            env_type="classic",
            hybrid_switch_rate=0.1
        )
        config = parse_args()
        self.assertIsInstance(config, TrainConfig)
        self.assertEqual(config.total_steps, 10000)

    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_args_custom(self, mock_parse):
        mock_parse.return_value = argparse.Namespace(
            total_steps=20000,
            n_envs=8,
            logdir="custom_logs",
            model_dir="custom_models",
            eval_freq=2000,
            eval_games=200,
            checkpoint_freq=2000,
            checkpoint_prefix="custom_checkpoint",
            eval_baselines="killer,defensive",
            learning_rate=0.0005,
            n_steps=4096,
            batch_size=128,
            ent_coef=0.02,
            max_turns=2000,
            imitation_enabled=True,
            imitation_strategies="random,killer",
            imitation_steps=5000,
            imitation_batch_size=128,
            imitation_epochs=2,
            imitation_entropy_boost=0.01,
            entropy_coef_initial=0.02,
            entropy_coef_final=0.002,
            entropy_anneal_steps=20000,
            capture_scale_initial=1.5,
            capture_scale_final=0.75,
            capture_scale_anneal_steps=20000,
            lr_final=0.000125,
            lr_anneal_enabled=True,
            anneal_log_freq=2000,
            env_type="hybrid",
            hybrid_switch_rate=0.2
        )
        config = parse_args()
        self.assertEqual(config.total_steps, 20000)
        self.assertEqual(config.imitation_enabled, True)
        self.assertEqual(config.env_type, "hybrid")


if __name__ == '__main__':
    unittest.main()