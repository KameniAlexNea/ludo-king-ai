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
    pass


if __name__ == '__main__':
    unittest.main()