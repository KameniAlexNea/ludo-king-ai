import unittest
from unittest.mock import Mock, patch

import numpy as np
from ludo_engine.models import AIDecisionContext

from ludo_rl.ludo_env.observation import ObservationBuilder
from ludo_rl.strategies.frozen_policy_strategy import FrozenPolicyStrategy


class TestFrozenPolicyStrategy(unittest.TestCase):
    def setUp(self):
        self.obs_builder = Mock(spec=ObservationBuilder)
        self.policy = Mock()

    def test_init(self):
        strategy = FrozenPolicyStrategy(
            self.policy, self.obs_builder, deterministic=True
        )
        self.assertEqual(strategy.policy, self.policy)
        self.assertEqual(strategy.obs_builder, self.obs_builder)
        self.assertTrue(strategy.deterministic)

    def test_build_action_mask(self):
        valid_moves = [Mock(token_id=0), Mock(token_id=2)]
        mask = FrozenPolicyStrategy._build_action_mask(valid_moves)
        expected = np.array([1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(mask, expected)

    def test_decide_no_policy_random(self):
        strategy = FrozenPolicyStrategy(None, self.obs_builder)
        context = Mock(spec=AIDecisionContext)
        context.valid_moves = [Mock(token_id=1), Mock(token_id=3)]
        with patch("random.choice") as mock_choice:
            mock_choice.return_value = Mock(token_id=1)
            result = strategy.decide(context)
            self.assertEqual(result, 1)

    @patch("torch.as_tensor")
    @patch("torch.no_grad")
    def test_decide_with_policy_deterministic(self, mock_no_grad, mock_tensor):
        strategy = FrozenPolicyStrategy(
            self.policy, self.obs_builder, deterministic=True
        )
        context = Mock(spec=AIDecisionContext)
        context.valid_moves = [Mock(token_id=0)]
        context.current_situation = Mock()
        context.current_situation.turn_count = 5
        context.current_situation.dice_value = 3

        self.obs_builder.build.return_value = np.array([1.0, 2.0])
        mock_tensor.return_value = Mock()

        dist = Mock()
        dist.distribution.probs.squeeze.return_value.cpu.return_value.numpy.return_value = np.array(
            [0.1, 0.9, 0.0, 0.0]
        )
        self.policy.get_distribution.return_value = dist

        result = strategy.decide(context)
        self.assertEqual(result, 0)  # argmax of masked probs

    @patch("torch.as_tensor")
    @patch("torch.no_grad")
    def test_decide_with_policy_stochastic(self, mock_no_grad, mock_tensor):
        strategy = FrozenPolicyStrategy(
            self.policy, self.obs_builder, deterministic=False
        )
        context = Mock(spec=AIDecisionContext)
        context.valid_moves = [Mock(token_id=0)]
        context.current_situation = Mock()
        context.current_situation.turn_count = 5
        context.current_situation.dice_value = 3

        self.obs_builder.build.return_value = np.array([1.0, 2.0])
        mock_tensor.return_value = Mock()

        dist = Mock()
        dist.distribution.probs.squeeze.return_value.cpu.return_value.numpy.return_value = np.array(
            [0.1, 0.9, 0.0, 0.0]
        )
        self.policy.get_distribution.return_value = dist

        with patch("numpy.random.choice") as mock_choice:
            mock_choice.return_value = 1
            result = strategy.decide(context)
            self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
