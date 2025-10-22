"""Functional tests for evaluation system in realistic scenarios."""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.models.config import EnvConfig
from src.models.eval_utils import evaluate_against_many


class TestEvaluationFunctionality(unittest.TestCase):
    """Functional tests for evaluation system."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_cfg = EnvConfig(max_turns=30, seed=42)  # Short games for testing
        self.games = 3  # Few games for quick testing

    def test_multi_opponent_evaluation(self):
        """Test evaluation against multiple opponents."""
        mock_model = Mock()

        opponents = ["random", "probabilistic_v2", "killer"]
        games_per_opponent = 2

        with patch('src.models.eval_utils.evaluate_against') as mock_evaluate:
            # Mock different performance against different opponents
            mock_evaluate.side_effect = [
                Mock(opponent="random", win_rate=0.6, avg_reward=5.0, avg_length=25.0),
                Mock(opponent="probabilistic_v2", win_rate=0.3, avg_reward=-2.0, avg_length=30.0),
                Mock(opponent="killer", win_rate=0.1, avg_reward=-10.0, avg_length=20.0),
            ]

            results = evaluate_against_many(mock_model, opponents, games_per_opponent, self.base_cfg, deterministic=True)

            # Verify results for each opponent
            self.assertEqual(len(results), len(opponents))
            self.assertEqual(results[0].opponent, "random")
            self.assertEqual(results[0].win_rate, 0.6)
            self.assertEqual(results[1].opponent, "probabilistic_v2")
            self.assertEqual(results[1].win_rate, 0.3)
            self.assertEqual(results[2].opponent, "killer")
            self.assertEqual(results[2].win_rate, 0.1)

    def test_evaluation_statistics_calculation(self):
        """Test that evaluation statistics are calculated correctly."""
        mock_model = Mock()

        # Mock evaluation results
        with patch('src.models.eval_utils.evaluate_against') as mock_evaluate:
            stats = Mock()
            stats.opponent = "random"
            stats.episodes = 10
            stats.total_reward = 50.0  # Average reward of 5.0
            stats.wins = 6
            stats.losses = 3
            stats.draws = 1
            stats.lengths = 250  # Average length of 25.0
            stats.win_rate = 0.6
            stats.avg_reward = 5.0
            stats.avg_length = 25.0
            stats.as_dict.return_value = {
                "opponent": "random",
                "episodes": 10.0,
                "win_rate": 0.6,
                "loss_rate": 0.3,
                "draw_rate": 0.1,
                "avg_reward": 5.0,
                "avg_length": 25.0,
            }

            mock_evaluate.return_value = stats

            result = evaluate_against_many(mock_model, ["random"], 10, self.base_cfg, deterministic=True)[0]

            # Verify statistics
            self.assertEqual(result.win_rate, 0.6)
            self.assertEqual(result.avg_reward, 5.0)
            self.assertEqual(result.avg_length, 25.0)

            # Verify as_dict output
            result_dict = result.as_dict()
            self.assertEqual(result_dict["win_rate"], 0.6)
            self.assertEqual(result_dict["loss_rate"], 0.3)
            self.assertEqual(result_dict["draw_rate"], 0.1)

    def test_evaluation_aggregates_multiple_opponents(self):
        """Test that evaluation correctly aggregates results from multiple opponents."""
        mock_model = Mock()

        opponents = ["weak", "medium", "strong"]
        games_per_opponent = 5

        with patch('src.models.eval_utils.evaluate_against') as mock_evaluate:
            # Mock varying performance levels
            mock_evaluate.side_effect = [
                Mock(opponent="weak", win_rate=0.8, avg_reward=8.0),      # Good against weak
                Mock(opponent="medium", win_rate=0.5, avg_reward=0.0),    # Even against medium
                Mock(opponent="strong", win_rate=0.2, avg_reward=-5.0),   # Poor against strong
            ]

            results = evaluate_against_many(mock_model, opponents, games_per_opponent, self.base_cfg, deterministic=True)

            # Verify we get results for all opponents
            self.assertEqual(len(results), 3)

            # Verify opponent names
            opponent_names = [r.opponent for r in results]
            self.assertEqual(opponent_names, opponents)

            # Verify win rates reflect expected difficulty
            win_rates = [r.win_rate for r in results]
            self.assertGreater(win_rates[0], win_rates[1], "Should perform better against weak opponent")
            self.assertGreater(win_rates[1], win_rates[2], "Should perform better against medium than strong opponent")

    def test_deterministic_vs_stochastic_settings(self):
        """Test that deterministic and stochastic settings are passed correctly."""
        mock_model = Mock()

        with patch('src.models.eval_utils.evaluate_against') as mock_evaluate:
            mock_evaluate.return_value = Mock(opponent="test", win_rate=0.5)

            # Test deterministic
            evaluate_against_many(mock_model, ["test"], 1, self.base_cfg, deterministic=True)
            mock_evaluate.assert_called_with(mock_model, "test", 1, self.base_cfg, True)

            # Reset mock
            mock_evaluate.reset_mock()

            # Test stochastic
            evaluate_against_many(mock_model, ["test"], 1, self.base_cfg, deterministic=False)
            mock_evaluate.assert_called_with(mock_model, "test", 1, self.base_cfg, False)


if __name__ == "__main__":
    unittest.main()