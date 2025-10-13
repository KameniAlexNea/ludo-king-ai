import unittest
from unittest.mock import Mock, patch

import numpy as np
from ludo_engine.models import GameConstants

from ludo_rl.config import EnvConfig
from ludo_rl.utils.move_utils import MoveUtils
from ludo_rl.utils.opponents import build_opponent_combinations, sample_opponents
from ludo_rl.utils.reward_calculator import RewardCalculator
from ludo_rl.utils.risk_opportunity import RiskOpportunityCalculator, SimpleROWeights


class TestMoveUtils(unittest.TestCase):
    def test_action_mask_no_valid_moves(self):
        mask = MoveUtils.action_masks([])
        expected = np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=bool)
        np.testing.assert_array_equal(mask, expected)

    def test_action_mask_with_valid_moves(self):
        valid_moves = [Mock(token_id=0), Mock(token_id=2)]
        mask = MoveUtils.action_masks(valid_moves)
        expected = np.array([True, False, True, False])
        np.testing.assert_array_equal(mask, expected)

    def test_get_action_mask_for_env_with_valid_moves(self):
        env = Mock()
        env.pending_valid_moves = [Mock(token_id=1)]
        mask = MoveUtils.get_action_mask_for_env(env)
        expected = np.array([False, True, False, False])
        np.testing.assert_array_equal(mask, expected)

    def test_get_action_mask_for_env_exception(self):
        env = Mock()
        env._pending_valid = None
        with patch.object(MoveUtils, "action_mask", side_effect=Exception):
            mask = MoveUtils.get_action_mask_for_env(env)
            expected = np.ones(GameConstants.TOKENS_PER_PLAYER, dtype=bool)
            np.testing.assert_array_equal(mask, expected)


class TestOpponents(unittest.TestCase):
    def test_sample_opponents_basic(self):
        candidates = ["random", "killer", "defensive"]
        progress = 0.5
        boundaries = [0.25, 0.6, 0.9]
        rgn = np.random.default_rng(42)
        opponents = sample_opponents(candidates, progress, boundaries, rgn)
        self.assertEqual(len(opponents), 3)
        self.assertTrue(all(o in candidates for o in opponents))

    def test_sample_opponents_few_candidates(self):
        candidates = ["random"]
        progress = None
        boundaries = [0.25, 0.6, 0.9]
        rgn = np.random.default_rng(42)
        opponents = sample_opponents(candidates, progress, boundaries, rgn)
        self.assertEqual(len(opponents), 1)

    def test_sample_opponents_progress_stages(self):
        candidates = ["random", "killer", "defensive", "probabilistic"]
        boundaries = [0.25, 0.6, 0.9]
        rgn = np.random.default_rng(42)
        # Test early progress
        opponents = sample_opponents(candidates, 0.1, boundaries, rgn)
        self.assertEqual(len(opponents), 3)
        # Test mid progress
        opponents = sample_opponents(candidates, 0.5, boundaries, rgn)
        self.assertEqual(len(opponents), 3)
        # Test late progress
        opponents = sample_opponents(candidates, 0.95, boundaries, rgn)
        self.assertEqual(len(opponents), 3)

    def test_build_opponent_triplets_few_baselines(self):
        baselines = ["random"]
        n_games = 2
        triplets = build_opponent_combinations(baselines, n_games)
        self.assertEqual(len(triplets), n_games)
        for triplet in triplets:
            self.assertEqual(len(triplet), 3)
            self.assertEqual(triplet, ["random", "random", "random"])

    def test_build_opponent_triplets_empty_baselines(self):
        baselines = []
        n_games = 1
        triplets = build_opponent_combinations(baselines, n_games)
        self.assertEqual(len(triplets), 0)
        self.assertEqual(triplets, [])


class TestRewardCalculator(unittest.TestCase):
    def setUp(self):
        self.cfg = EnvConfig()
        self.calc = RewardCalculator()

    def test_compute_basic_reward(self):
        res = Mock()
        res.captured_tokens = []
        res.finished_token = False
        res.new_position = 5
        res.old_position = 0
        illegal = False
        game_over = False
        reward = self.calc.compute(res, illegal, self.cfg, game_over)
        self.assertIsInstance(reward, float)
        # Should be time_penalty
        self.assertEqual(reward, self.cfg.reward.time_penalty)

    def test_compute_with_capture(self):
        res = Mock()
        res.captured_tokens = [Mock()]  # One capture
        res.finished_token = False
        res.new_position = 5
        res.old_position = 0
        illegal = False
        game_over = False
        reward = self.calc.compute(res, illegal, self.cfg, game_over)
        expected = (
            self.cfg.reward.capture
            + self.cfg.reward.capture_choice_bonus
            + self.cfg.reward.time_penalty
        )
        self.assertAlmostEqual(reward, expected, places=3)

    def test_compute_illegal_action(self):
        res = Mock()
        res.captured_tokens = []
        res.finished_token = False
        res.new_position = 5
        res.old_position = 0
        illegal = True
        game_over = False
        reward = self.calc.compute(res, illegal, self.cfg, game_over)
        expected = self.cfg.reward.illegal_action + self.cfg.reward.time_penalty
        self.assertEqual(reward, expected)

    def test_compute_win(self):
        from ludo_engine.models import PlayerColor

        res = Mock()
        res.captured_tokens = []
        res.finished_token = False
        res.new_position = 5
        res.old_position = 0
        illegal = False
        game_over = True
        winner = Mock()
        winner.color = PlayerColor.RED
        agent_color = PlayerColor.RED
        reward = self.calc.compute(
            res, illegal, self.cfg, game_over, winner=winner, agent_color=agent_color
        )
        expected = self.cfg.reward.win + self.cfg.reward.time_penalty
        self.assertEqual(reward, expected)

    def test_compute_lose(self):
        from ludo_engine.models import PlayerColor

        res = Mock()
        res.captured_tokens = []
        res.finished_token = False
        res.new_position = 5
        res.old_position = 0
        illegal = False
        game_over = True
        winner = Mock()
        winner.color = PlayerColor.BLUE
        agent_color = PlayerColor.RED
        reward = self.calc.compute(
            res, illegal, self.cfg, game_over, winner=winner, agent_color=agent_color
        )
        expected = self.cfg.reward.lose + self.cfg.reward.time_penalty
        self.assertEqual(reward, expected)

    def test_compute_finish_token(self):
        res = Mock()
        res.captured_tokens = []
        res.finished_token = True
        res.new_position = GameConstants.FINISH_POSITION
        res.old_position = 0
        illegal = False
        game_over = False
        reward = self.calc.compute(res, illegal, self.cfg, game_over)
        expected = 23.499
        self.assertAlmostEqual(reward, expected)

    def test_compute_captured_by_opponents(self):
        res = Mock()
        res.captured_tokens = []
        res.finished_token = False
        res.new_position = 5
        res.old_position = 0
        illegal = False
        game_over = False
        reward = self.calc.compute(
            res, illegal, self.cfg, game_over, captured_by_opponents=2, home_tokens=1
        )
        expected = self.cfg.reward.got_captured * 2 + self.cfg.reward.time_penalty
        self.assertEqual(reward, expected)

    def test_compute_exit_start(self):
        res = Mock()
        res.captured_tokens = []
        res.finished_token = False
        res.new_position = 1
        res.old_position = GameConstants.HOME_POSITION
        illegal = False
        game_over = False
        reward = self.calc.compute(res, illegal, self.cfg, game_over)
        expected = 5.999
        self.assertAlmostEqual(reward, expected)

    def test_compute_extra_turn(self):
        res = Mock()
        res.captured_tokens = []
        res.finished_token = False
        res.new_position = 5
        res.old_position = 0
        illegal = False
        game_over = False
        reward = self.calc.compute(res, illegal, self.cfg, game_over, extra_turn=True)
        expected = self.cfg.reward.extra_turn + self.cfg.reward.time_penalty
        self.assertEqual(reward, expected)


class TestRiskOpportunityCalculator(unittest.TestCase):
    def setUp(self):
        self.weights = SimpleROWeights()
        self.calc = RiskOpportunityCalculator(self.weights)

    def test_init(self):
        calc = RiskOpportunityCalculator()
        # weights can be None, but compute uses default
        self.assertIsNone(calc.weights)

    @patch(
        "ludo_rl.utils.risk_opportunity.RiskOpportunityCalculator._iter_opponent_positions"
    )
    @patch("ludo_rl.utils.risk_opportunity.RiskOpportunityCalculator._forward_distance")
    def test_risk_score_no_threats(self, mock_forward, mock_iter):
        mock_iter.return_value = []
        game = Mock()
        agent_color = Mock()
        target_pos = 10
        score = self.calc._risk_score(game, agent_color, target_pos, self.weights)
        self.assertEqual(score, 0.0)

    def test_opportunity_score_capture(self):
        move = Mock()
        move.captured_tokens = [Mock()]
        move.finished_token = False
        move.extra_turn = False
        old_pos = 0
        new_pos = 5
        score = self.calc._opportunity_score(move, old_pos, new_pos, self.weights)
        expected = self.weights.capture + 5 * self.weights.progress_per_step  # 5 steps
        self.assertAlmostEqual(score, expected, places=3)

    def test_compute(self):
        game = Mock()
        game.players = []  # Make it iterable
        agent_color = Mock()
        move = Mock()
        move.old_position = 0
        move.new_position = 5
        move.captured_tokens = []
        move.finished_token = False
        move.extra_turn = False
        score = self.calc.compute(game, agent_color, move)
        self.assertIsInstance(score, float)

    def test_compute_with_breakdown(self):
        game = Mock()
        game.players = []
        agent_color = Mock()
        move = Mock()
        move.old_position = 0
        move.new_position = 5
        move.captured_tokens = []
        move.finished_token = False
        move.extra_turn = False
        result = self.calc.compute(game, agent_color, move, return_breakdown=True)
        self.assertIsInstance(result, tuple)
        score, breakdown = result
        self.assertIn("opportunity", breakdown)
        self.assertIn("risk", breakdown)

    def test_opportunity_score_finish_token(self):
        move = Mock()
        move.captured_tokens = []
        move.finished_token = True
        move.extra_turn = False
        old_pos = 0
        new_pos = GameConstants.FINISH_POSITION
        score = self.calc._opportunity_score(move, old_pos, new_pos, self.weights)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0)

    def test_opportunity_score_leave_home(self):
        move = Mock()
        move.captured_tokens = []
        move.finished_token = False
        move.extra_turn = False
        old_pos = -1  # HOME
        new_pos = 0
        score = self.calc._opportunity_score(move, old_pos, new_pos, self.weights)
        self.assertEqual(score, self.weights.leave_home)

    def test_opportunity_score_safe_square(self):
        move = Mock()
        move.captured_tokens = []
        move.finished_token = False
        move.extra_turn = False
        old_pos = 0
        new_pos = 8  # Assume safe
        with patch(
            "ludo_rl.utils.risk_opportunity.BoardConstants.is_safe_position",
            return_value=True,
        ):
            score = self.calc._opportunity_score(move, old_pos, new_pos, self.weights)
            self.assertIsInstance(score, float)

    def test_risk_score_with_threats(self):
        game = Mock()
        agent_color = Mock()
        target_pos = 10
        # Mock opponent positions
        with patch.object(self.calc, "_iter_opponent_positions", return_value=[5, 15]):
            with patch(
                "ludo_rl.utils.risk_opportunity.RiskOpportunityCalculator._forward_distance",
                side_effect=[5, 1],
            ):
                score = self.calc._risk_score(
                    game, agent_color, target_pos, self.weights
                )
                self.assertIsInstance(score, float)


if __name__ == "__main__":
    unittest.main()
