import unittest

from ludo_rl.config import (
    CurriculumConfig,
    EnvConfig,
    OpponentConfig,
    RewardConfig,
    TrainConfig,
)


class TestRewardConfig(unittest.TestCase):
    def test_custom_values(self):
        cfg = RewardConfig(win=200.0, capture=10.0)
        self.assertEqual(cfg.win, 200.0)
        self.assertEqual(cfg.capture, 10.0)


class TestOpponentConfig(unittest.TestCase):
    def test_default_candidates(self):
        cfg = OpponentConfig()
        self.assertIn("random", cfg.candidates)
        self.assertIn("probabilistic_v2", cfg.candidates)

    def test_evaluation_candidates(self):
        cfg = OpponentConfig()
        self.assertIn("probabilistic_v2", cfg.evaluation_candidates)


class TestCurriculumConfig(unittest.TestCase):
    def test_default_values(self):
        cfg = CurriculumConfig()
        self.assertTrue(cfg.enabled)
        self.assertEqual(len(cfg.boundaries), 3)


class TestEnvConfig(unittest.TestCase):
    def test_default_values(self):
        cfg = EnvConfig()
        self.assertEqual(cfg.max_turns, 500)
        self.assertIsNone(cfg.seed)
        self.assertTrue(cfg.randomize_agent)


class TestTrainConfig(unittest.TestCase):
    def test_default_values(self):
        cfg = TrainConfig()
        self.assertEqual(cfg.total_steps, 20_000_000)
        self.assertEqual(cfg.n_envs, 8)
        self.assertEqual(cfg.learning_rate, 3e-4)

    def test_post_init_selfplay(self):
        cfg = TrainConfig(env_type="selfplay")
        cfg.__post_init__()
        self.assertEqual(cfg.n_envs, 1)


if __name__ == "__main__":
    unittest.main()
