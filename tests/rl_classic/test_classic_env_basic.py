import unittest

from rl.envs.ludo_env.ludo_env_multi_seat import EnvConfig, LudoGymEnv


class TestClassicEnvBasic(unittest.TestCase):
    def setUp(self):
        self.env = LudoGymEnv(EnvConfig(max_turns=50))

    def test_reset_shapes(self):
        obs, info = self.env.reset(seed=123)
        self.assertIsInstance(obs, type(self.env.last_obs))
        self.assertEqual(obs.shape, self.env.observation_space.shape)

    def test_step_no_crash(self):
        obs, info = self.env.reset(seed=42)
        action = 0
        nobs, reward, terminated, truncated, info = self.env.step(action)
        self.assertEqual(nobs.shape, obs.shape)
        self.assertIn("illegal_action", info)


if __name__ == "__main__":
    unittest.main()
