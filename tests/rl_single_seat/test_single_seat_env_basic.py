import unittest

try:
    from rl.envs.ludo_env.ludo_env_single_seat import EnvConfig
    from rl.envs.ludo_env.ludo_env_single_seat import LudoGymEnv as SingleSeatEnv
except Exception:
    SingleSeatEnv = None
    EnvConfig = None


class TestSingleSeatEnvBasic(unittest.TestCase):
    def setUp(self):
        if SingleSeatEnv is None:
            self.skipTest("Single seat env not available")
        self.env = SingleSeatEnv(EnvConfig(max_turns=50))

    def test_reset(self):
        obs, info = self.env.reset(seed=77)
        self.assertEqual(obs.shape, self.env.observation_space.shape)

    def test_action_mask_in_info(self):
        obs, info = self.env.reset(seed=11)
        # Mask appears after first step, not on reset
        obs2, r, term, trunc, info2 = self.env.step(0)
        self.assertIn("action_mask", info2)
        self.assertEqual(len(info2["action_mask"]), self.env.action_space.n)


if __name__ == "__main__":
    unittest.main()
