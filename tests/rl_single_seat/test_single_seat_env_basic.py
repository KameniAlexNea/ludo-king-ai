import unittest
try:
    from ludo_rls.envs.ludo_env import LudoGymEnv as SingleSeatEnv, EnvConfig
except Exception:
    SingleSeatEnv = None
    EnvConfig = None

class TestSingleSeatEnvBasic(unittest.TestCase):
    def setUp(self):
        if SingleSeatEnv is None:
            self.skipTest('Single seat env not available')
        self.env = SingleSeatEnv(EnvConfig(max_turns=50))

    def test_reset(self):
        obs, info = self.env.reset(seed=77)
        self.assertEqual(obs.shape, self.env.observation_space.shape)

    def test_action_mask_in_info(self):
        obs, info = self.env.reset(seed=11)
        self.assertIn('action_mask', info)
        self.assertEqual(len(info['action_mask']), self.env.action_space.n)

if __name__ == '__main__':
    unittest.main()
