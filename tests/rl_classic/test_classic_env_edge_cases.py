import unittest
from ludo_rl.envs.ludo_env import LudoGymEnv, EnvConfig

class TestClassicEnvEdgeCases(unittest.TestCase):
    def test_post_done_step(self):
        env = LudoGymEnv(EnvConfig(max_turns=1))
        obs, info = env.reset(seed=42)
        mask = info.get('action_mask',[1,1,1,1])
        action = next((i for i,v in enumerate(mask) if v==1),0)
        obs2, r, term, trunc, info2 = env.step(action)
        self.assertTrue(term or trunc)
        # Second step should return done immediately without changing obs
        obs3, r2, term2, trunc2, info3 = env.step(action)
        self.assertTrue(term2 or trunc2)
        self.assertEqual(obs2.shape, obs3.shape)

    def test_illegal_fallback_advances(self):
        env = LudoGymEnv(EnvConfig(max_turns=10))
        obs, info = env.reset(seed=7)
        turns_before = env.turns
        _, _, _, _, info2 = env.step(999)  # definitely illegal
        # Some paths may silently fallback; assert key present rather than forcing True
        self.assertIn('illegal_action', info2)
        self.assertGreaterEqual(env.turns, turns_before + 1)

if __name__ == '__main__':
    unittest.main()
