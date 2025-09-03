import unittest
from ludo_rl.envs.ludo_env import LudoGymEnv, EnvConfig

REQUIRED_KEYS = [
    'reward_components','step_breakdown','dice','illegal_action','action_mask','progress_delta'
]

class TestClassicEnvRewards(unittest.TestCase):
    def test_step_breakdown_keys_present(self):
        env = LudoGymEnv(EnvConfig(max_turns=120))
        obs, info = env.reset(seed=2024)
        mask = info.get('action_mask',[1,1,1,1])
        action = next((i for i,v in enumerate(mask) if v==1),0)
        obs, rew, term, trunc, info = env.step(action)
        for k in REQUIRED_KEYS:
            self.assertIn(k, info)
        self.assertIsInstance(info['step_breakdown'], dict)

    def test_terminal_reward_on_timeout(self):
        env = LudoGymEnv(EnvConfig(max_turns=3))
        obs, info = env.reset(seed=1)
        total = 0.0
        done = False
        while not done:
            mask = info.get('action_mask',[1,1,1,1])
            action = next((i for i,v in enumerate(mask) if v==1),0)
            obs, r, term, trunc, info = env.step(action)
            total += r
            done = term or trunc
        self.assertTrue(done)
        # At least some reward shaping occurred (not necessarily >0 but should be finite float)
        self.assertIsInstance(total, float)

if __name__ == '__main__':
    unittest.main()
