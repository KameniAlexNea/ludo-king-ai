import unittest
from ludo_rl.envs.ludo_env import LudoGymEnv, EnvConfig

class TestClassicEnvExtended(unittest.TestCase):
    def setUp(self):
        self.env = LudoGymEnv(EnvConfig(max_turns=200))

    def test_action_mask_shape(self):
        obs, info = self.env.reset(seed=123)
        self.assertIn('action_mask', info)
        mask = info['action_mask']
        self.assertEqual(len(mask), self.env.action_space.n)
        # Mask should be 0/1 only
        self.assertTrue(all(m in (0,1) for m in mask))

    def test_illegal_action_penalty(self):
        obs, info = self.env.reset(seed=321)
        # Force an illegal action id outside [0,3]
        illegal_action = 99
        _, _, _, _, info2 = self.env.step(illegal_action)
        # Env autocorrects but flags illegal_action True
        self.assertTrue(info2['illegal_action'])

    def test_extra_turn_chain(self):
        obs, info = self.env.reset(seed=777)
        turns_before = self.env.turns
        # Try many steps; ensure at least one extra turn occurs over several decisions
        had_extra = False
        for _ in range(30):
            # Pick first legal from mask else 0
            mask = info.get('action_mask', [1,1,1,1])
            action = next((i for i,v in enumerate(mask) if v==1), 0)
            obs, reward, term, trunc, info = self.env.step(action)
            if info.get('had_extra_turn'):
                had_extra = True
            if term or trunc:
                break
        self.assertTrue(had_extra, 'Expected at least one extra turn in 30 decisions')

if __name__ == '__main__':
    unittest.main()
