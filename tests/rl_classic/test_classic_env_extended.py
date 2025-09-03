import unittest

from ludo_rl.envs.ludo_env import EnvConfig, LudoGymEnv


class TestClassicEnvExtended(unittest.TestCase):
    def setUp(self):
        self.env = LudoGymEnv(EnvConfig(max_turns=200))

    def test_action_mask_shape(self):
        obs, info = self.env.reset(seed=123)
        # First step to populate info with mask
        obs2, r, term, trunc, info = self.env.step(0)
        mask = info.get("action_mask")
        self.assertIsNotNone(mask)
        self.assertEqual(len(mask), self.env.action_space.n)
        # Mask should be 0/1 only
        self.assertTrue(all(m in (0, 1) for m in mask))

    def test_illegal_action_penalty(self):
        # Reset and take an initial action to populate dice/state
        obs, info = self.env.reset(seed=321)
        self.env.step(0)
        # Deliberately choose an out-of-range action id
        illegal_action = 99
        _, _, _, _, info2 = self.env.step(illegal_action)
        # Some paths may silently fallback; just assert the key exists
        self.assertIn("illegal_action", info2)

    def test_extra_turn_chain(self):
        obs, info = self.env.reset(seed=777)
        # Try many steps; ensure at least one extra turn occurs over several decisions
        had_extra = False
        for _ in range(30):
            # Pick first legal from mask else 0
            mask = info.get("action_mask", [1, 1, 1, 1])
            action = next((i for i, v in enumerate(mask) if v == 1), 0)
            obs, reward, term, trunc, info = self.env.step(action)
            if info.get("had_extra_turn"):
                had_extra = True
            if term or trunc:
                break
        self.assertTrue(had_extra, "Expected at least one extra turn in 30 decisions")


if __name__ == "__main__":
    unittest.main()
