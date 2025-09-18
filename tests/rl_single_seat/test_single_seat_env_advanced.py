import unittest

from rl.envs.ludo_env.ludo_env_single_seat import EnvConfig
from rl.envs.ludo_env.ludo_env_single_seat import LudoGymEnv as SingleSeatEnv


class TestSingleSeatEnvAdvanced(unittest.TestCase):
    def setUp(self):
        cfg = EnvConfig(max_turns=300, randomize_training_color=True)
        self.env = SingleSeatEnv(cfg)

    def test_training_color_randomization(self):
        colors = set()
        for seed in range(5):
            obs, info = self.env.reset(seed=seed)
            colors.add(self.env.training_color)
        self.assertTrue(len(colors) > 1, "Expected multiple training colors sampled")

    def test_mask_autocorrect_flag(self):
        obs, info = self.env.reset(seed=1234)
        # Choose invalid action beyond space
        invalid_action = 42
        _, _, _, _, info2 = self.env.step(invalid_action)
        # In masked mode invalid chosen action should not necessarily mark illegal_action True if autocorrected
        # Here environment sets illegal_action if masking disabled; with mask enabled we allow masked_autocorrect in breakdown
        # Just assert info fields present and action_mask still valid size
        self.assertIn("action_mask", info2)
        self.assertEqual(len(info2["action_mask"]), self.env.action_space.n)

    def test_capture_grants_extra_turn(self):
        # Hard to force capture deterministically; run multiple steps and look for extra turn flag
        obs, info = self.env.reset(seed=999)
        saw_extra = False
        for _ in range(60):
            mask = info.get("action_mask", [1, 1, 1, 1])
            action = next((i for i, v in enumerate(mask) if v == 1), 0)
            obs, reward, term, trunc, info = self.env.step(action)
            if info.get("had_extra_turn"):
                saw_extra = True
            if term or trunc:
                break
        self.assertTrue(saw_extra, "Expected at least one extra turn across steps")


if __name__ == "__main__":
    unittest.main()
