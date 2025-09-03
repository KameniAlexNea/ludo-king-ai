import unittest

from ludo_rls.envs.ludo_env import EnvConfig
from ludo_rls.envs.ludo_env import LudoGymEnv as SingleSeatEnv

REQ = [
    "reward_components",
    "step_breakdown",
    "dice",
    "illegal_action",
    "action_mask",
    "progress_delta",
]


class TestSingleSeatEnvRewards(unittest.TestCase):
    def test_step_info_keys(self):
        env = SingleSeatEnv(EnvConfig(max_turns=120))
        obs, info = env.reset(seed=55)
        mask = info.get("action_mask", [1, 1, 1, 1])
        action = next((i for i, v in enumerate(mask) if v == 1), 0)
        obs, r, term, trunc, info = env.step(action)
        for k in REQ:
            self.assertIn(k, info)
        self.assertIsInstance(info["step_breakdown"], dict)

    def test_timeout_terminal_draw_penalty(self):
        env = SingleSeatEnv(EnvConfig(max_turns=2))
        obs, info = env.reset(seed=99)
        total = 0.0
        done = False
        while not done:
            mask = info.get("action_mask", [1, 1, 1, 1])
            action = next((i for i, v in enumerate(mask) if v == 1), 0)
            obs, r, term, trunc, info = env.step(action)
            total += r
            done = term or trunc
        self.assertTrue(done)
        self.assertIsInstance(total, float)


if __name__ == "__main__":
    unittest.main()
