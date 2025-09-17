import unittest

from ludo_engine.token import TokenState

from ludo_rls.envs.ludo_env import EnvConfig
from ludo_rls.envs.ludo_env import LudoGymEnv as SingleSeatEnv


class TestSingleSeatEnvEdgeCases(unittest.TestCase):
    def test_mask_vs_no_mask_illegal_flag(self):
        # Mask enabled (default)
        env_mask = SingleSeatEnv(EnvConfig(max_turns=5, use_action_mask=True))
        obs, info = env_mask.reset(seed=11)
        # Prime with a legal step if possible to generate mask
        env_mask.step(0)
        _, _, _, _, info2 = env_mask.step(999)
        masked_illegal = info2.get("illegal_action")
        # Mask disabled
        env_no_mask = SingleSeatEnv(EnvConfig(max_turns=5, use_action_mask=False))
        obs, info = env_no_mask.reset(seed=11)
        env_no_mask.step(0)
        _, _, _, _, info3 = env_no_mask.step(999)
        unmasked_illegal = info3.get("illegal_action")
        # If unmasked path flags illegal, masked path should not be True simultaneously (either False or absent)
        if unmasked_illegal:
            self.assertTrue(masked_illegal in (False, None))
        else:
            # Both could be False/None; minimally ensure keys exist
            self.assertIn("illegal_action", info3)

    def test_forced_terminal_reward(self):
        env = SingleSeatEnv(EnvConfig(max_turns=50))
        obs, info = env.reset(seed=22)
        # Force all training seat tokens finished then take a dummy step to trigger terminal reward path
        training_color = env.training_color
        player = next(p for p in env.game.players if p.color.value == training_color)
        for t in player.tokens:
            t.state = TokenState.FINISHED
        # Provide a step; env should detect finished player naturally
        _, reward, term, trunc, info2 = env.step(0)
        # Allow either termination OR at least non-zero reward + finished tokens
        finished_tokens = sum(1 for t in player.tokens if t.is_finished())
        self.assertEqual(finished_tokens, 4)
        self.assertTrue((term or trunc) or reward != 0.0)


if __name__ == "__main__":
    unittest.main()
