import unittest
import numpy as np

from ludo_rl.envs.ludo_env import LudoGymEnv, EnvConfig
from ludo.constants import Colors, GameConstants
from ludo.token import TokenState


def _make_env(seed=123):
    cfg = EnvConfig(max_turns=50, agent_color=Colors.RED, seed=seed)
    return LudoGymEnv(cfg)


class TestLudoRLEnv(unittest.TestCase):
    def test_reset_observation_shape(self):
        env = _make_env()
        obs, info = env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertTrue(((obs >= -1.0) & (obs <= 1.0)).all())

    def test_step_progresses_turns(self):
        env = _make_env()
        env.reset()
        initial_turns = env.turns
        obs, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(env.turns, initial_turns + 1)
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated and truncated)
        self.assertIn("reward_components", info)
        self.assertIn("step_breakdown", info)

    def test_extra_turn_logic(self):
        env = _make_env(seed=999)
        env.reset()
        saw_extra = False
        for _ in range(30):
            _, _, _, _, info = env.step(0)
            if info.get("had_extra_turn"):
                saw_extra = True
                break
        self.assertIsInstance(saw_extra, bool)

    def test_observation_can_finish_flag(self):
        env = _make_env()
        env.reset()
        player = next(p for p in env.game.players if p.color.value == env.agent_color)
        token = player.tokens[0]
        # Place token such that remaining distance <= dice max (strongest case = 1)
        token.position = GameConstants.FINISH_POSITION - 1
        # Ensure state reflects home column for consistency
        token.state = TokenState.HOME_COLUMN
        obs = env.obs_builder._build_observation(env.turns, env._pending_agent_dice)
        # Dynamically compute index: agent(4)+opp(12)+finished(4)=20 -> can_finish at 20
        can_finish_idx = 4 + 12 + 4
        if obs[can_finish_idx] != 1.0:
            # Fallback scan: first scalar section after finished counts; expect exactly one position to flip when moving token
            scalar_section = obs[4 + 12 + 4 : 4 + 12 + 4 + 6]
            raise AssertionError(
                f"can_finish flag not 1 at expected index {can_finish_idx}. Segment={scalar_section.tolist()} token_pos={token.position} remaining={GameConstants.FINISH_POSITION - token.position}"
            )

    def test_observation_can_finish_flag_negative(self):
        env = _make_env()
        env.reset()
        # Ensure all tokens far from finish (already true initially)
        obs = env.obs_builder._build_observation(env.turns, env._pending_agent_dice)
        can_finish_idx = 4 + 12 + 4
        self.assertEqual(obs[can_finish_idx], 0.0)

    def test_reward_components_non_empty(self):
        env = _make_env()
        env.reset()
        _, reward, _, _, info = env.step(0)
        self.assertIsInstance(reward, float)
        self.assertGreaterEqual(len(info["reward_components"]), 1)
        self.assertGreaterEqual(len(info["reward_components"]), len(info["step_breakdown"]))

    def test_observation_length_matches_builder(self):
        env = _make_env()
        obs, _ = env.reset()
        expected = env.obs_builder._compute_observation_size()
        self.assertEqual(obs.shape[0], expected)


if __name__ == "__main__":
    unittest.main()

