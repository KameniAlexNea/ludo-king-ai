import types

import numpy as np
import pytest
from gymnasium import spaces

from src.models.configs.config import EnvConfig
from src.models.envs.ludo_env_aec.opponent_pool import OpponentPoolManager
from src.models.envs.ludo_env_aec.raw_env import raw_env
from src.models.envs.ludo_env_aec.turn_based_env import TurnBasedSelfPlayEnv


class FakeGame:
    """Minimal fake game object to satisfy TurnBasedSelfPlayEnv's scripted opponent logic."""

    def __init__(self):
        self.players = [types.SimpleNamespace() for _ in range(4)]


class SimpleBaseEnv:
    """Minimal fake PettingZoo-like base env for testing TurnBasedSelfPlayEnv."""

    def __init__(self):
        self.possible_agents = [f"player_{i}" for i in range(4)]
        self.agents = list(self.possible_agents)
        self._idx = 0
        self.agent_selection = self.agents[self._idx]
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.rewards = {a: 0.0 for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.game = FakeGame()
        self._pending_dice = {}
        self._pending_valid_moves = {}

    # Provide compatibility accessors used by the production env
    # (the real raw_env exposes these as public methods so wrappers
    # don't need to reach into private attributes).
    def pending_dice(self, agent):
        return int(self._pending_dice.get(agent, 0))

    def valid_move_tokens(self, agent):
        return [m.token_id for m in self._pending_valid_moves.get(agent, [])]

    def observation_space(self, agent):
        return spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

    def action_space(self, agent):
        return spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        self._idx = 0
        self.agent_selection = self.agents[self._idx]
        return None

    def observe(self, agent):
        if agent == self.agent_selection:
            return {
                "observation": np.zeros(8, dtype=np.float32),
                "action_mask": np.ones(4, dtype=np.int8),
            }
        return {
            "observation": np.zeros(8, dtype=np.float32),
            "action_mask": np.zeros(4, dtype=np.int8),
        }

    def step(self, action):
        # advance to next agent cyclically
        self._idx = (self._idx + 1) % len(self.agents)
        self.agent_selection = self.agents[self._idx]


def test_single_learning_seat_assignment():
    base = SimpleBaseEnv()
    env = TurnBasedSelfPlayEnv(base, opponent_pool=None)

    # deterministically seed RNG
    obs, info = env.reset(seed=123)

    total_agents = len(env.possible_agents)
    assigned = len(env.opponent_assignments) + len(env.scripted_assignments)
    # All but one agent should be assigned as opponents
    assert assigned == total_agents - 1


def test_prune_opponent_model_cache(tmp_path):
    base = SimpleBaseEnv()
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()

    # create fake opponent files
    f1 = pool_dir / "opponent_1.zip"
    f1.write_text("x")
    f2 = pool_dir / "opponent_2.zip"
    f2.write_text("y")

    manager = OpponentPoolManager(str(pool_dir), pool_size=1)

    env = TurnBasedSelfPlayEnv(base, opponent_pool=manager)
    # manually populate cache with both files (simulate earlier loads)
    env.opponent_models[str(f1)] = object()
    env.opponent_models[str(f2)] = object()

    # manager should only keep the last one due to pool_size=1
    manager._load_existing_opponents()

    # Only models present in pool should remain cached
    valid = set(manager.get_all_opponents())
    assert set(env.opponent_models.keys()).issubset(valid)


def test_turn_handoff_and_masks():
    base = SimpleBaseEnv()
    env = TurnBasedSelfPlayEnv(base, opponent_pool=None)

    obs, info = env.reset(seed=1)
    # initial observation should include action_mask and agent_index
    assert "action_mask" in obs and "observation" in obs and "agent_index" in obs

    # Simulate one learning action; since no opponents assigned, action passes through
    next_obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(next_obs, dict)
    assert "action_mask" in next_obs


def test_raw_env_illegal_action_handling():
    """Test that raw_env properly detects and handles illegal actions."""
    cfg = EnvConfig()
    cfg.seed = 42
    env = raw_env(cfg)
    env.reset(seed=42)

    agent = env.agent_selection

    # Get valid moves for the current state
    valid_moves = env._pending_valid_moves.get(agent, [])

    if valid_moves:
        # Find a token ID that is NOT in the valid moves
        valid_token_ids = {move.token_id for move in valid_moves}
        all_token_ids = {0, 1, 2, 3}
        illegal_token_ids = all_token_ids - valid_token_ids

        if illegal_token_ids:
            # Attempt an illegal action
            illegal_action = min(illegal_token_ids)
            env.step(illegal_action)

            # Check that is_illegal flag was set in info
            assert env.infos[agent].get("illegal_action", False) is True

            # Verify reward breakdown includes illegal action penalty
            reward_breakdown = env.infos[agent].get("reward_breakdown", {})
            assert "illegal_action" in reward_breakdown
        else:
            # All tokens are valid, skip this part of the test
            pytest.skip("All tokens are valid moves in this game state")
    else:
        # No valid moves scenario - this is legal (pass turn)
        # Try any action, should not be marked illegal when no moves available
        env.step(0)
        assert env.infos[agent].get("illegal_action", False) is False
