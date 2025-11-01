import types

import numpy as np
import pytest
from gymnasium import spaces
from ludo_engine.models import ALL_COLORS

from src.models.configs.config import EnvConfig
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
    env = TurnBasedSelfPlayEnv(base)

    # deterministically seed RNG
    obs, info = env.reset(seed=123)

    total_agents = len(env.possible_agents)
    assigned = len(env.opponent_assignments) + len(env.scripted_assignments)
    # No scripted opponents are registered in the simplified test harness
    assert assigned == 0
    assert total_agents == len(base.possible_agents)


def test_turn_handoff_and_masks():
    base = SimpleBaseEnv()
    env = TurnBasedSelfPlayEnv(base)

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


def test_raw_env_respects_matchup_configuration():
    cfg = EnvConfig(matchup="1v1", seed=21)
    env = raw_env(cfg)

    assert len(env.possible_agents) == cfg.player_count
    assert env.opponent_slots == max(1, cfg.player_count - 1)

    env.reset()
    assert len(env.agents) == cfg.player_count

    sample_agent = env.possible_agents[0]
    obs_space = env.observation_space(sample_agent)
    assert obs_space.shape[0] > 0

    colors = [player.color for player in env.game.players]
    assert len(colors) == cfg.player_count
    all_colors = list(ALL_COLORS)
    first_idx = all_colors.index(colors[0])
    second_idx = all_colors.index(colors[1])
    assert second_idx == (first_idx + len(all_colors) // 2) % len(all_colors)
