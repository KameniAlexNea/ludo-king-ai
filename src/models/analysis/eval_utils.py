"""Shared evaluation helpers for training callbacks and CLI scripts."""

from __future__ import annotations

import copy
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ludo_engine import LudoGame
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from ..configs.config import EnvConfig
from ..envs.ludo_env import LudoRLEnv
from ..envs.spaces import get_flat_space_config, get_space_config


def _mask_fn(env: LudoRLEnv):
    return env.valid_action_mask()


class SharedPolicyEvalEnv(gym.Wrapper):
    """Adapts the single-agent eval env to the shared-policy observation format."""

    def __init__(
        self, env: gym.Env, num_agents: int = 4, opponent_count: int | None = None
    ):
        super().__init__(env)
        opponent_slots = (
            max(1, opponent_count)
            if opponent_count is not None
            else max(1, num_agents - 1)
        )
        base_space = get_flat_space_config(opponent_slots)
        obs_low = np.asarray(base_space.low, dtype=np.float32)
        obs_high = np.asarray(base_space.high, dtype=np.float32)
        action_dim = int(env.action_space.n)
        self._mask_shape = (action_dim,)
        self._obs_keys = list(get_space_config(opponent_slots).spaces.keys())
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=obs_low,
                    high=obs_high,
                    dtype=np.float32,
                ),
                "prev_observation": spaces.Box(
                    low=obs_low,
                    high=obs_high,
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=self._mask_shape,
                    dtype=np.int8,
                ),
                "agent_index": spaces.Discrete(num_agents),
            }
        )
        self._agent_index = np.array(0, dtype=np.int64)
        self._last_mask = np.ones(self._mask_shape, dtype=np.int8)
        self._zero_prev_obs = np.zeros_like(obs_low, dtype=np.float32)
        self._prev_obs = self._zero_prev_obs.copy()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._prev_obs = self._zero_prev_obs.copy()
        obs, info = self.env.reset(seed=seed, options=options)
        mask = self._extract_mask(info)
        converted = self._convert_obs(obs, mask)
        obs_with_prev = {
            "observation": converted["observation"],
            "prev_observation": self._prev_obs.copy(),
            "action_mask": converted["action_mask"],
            "agent_index": converted["agent_index"],
        }
        self._prev_obs = converted["observation"].copy()
        info = {**info, "action_mask": mask.copy()}
        return obs_with_prev, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        mask = self._extract_mask(info)
        converted = self._convert_obs(obs, mask)
        obs_with_prev = {
            "observation": converted["observation"],
            "prev_observation": self._prev_obs.copy(),
            "action_mask": converted["action_mask"],
            "agent_index": converted["agent_index"],
        }
        self._prev_obs = converted["observation"].copy()
        info = {**info, "action_mask": mask.copy()}
        return obs_with_prev, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self._last_mask.copy()

    def _extract_mask(self, info: dict) -> np.ndarray:
        if "action_mask" in info:
            mask = np.asarray(info["action_mask"], dtype=np.int8)
        elif hasattr(self.env, "action_masks"):
            mask = np.asarray(self.env.action_masks(), dtype=np.int8)
        else:
            mask = np.ones(self._mask_shape, dtype=np.int8)
        self._last_mask = mask
        return mask

    def _convert_obs(
        self, obs: Dict[str, np.ndarray], mask: np.ndarray
    ) -> Dict[str, np.ndarray]:
        if not isinstance(obs, dict):  # already flatten observation
            return {
                "observation": obs,
                "action_mask": mask.astype(np.int8, copy=False),
                "agent_index": self._agent_index.copy(),
            }
        flat_values = []
        agent_index = 0
        for key in self._obs_keys:
            if key not in obs:
                raise KeyError(f"Observation missing expected key '{key}'")
            value = np.asarray(obs[key], dtype=np.float32).ravel()
            if key == "agent_index":
                agent_index = int(value[0])
            flat_values.append(value)
        flat_obs = np.concatenate(flat_values).astype(np.float32, copy=False)
        return {
            "observation": flat_obs,
            "action_mask": mask.astype(np.int8, copy=False),
            "agent_index": np.array(agent_index, dtype=np.int64),
        }


@dataclass
class EvalStats:
    opponent: str
    episodes: int
    total_reward: float = 0.0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    lengths: int = 0
    # Terminal-only score (+1 win, 0 draw, -1 loss)
    terminal_score_sum: float = 0.0
    # Cumulative reward breakdown sums (episode-summed, then aggregated over episodes)
    breakdown_sums: Dict[str, float] = field(
        default_factory=lambda: {
            "progress": 0.0,
            "capture": 0.0,
            "finish": 0.0,
            "got_captured": 0.0,
            "illegal": 0.0,
            "time_penalty": 0.0,
            "terminal": 0.0,
        }
    )

    def update(self, reward: float, length: int, result: str) -> None:
        self.total_reward += reward
        self.lengths += length
        if result == "win":
            self.wins += 1
        elif result == "loss":
            self.losses += 1
        else:
            self.draws += 1

    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(1, self.episodes)

    @property
    def avg_length(self) -> float:
        return self.lengths / max(1, self.episodes)

    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.episodes)

    def avg_terminal_score(self) -> float:
        return self.terminal_score_sum / max(1, self.episodes)

    def as_dict(self) -> Dict[str, float]:
        return {
            "opponent": self.opponent,
            "episodes": float(self.episodes),
            "win_rate": self.win_rate,
            "loss_rate": self.losses / max(1, self.episodes),
            "draw_rate": self.draws / max(1, self.episodes),
            "avg_reward": self.avg_reward,
            "avg_length": self.avg_length,
            "avg_terminal_score": self.avg_terminal_score(),
            # Selected breakdown components for quick inspection
            "breakdown/progress": self.breakdown_sums["progress"]
            / max(1, self.episodes),
            "breakdown/capture": self.breakdown_sums["capture"] / max(1, self.episodes),
            "breakdown/finish": self.breakdown_sums["finish"] / max(1, self.episodes),
            "breakdown/got_captured": self.breakdown_sums["got_captured"]
            / max(1, self.episodes),
            "breakdown/illegal": self.breakdown_sums["illegal"] / max(1, self.episodes),
            "breakdown/time_penalty": self.breakdown_sums["time_penalty"]
            / max(1, self.episodes),
            "breakdown/terminal": self.breakdown_sums["terminal"]
            / max(1, self.episodes),
        }


def build_eval_env(opponent: str, cfg: EnvConfig) -> DummyVecEnv:
    def _init():
        opponent_cfg = copy.deepcopy(cfg)
        opponent_cfg.opponent_strategy = opponent
        env = LudoRLEnv(opponent_cfg)
        env = ActionMasker(env, _mask_fn)
        if cfg.multi_agent:
            env = SharedPolicyEvalEnv(
                env,
                num_agents=cfg.player_count,
                opponent_count=cfg.opponent_count,
            )
        return env

    return DummyVecEnv([_init])


def evaluate_against(
    model: MaskablePPO,
    opponent: str,
    games: int,
    base_cfg: EnvConfig,
    deterministic: bool,
) -> EvalStats:
    env = build_eval_env(opponent, deepcopy(base_cfg))
    stats = EvalStats(opponent=opponent, episodes=games)

    try:
        seed = base_cfg.seed or 0
        for _ in range(games):
            set_random_seed(seed)
            seed += 1
            observation = env.reset()
            done = False
            episode_reward = 0.0
            steps = 0
            base_env: LudoRLEnv = env.envs[0].unwrapped
            game: LudoGame = base_env.game
            agent_color = base_env.agent_color
            # Accumulate per-episode breakdown totals for diagnostics
            episode_breakdown: Dict[str, float] = {
                "progress": 0.0,
                "capture": 0.0,
                "finish": 0.0,
                "got_captured": 0.0,
                "illegal": 0.0,
                "time_penalty": 0.0,
                "terminal": 0.0,
            }
            while not done:
                # Provide action masks explicitly to predict to avoid any mismatch
                # with wrapper detection across SB3 contrib versions.
                action_masks = None
                if isinstance(observation, dict) and "action_mask" in observation:
                    action_masks = np.asarray(observation["action_mask"]).astype(bool)
                try:
                    action, _ = model.predict(
                        observation,
                        deterministic=deterministic,
                        action_masks=action_masks,
                    )
                except TypeError:
                    # Fallback for older versions that don't support action_masks kwarg
                    action, _ = model.predict(observation, deterministic=deterministic)
                observation, reward, dones, infos = env.step(action)
                episode_reward += float(reward[0])
                steps += 1
                done = bool(dones[0])
                # Pull shaped breakdown components when available
                info = infos[0] if isinstance(infos, (list, tuple)) else infos
                if isinstance(info, dict) and "reward_breakdown" in info:
                    rb = info["reward_breakdown"] or {}
                    for k in episode_breakdown.keys():
                        episode_breakdown[k] += float(rb.get(k, 0.0))
                if done:
                    winner = game.winner
                    if winner is not None and winner.color == agent_color:
                        stats.update(episode_reward, steps, "win")
                        stats.terminal_score_sum += 1.0
                    elif winner is None:
                        stats.update(episode_reward, steps, "draw")
                    else:
                        stats.update(episode_reward, steps, "loss")
                        stats.terminal_score_sum -= 1.0
                    # Aggregate episode breakdown into global sums
                    for k, v in episode_breakdown.items():
                        stats.breakdown_sums[k] += v
    finally:
        env.close()

    return stats


def evaluate_against_many(
    model,
    opponents: Sequence[str],
    games: int,
    base_cfg: EnvConfig,
    deterministic: bool,
) -> List[EvalStats]:
    return [
        evaluate_against(model, opponent, games, base_cfg, deterministic)
        for opponent in opponents
    ]
