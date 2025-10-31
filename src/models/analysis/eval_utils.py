"""Shared evaluation helpers for training callbacks and CLI scripts."""

from __future__ import annotations

import copy
from copy import deepcopy
from dataclasses import dataclass
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

    def __init__(self, env: gym.Env, num_agents: int = 4):
        super().__init__(env)
        base_space = get_flat_space_config()
        action_dim = int(env.action_space.n)
        self._mask_shape = (action_dim,)
        self._obs_keys = list(get_space_config().spaces.keys())
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=base_space.low,
                    high=base_space.high,
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

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        mask = self._extract_mask(info)
        converted = self._convert_obs(obs, mask)
        info = {**info, "action_mask": mask.copy()}
        return converted, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        mask = self._extract_mask(info)
        converted = self._convert_obs(obs, mask)
        info = {**info, "action_mask": mask.copy()}
        return converted, reward, terminated, truncated, info

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

    def as_dict(self) -> Dict[str, float]:
        return {
            "opponent": self.opponent,
            "episodes": float(self.episodes),
            "win_rate": self.win_rate,
            "loss_rate": self.losses / max(1, self.episodes),
            "draw_rate": self.draws / max(1, self.episodes),
            "avg_reward": self.avg_reward,
            "avg_length": self.avg_length,
        }


def build_eval_env(opponent: str, cfg: EnvConfig) -> DummyVecEnv:
    def _init():
        opponent_cfg = copy.deepcopy(cfg)
        opponent_cfg.opponent_strategy = opponent
        env = LudoRLEnv(opponent_cfg)
        env = ActionMasker(env, _mask_fn)
        if cfg.multi_agent:
            env = SharedPolicyEvalEnv(env)
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
            while not done:
                action, _ = model.predict(observation, deterministic=deterministic)
                observation, reward, dones, infos = env.step(action)
                episode_reward += float(reward[0])
                steps += 1
                done = bool(dones[0])
                if done:
                    winner = game.winner
                    if winner is not None and winner.color == agent_color:
                        stats.update(episode_reward, steps, "win")
                    elif winner is None:
                        stats.update(episode_reward, steps, "draw")
                    else:
                        stats.update(episode_reward, steps, "loss")
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
