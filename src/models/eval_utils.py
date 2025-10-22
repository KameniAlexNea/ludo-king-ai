"""Shared evaluation helpers for training callbacks and CLI scripts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Sequence

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

from models.config import EnvConfig
from models.ludo_env import LudoRLEnv


def _mask_fn(env: LudoRLEnv):
    return env.valid_action_mask()


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
        opponent_cfg = EnvConfig(
            max_turns=cfg.max_turns,
            seed=cfg.seed,
            randomize_agent=cfg.randomize_agent,
            fixed_agent_color=cfg.fixed_agent_color,
            opponent_strategy=opponent,
            reward=cfg.reward,
            obs=cfg.obs,
        )
        return ActionMasker(LudoRLEnv(opponent_cfg), _mask_fn)

    return DummyVecEnv([_init])


def evaluate_against(
    model,
    opponent: str,
    games: int,
    base_cfg: EnvConfig,
    deterministic: bool,
) -> EvalStats:
    env = build_eval_env(opponent, deepcopy(base_cfg))
    stats = EvalStats(opponent=opponent, episodes=games)

    try:
        for _ in range(games):
            observation = env.reset()
            done = False
            episode_reward = 0.0
            steps = 0
            while not done:
                action, _ = model.predict(observation, deterministic=deterministic)
                observation, reward, dones, infos = env.step(action)
                episode_reward += float(reward[0])
                steps += 1
                done = bool(dones[0])
                if done:
                    base_env: LudoRLEnv = env.envs[0].unwrapped
                    winner = getattr(base_env.game, "winner", None)
                    if winner is not None and winner.color == base_env.agent_color:
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
