"""Minimal evaluation script for running a trained policy against scripted opponents."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Sequence

from sb3_contrib import MaskablePPO
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


def _build_env(opponent: str, cfg: EnvConfig) -> DummyVecEnv:
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


def _evaluate_against(
    model: MaskablePPO,
    opponent: str,
    games: int,
    base_cfg: EnvConfig,
    deterministic: bool,
) -> EvalStats:
    env = _build_env(opponent, base_cfg)
    stats = EvalStats(opponent=opponent, episodes=games)

    try:
        for _ in range(games):
            obs = env.reset()
            done = False
            episode_reward = 0.0
            steps = 0
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, dones, infos = env.step(action)
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO Ludo model.")
    parser.add_argument(
        "model", type=str, help="Path to the saved MaskablePPO model .zip file."
    )
    parser.add_argument(
        "--opponents",
        type=str,
        default="probabilistic_v3,killer,cautious",
        help="Comma-separated list of opponent strategies to evaluate against.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=20,
        help="Number of evaluation games per opponent.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=300,
        help="Maximum turns per episode before declaring a draw.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed for deterministic evaluation.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy actions during evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load the model on (cpu or cuda).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    opponents = [op.strip() for op in args.opponents.split(",") if op.strip()]
    if not opponents:
        raise ValueError("At least one opponent strategy must be provided.")

    model: MaskablePPO = MaskablePPO.load(args.model, device=args.device)
    print(model.policy)
    env_cfg = EnvConfig(max_turns=args.max_turns, seed=args.seed)

    summaries: Sequence[EvalStats] = [
        _evaluate_against(model, opponent, args.games, env_cfg, args.deterministic)
        for opponent in opponents
    ]

    print("Opponent,Games,WinRate,LossRate,DrawRate,AvgReward,AvgLength")
    for summary in summaries:
        data = summary.as_dict()
        print(
            f"{data['opponent']},{int(data['episodes'])},"
            f"{data['win_rate']:.3f},{data['loss_rate']:.3f},"
            f"{data['draw_rate']:.3f},{data['avg_reward']:.2f},"
            f"{data['avg_length']:.1f}"
        )


if __name__ == "__main__":
    main()
