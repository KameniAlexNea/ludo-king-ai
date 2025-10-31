from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class IPPOConfig:
    total_rounds: int = 1000              # Number of training cycles over the 4 policies
    steps_per_update: int = 64_000        # Timesteps per policy between updates
    n_envs: int = 16                      # Parallel envs per policy
    learning_rate: float = 3e-4           # PPO LR (with cosine/linear schedule later if desired)
    n_steps: int = 256                    # PPO rollout length per env
    batch_size: int = 256
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    seed: int | None = None
    device: str = "cpu"
    pi_net_arch: Tuple[int, ...] = (128, 128)
    vf_net_arch: Tuple[int, ...] = (256, 256)

    # Evaluation
    eval_every_rounds: int = 10
    eval_games_vs_fixed: int = 50
    eval_fixed_opponents: Tuple[str, ...] = (
        "balanced",
        "killer",
        "cautious",
        "winner",
    )

    # Match composition: fraction of games collected as 1v3 vs 1v1 per policy
    frac_1v3: float = 0.6
    frac_1v1: float = 0.4


AGENT_IDS = ("player_0", "player_1", "player_2", "player_3")
