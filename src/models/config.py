"""Minimal configuration dataclasses for the simplified Ludo RL setup."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RewardConfig:
    """Reward shaping coefficients for the simplified environment."""

    win: float = 50.0
    lose: float = -50.0
    draw: float = 0.0
    finish_token: float = 5.0
    capture: float = 2.0
    got_captured: float = -2.0
    illegal_action: float = -5.0
    time_penalty: float = -0.02
    progress_scale: float = 1.0


@dataclass
class ObservationConfig:
    """Options that drive how the observation vector is built."""

    discrete: bool = False


@dataclass
class EnvConfig:
    """Runtime parameters for creating the classic Ludo environment."""

    max_turns: int = 300
    seed: Optional[int] = None
    randomize_agent: bool = True
    fixed_agent_color: Optional[str] = None
    opponent_strategy: str = "probabilistic_v3"
    reward: RewardConfig = field(default_factory=RewardConfig)
    obs: ObservationConfig = field(default_factory=ObservationConfig)


@dataclass
class TrainConfig:
    """High level hyper-parameters for PPO training."""

    total_steps: int = 500_000
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 256
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    logdir: str = "./training/logs"
    model_dir: str = "./training/models"
    seed: Optional[int] = None
    device: str = "cpu"
