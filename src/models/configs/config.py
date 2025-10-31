"""Minimal configuration dataclasses for the simplified Ludo RL setup."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


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
    time_penalty: float = -0.01
    progress_scale: float = 1.0


@dataclass
class ObservationConfig:
    """Options that drive how the observation vector is built."""

    discrete: bool = False


@dataclass
class EnvConfig:
    """Runtime parameters for creating the classic Ludo environment."""

    max_turns: int = 250
    seed: Optional[int] = None
    randomize_agent: bool = True
    fixed_agent_color: Optional[str] = None
    opponent_strategy: str = "probabilistic_v3,probabilistic_v2,probabilistic,hybrid_prob,killer,cautious,defensive,balanced,winner,optimist,random,weighted_random"
    reward: RewardConfig = field(default_factory=RewardConfig)
    obs: ObservationConfig = field(default_factory=ObservationConfig)
    multi_agent: bool = False


@dataclass
class TrainConfig:
    """High level hyper-parameters for PPO training."""

    total_steps: int = 5_000_000
    learning_rate: float = 5e-5
    n_steps: int = 256
    batch_size: int = 256
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    logdir: str = "./training/logs"
    model_dir: str = "./training/models"
    seed: Optional[int] = None
    device: str = "cpu"
    save_steps: int = 100_000
    pi_net_arch: Tuple[int, ...] = (64, 64)
    vf_net_arch: Tuple[int, ...] = (256, 256)
    n_envs: int = 16
    eval_freq: int = 100_000
    eval_episodes: int = 20
    eval_opponents: tuple[str, ...] = (
        "probabilistic_v3",
        "balanced",
        "killer",
        "cautious",
        "winner",
    )
    eval_deterministic: bool = True


@dataclass
class MultiAgentConfig:
    """Configuration specific to multi-agent training."""

    # Policy sharing: all agents share the same policy network
    shared_policy: bool = True

    # Self-play configuration
    enable_self_play: bool = True
    opponent_pool_size: int = 5  # Keep last N model versions as opponents
    save_opponent_freq: int = 250_000  # Save opponent every N timesteps
    self_play_num_opponents: int = 3  # Number of agents to assign from opponent pool
    self_play_opponent_fallback: str = (
        "balanced"  # Fallback scripted bot if no opponents
    )

    # Opponent mix during training
    # If True, use mix of self-play + fixed opponents
    # If False, only self-play
    use_fixed_opponents: bool = True
    fixed_opponent_ratio: float = 0.3  # 30% fixed opponents, 70% self-play
    fixed_opponent_strategies: tuple[str, ...] = (
        "probabilistic_v3",
        "killer",
        "balanced",
        "cautious",
    )  # Scripted strategies to mix with self-play

    # Opponent model inference options
    opponent_model_device: str = "cpu"  # Load opponent models on CPU to save VRAM
    opponent_stochastic_prob: float = 0.1  # Chance to use stochastic actions for opponents
