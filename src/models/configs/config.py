"""Minimal configuration dataclasses for the simplified Ludo RL setup."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

MATCHUP_TO_OPPONENTS = {"1v1": 1, "1v3": 3}


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
    matchup: str = "1v1"

    def __post_init__(self) -> None:
        matchup = (self.matchup or "1v1").lower()
        if matchup not in MATCHUP_TO_OPPONENTS:
            raise ValueError(
                f"Unsupported matchup '{self.matchup}'. Expected one of: {tuple(MATCHUP_TO_OPPONENTS)}"
            )
        self.matchup = matchup

    @property
    def opponent_count(self) -> int:
        return MATCHUP_TO_OPPONENTS[self.matchup]

    @property
    def player_count(self) -> int:
        return 1 + self.opponent_count


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
