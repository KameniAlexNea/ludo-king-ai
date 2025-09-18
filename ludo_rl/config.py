from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RewardConfig:
    # Terminal
    win: float = 100.0
    lose: float = -100.0
    finish_token: float = 10.0
    # Events
    capture: float = 8.0
    got_captured: float = -8.0
    extra_turn: float = 3.0
    # Shaping
    progress_scale: float = 0.05
    active_token_bonus: float = 0.1
    inactivity_penalty: float = -0.02
    # Constraints
    illegal_action: float = -5.0
    time_penalty: float = -0.001


@dataclass
class ObservationConfig:
    include_turn_index: bool = True
    include_dice_one_hot: bool = True


@dataclass
class OpponentConfig:
    candidates: List[str] = field(
        default_factory=lambda: [
            "probabilistic_v2",
            "probabilistic_v3",
            "probabilistic",
            "hybrid_prob",
            "killer",
            "cautious",
            "defensive",
            "balanced",
            "winner",
            "optimist",
            "random",
            "weighted_random",
        ]
    )
    evaluation_candidates: List[str] = field(
        default_factory=lambda: [
            "probabilistic_v2",
            "probabilistic_v3",
            "probabilistic",
            "hybrid_prob",
            "killer",
            "cautious",
        ]
    )


@dataclass
class CurriculumConfig:
    enabled: bool = True
    boundaries: List[float] = field(default_factory=lambda: [0.25, 0.6, 0.9])



@dataclass
class EnvConfig:
    max_turns: int = 500
    seed: Optional[int] = None
    randomize_agent: bool = True
    reward: RewardConfig = field(default_factory=RewardConfig)
    obs: ObservationConfig = field(default_factory=ObservationConfig)
    opponents: OpponentConfig = field(default_factory=OpponentConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)


@dataclass
class TrainConfig:
    total_steps: int = 2_000_000
    n_envs: int = 8
    eval_freq: int = 100_000
    tournament_games: int = 240
    algorithm: str = "maskable_ppo"
    learning_rate: float = 1e-4
    n_steps: int = 2048
    batch_size: int = 512
    ent_coef: float = 0.1
    logdir: str = "./training/logs"
    model_dir: str = "./training/models"
    max_turns: int = 500
    eval_games: int = 60
    eval_baselines: str = ",".join(OpponentConfig.evaluation_candidates)
