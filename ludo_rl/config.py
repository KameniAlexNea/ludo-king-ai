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
    # Shaping toggles & extras
    enable_capture_shaping: bool = True
    capture_choice_bonus: float = 0.5  # added when a capturing move is chosen
    decline_capture_penalty: float = (
        -0.1
    )  # penalty when capture available but not taken
    enable_progressive_finish: bool = True
    finish_multipliers: List[float] = field(
        default_factory=lambda: [1.0, 1.1, 1.3, 1.8]
    )
    # Scaling / annealing
    capture_reward_scale: float = 1.0  # can be annealed back toward 1.0
    finish_reward_scale: float = 1.0


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
    debug_capture_logging: bool = True
    # Instrumentation
    track_opportunities: bool = True
    log_opportunity_debug: bool = False


@dataclass
class TrainConfig:
    total_steps: int = 5_000_000
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
    eval_baselines: str = ",".join(OpponentConfig().evaluation_candidates)
    # Imitation kickstart
    imitation_enabled: bool = False
    imitation_strategies: str = ",".join(OpponentConfig().evaluation_candidates)
    imitation_steps: int = (
        50_000  # number of environment steps worth of samples to collect
    )
    imitation_batch_size: int = 1024
    imitation_epochs: int = 3
    imitation_entropy_boost: float = 0.01
    # Scheduling / annealing
    entropy_coef_initial: float = 0.1
    entropy_coef_final: float = 0.02
    entropy_anneal_steps: int = 2_000_000
    capture_scale_initial: float = 1.3
    capture_scale_final: float = 1.0
    capture_scale_anneal_steps: int = 1_500_000
