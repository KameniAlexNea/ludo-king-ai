import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class RewardConfig:
    # Terminal
    # Terminal (win-focused preset)
    win: float = 100.0
    lose: float = -100.0
    draw: float = -30.0
    # Per-token rewards
    finish_token: float = 1.0
    # Events
    # Per-capture reward increased
    capture: float = 0.3
    # Being captured is penalized more strongly to discourage unsafe play
    got_captured: float = -0.5
    all_captured: float = -1.0
    # Reward for leaving home and diversity bonus
    exit_start: float = 0.2
    diversity_bonus: float = 0.01
    extra_turn: float = 0.05
    # Shaping
    # Increase shaping to make rewards denser
    progress_scale: float = 0.005
    safe_zone_reward: float = 0.2
    # Constraints
    illegal_action: float = -0.2
    # Reduce time penalty to encourage longer games if needed, but keep small
    time_penalty: float = -0.01
    # reward signal function
    reward_type: Literal["sparse", "merged", "risk_opportunity"] = os.getenv(
        "REWARD_TYPE", "sparse"
    )  # "sparse", "merged", "risk_opportunity"


@dataclass
class ObservationConfig:
    # Encoding choices: prefer normalized floats by default for compactness.
    include_dice_one_hot: bool = True  # always True, not used
    include_color_one_hot: bool = True  # always True, not used
    # Use discrete encoding (MultiDiscrete) instead of continuous Box
    discrete: bool = os.getenv("DISCRETE_OBS", "false").lower() == "true"


@dataclass
class OpponentConfig:
    candidates: List[str] = field(
        default_factory=lambda: [
            "random",
            "weighted_random",
            "optimist",
            "probabilistic_v2",
            "probabilistic_v3",
            "probabilistic",
            "hybrid_prob",
            "killer",
            "cautious",
            "defensive",
            "balanced",
            "winner",
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
    boundaries: List[float] = field(default_factory=lambda: [0.1, 0.4, 0.7])


@dataclass
class EnvConfig:
    max_turns: int = 500
    seed: Optional[int] = None
    randomize_agent: bool = True
    # If set, forces env to use this many players (e.g., 2 or 4). If None, a
    # player count will be sampled per-reset from `allowed_player_counts`.
    fixed_num_players: Optional[int] = (
        int(os.getenv("FIXED_NUM_PLAYERS")) if os.getenv("FIXED_NUM_PLAYERS") else None
    )
    # Allowed player counts to sample from when `fixed_num_players` is None.
    allowed_player_counts: List[int] = field(
        default_factory=lambda: list(
            map(int, os.getenv("ALLOWED_PLAYER_COUNTS", "2,4").split(","))
        )
    )
    reward: RewardConfig = field(default_factory=RewardConfig)
    obs: ObservationConfig = field(default_factory=ObservationConfig)
    opponents: OpponentConfig = field(default_factory=OpponentConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    # Instrumentation
    track_opportunities: bool = True


@dataclass
class TrainConfig:
    total_steps: int = 20_000_000
    n_envs: int = 8
    eval_freq: int = 200_000
    algorithm: str = "maskable_ppo"
    learning_rate: float = 3e-4
    lr_final: float = 8e-5
    n_steps: int = 2048
    batch_size: int = 512
    ent_coef: float = 0.2
    vf_coef: float = 0.5
    gamma: float = 0.999
    gae_lambda: float = 0.95
    logdir: str = "./training/logs"
    model_dir: str = "./training/models"
    max_turns: int = 500
    eval_games: int = 240
    eval_baselines: str = ",".join(OpponentConfig().evaluation_candidates)
    # Scheduling / annealing
    use_entropy_annealing: bool = False
    entropy_coef_initial: float = 0.5
    entropy_coef_final: float = 0.3
    entropy_anneal_steps: int = 3_000_000
    capture_scale_initial: float = 1.3
    capture_scale_final: float = 1.1
    capture_scale_anneal_steps: int = 1_500_000
    # Additional training options
    checkpoint_freq: int = 100_000
    save_freq: int = 100_000
    checkpoint_prefix: str = "ppo_ludo"
    lr_anneal_enabled: bool = False
    anneal_log_freq: int = 50_000
    env_type: Literal["classic", "selfplay", "hybrid"] = "classic"
    hybrid_switch_rate: float = 0.55
    # embedding dimension for discrete observation extractor
    embed_dim: int = 32
    load_model: Optional[str] = None  # path to model to load

    def __post_init__(self):
        if self.env_type in ["selfplay", "hybrid"]:
            self.n_envs = 1
