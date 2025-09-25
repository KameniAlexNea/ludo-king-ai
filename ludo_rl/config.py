from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class RewardConfig:
    # Terminal
    # Terminal (win-focused preset)
    win: float = 50.0
    lose: float = -20.0
    draw: float = 0.0
    # Reduce heavy shaping on finishing tokens
    finish_token: float = 6.0
    # Events
    # Per-capture reward reduced (to avoid overvaluing captures vs finishing)
    capture: float = 0.4
    # Being captured is penalized more strongly to discourage unsafe play
    got_captured: float = -3.0
    all_captured: float = -6.0
    # Reward for leaving home (reduced). Will be given only when another token
    # is already active (diversity exit reward).
    exit_start: float = 1.0
    extra_turn: float = 0.1
    # Shaping
    # Reduce shaping so terminal win matters more
    # Reduce shaping so terminal win matters more
    progress_scale: float = 0.01
    safe_zone_reward: float = 0.1
    active_token_bonus: float = 0.001
    inactivity_penalty: float = -0.005
    # Constraints
    illegal_action: float = -1.0
    # Slightly stronger time penalty to discourage long-running episodes
    time_penalty: float = -0.002
    # Shaping toggles & extras
    enable_capture_shaping: bool = True
    capture_choice_bonus: float = 0.005  # small extra for explicit capture choice
    decline_capture_penalty: float = (
        -0.02
    )  # penalty when capture available but not taken
    enable_progressive_finish: bool = True
    finish_multipliers: List[float] = field(
        default_factory=lambda: [1.2, 1.4, 1.7, 2.2]
    )
    # Scaling / annealing
    capture_reward_scale: float = 1.0  # can be annealed back toward 1.0
    finish_reward_scale: float = 1.0


@dataclass
class ObservationConfig:
    include_turn_index: bool = True
    # Encoding choices: prefer normalized floats by default for compactness.
    include_dice_one_hot: bool = False
    include_color_one_hot: bool = False


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
    reward: RewardConfig = field(default_factory=RewardConfig)
    obs: ObservationConfig = field(default_factory=ObservationConfig)
    opponents: OpponentConfig = field(default_factory=OpponentConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    debug_capture_logging: bool = False
    # Instrumentation
    track_opportunities: bool = True
    log_opportunity_debug: bool = False


@dataclass
class TrainConfig:
    total_steps: int = 20_000_000
    n_envs: int = 8
    eval_freq: int = 200_000
    tournament_games: int = 240
    algorithm: str = "maskable_ppo"
    learning_rate: float = 3e-4
    lr_final: float = 8e-5
    n_steps: int = 2048
    batch_size: int = 512
    ent_coef: float = 0.2
    vf_coef: float = 0.5
    logdir: str = "./training/logs"
    model_dir: str = "./training/models"
    max_turns: int = 500
    eval_games: int = 240
    eval_baselines: str = ",".join(OpponentConfig().evaluation_candidates)
    # Imitation kickstart
    imitation_enabled: bool = False
    imitation_strategies: str = ",".join(OpponentConfig().evaluation_candidates)
    imitation_steps: int = (
        50_000  # number of environment steps worth of samples to collect
    )
    imitation_batch_size: int = 1024
    imitation_epochs: int = 5
    imitation_entropy_boost: float = 0.01
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
    checkpoint_prefix: str = "ppo_ludo"
    lr_anneal_enabled: bool = False
    anneal_log_freq: int = 50_000
    env_type: Literal["classic", "selfplay", "hybrid"] = "classic"
    hybrid_switch_rate: float = 0.55

    def __post_init__(self):
        if self.env_type == "selfplay":
            self.n_envs = 1
