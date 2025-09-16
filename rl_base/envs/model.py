from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ludo.constants import Colors


@dataclass
class BaseRewardConfig:
    """Base reward configuration for strategic Ludo play.

    Values are tuned for stable RL training with proper reward shaping.
    Terminal rewards are high to provide clear learning signals.
    Action rewards are moderate to encourage good play without dominating.
    """

    # Terminal rewards (highest priority - sparse but strong signals)
    win: float = 100.0  # Game victory - primary objective
    lose: float = -150.0  # Game loss - symmetric penalty
    finish_token: float = 10.0  # Finishing individual tokens - important milestone
    all_tokens_killed: float = -25.0  # All tokens captured - severe penalty

    # Major action rewards (should encourage key strategic actions)
    capture: float = 8.0  # Capturing opponent tokens - very valuable
    got_captured: float = -8.0  # Being captured - symmetric penalty
    extra_turn: float = 3.0  # Rolling 6 for another turn - useful but not dominant

    # Strategic positioning rewards (moderate shaping)
    home_progress_bonus: float = 2.0  # Progress in home column
    blocking_bonus: float = 1.5  # Creating blocks to protect position
    safety_bonus: float = 2.0  # Moving to safe positions
    home_approach_bonus: float = 1.0  # Approaching home entry

    # Small continuous rewards (minimal shaping to avoid reward hacking)
    progress_scale: float = 0.05  # General board progress
    diversity_bonus: float = 0.5  # Encouraging token activation
    active_token_bonus: float = 0.1  # Bonus per active token
    inactivity_penalty: float = -0.02  # Penalty per token stuck at home

    # Clear boundaries (strong penalties for invalid actions)
    illegal_action: float = -15.0  # Invalid moves - strong negative
    time_penalty: float = -0.005  # Small efficiency penalty

    # Advanced features (disabled by default for simplicity)
    use_probabilistic_rewards: bool = False
    risk_weight: float = 0.2


@dataclass
class ObservationConfig:
    include_blocking_count: bool = True
    include_turn_index: bool = True
    include_raw_dice: bool = True
    normalize_positions: bool = True


@dataclass
class OpponentsConfig:
    candidates: List[str] = field(
        default_factory=lambda: [
            "killer",
            "winner",
            "optimist",
            "balanced",
            "defensive",
            "random",
            "weighted_random",
            "cautious",
            "probabilistic",
            "probabilistic_v2",
            "probabilistic_v3",
            "hybrid_prob",
        ]
    )


@dataclass
class OpponentCurriculumConfig:
    """Config for progressive opponent sampling during training.

    The curriculum exposes tiers of opponents and episode thresholds. Sampling pool
    grows cumulatively with each phase: at phase i, we sample from the union of
    tiers[0..i]. This yields an easy->hard progression without needing global
    training step coordination across vectorized environments.
    """

    enabled: bool = True
    # Use training progress (0..1) instead of local episode count
    use_progress: bool = True
    # Tiers ordered from easiest to hardest (names must match StrategyFactory keys)
    tiers: List[List[str]] = field(
        default_factory=lambda: [
            # Easy starters (weak baselines)
            ["random", "weighted_random", "optimist"],
            # Moderate heuristics
            ["winner", "defensive"],
            # Strong mid-tier probabilistic/aggro
            ["killer", "hybrid_prob", "probabilistic_v2", "probabilistic_v3"],
            # Top strategies
            ["balanced", "probabilistic", "cautious"],
        ]
    )
    # Episode thresholds per environment instance; must be same length as tiers.
    # Example fallback when progress is not provided: first 100 episodes -> tier 0; 100-300 -> include tier 1; 300-600 -> include tier 2; 600+ -> include tier 3
    phase_episodes: List[int] = field(default_factory=lambda: [100, 300, 600, 10_000])
    # Progress boundaries (fractions 0..1). Phases determined by where progress falls:
    # [0, b0) -> phase 0; [b0, b1) -> phase 1; [b1, b2) -> phase 2; [b2, 1] -> phase 3
    progress_boundaries: List[float] = field(default_factory=lambda: [0.1, 0.4, 0.8])

    # Difficulty buckets for controlled sampling
    poor: List[str] = field(
        default_factory=lambda: [
            "random",
            "weighted_random",
            "optimist",
        ]
    )
    medium: List[str] = field(
        default_factory=lambda: [
            "winner",
            "defensive",
            "hybrid_prob",
            "probabilistic_v2",
        ]
    )
    hard: List[str] = field(
        default_factory=lambda: [
            "balanced",
            "probabilistic",
            "cautious",
            "killer",
            "probabilistic_v3",
        ]
    )


@dataclass
class BaseEnvConfig:
    """Base environment configuration shared between RL implementations."""

    agent_color: str = Colors.RED
    max_turns: int = 1000
    obs_cfg: ObservationConfig = field(default_factory=ObservationConfig)
    seed: Optional[int] = None
