"""
Opponent lineup sampling utilities for curriculum learning.

This module provides flexible opponent selection strategies that:
1. Start with mixed opponents from the beginning (not homogeneous)
2. Sample different opponent mixes each episode for diversity
3. Use soft curriculum blending instead of hard cutoffs
4. Support strategy weighting based on difficulty
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from loguru import logger


@dataclass
class StrategyWeight:
    """Weight configuration for a strategy in the opponent pool."""

    name: str
    base_weight: float = 1.0  # Base sampling probability weight
    difficulty: float = 0.5  # 0.0 = easiest, 1.0 = hardest
    min_curriculum_stage: float = 0.0  # When this strategy becomes available (0-1)


# Default strategy difficulty ratings based on arena results
# Higher difficulty = harder to beat
DEFAULT_STRATEGY_WEIGHTS: Dict[str, StrategyWeight] = {
    "random": StrategyWeight(
        "random", base_weight=1.0, difficulty=0.0, min_curriculum_stage=0.0
    ),
    "killer": StrategyWeight(
        "killer", base_weight=1.0, difficulty=0.3, min_curriculum_stage=0.0
    ),
    "hoarder": StrategyWeight(
        "hoarder", base_weight=1.0, difficulty=0.5, min_curriculum_stage=0.1
    ),
    "homebody": StrategyWeight(
        "homebody", base_weight=1.0, difficulty=0.55, min_curriculum_stage=0.1
    ),
    "cautious": StrategyWeight(
        "cautious", base_weight=1.0, difficulty=0.6, min_curriculum_stage=0.2
    ),
    "defensive": StrategyWeight(
        "defensive", base_weight=1.0, difficulty=0.65, min_curriculum_stage=0.2
    ),
    "hybrid": StrategyWeight(
        "hybrid", base_weight=1.0, difficulty=0.7, min_curriculum_stage=0.3
    ),
}


@dataclass
class OpponentLineupSampler:
    """
    Samples opponent lineups with curriculum-aware diversity.

    Key features:
    - Always mixes strategies (no homogeneous opponents)
    - Each episode samples a fresh lineup
    - Soft curriculum: gradually increases average difficulty
    - Maintains diversity throughout training

    Note on multi-env training:
    - When using vectorized environments (n_envs > 1), each env has its own sampler
    - Progress should be synced via `set_global_timesteps()` from a callback
    - This ensures curriculum progresses based on total training steps, not per-env resets
    """

    available_strategies: List[str] = field(default_factory=list)
    strategy_weights: Dict[str, StrategyWeight] = field(default_factory=dict)

    # Curriculum settings - use timesteps for multi-env compatibility
    curriculum_total_timesteps: int = 50_000_000  # Total timesteps for full curriculum
    min_difficulty: float = 0.2  # Starting average difficulty
    max_difficulty: float = 0.8  # Ending average difficulty

    # Diversity settings
    force_diversity: bool = True  # Ensure at least 2 different strategies
    diversity_bonus: float = 0.3  # Bonus weight for underrepresented strategies

    # Caching
    cache_interval: int = 1000  # Resample lineup every N resets (0 = every reset)

    # Internal state
    _rng: random.Random = field(default_factory=random.Random, repr=False)
    _reset_count: int = field(default=0, repr=False)
    _global_timesteps: int = field(
        default=0, repr=False
    )  # Synced from training callback
    _cached_lineup: Optional[List[str]] = field(default=None, repr=False)
    _strategy_usage: Dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        # Initialize strategy weights for available strategies
        if not self.strategy_weights:
            self.strategy_weights = {}
            for name in self.available_strategies:
                if name in DEFAULT_STRATEGY_WEIGHTS:
                    self.strategy_weights[name] = DEFAULT_STRATEGY_WEIGHTS[name]
                else:
                    # Unknown strategy gets medium difficulty
                    self.strategy_weights[name] = StrategyWeight(
                        name, base_weight=1.0, difficulty=0.5, min_curriculum_stage=0.0
                    )

        # Initialize usage tracking
        for name in self.available_strategies:
            self._strategy_usage[name] = 0

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self._rng.seed(seed)

    def set_global_timesteps(self, timesteps: int) -> None:
        """
        Update curriculum progress based on global training timesteps.

        This should be called from a training callback to sync progress
        across all parallel environments. Each env's sampler will then
        use this global count for curriculum progression.

        Args:
            timesteps: Total timesteps from model.num_timesteps
        """
        self._global_timesteps = timesteps

    def get_curriculum_progress(self) -> float:
        """
        Returns curriculum progress as a float in [0, 1].

        Uses global timesteps if set (multi-env training), otherwise
        falls back to local reset count (single env / testing).
        """
        if self.curriculum_total_timesteps <= 0:
            return 1.0
        # Use global timesteps for progress (synced from training callback)
        return min(1.0, self._global_timesteps / self.curriculum_total_timesteps)

    def get_target_difficulty(self) -> float:
        """Returns the target average difficulty for current curriculum stage."""
        progress = self.get_curriculum_progress()
        # Smooth sigmoid-like curve for difficulty ramp
        # Starts slow, accelerates in middle, slows at end
        smoothed = progress**0.7  # Slightly faster early ramp
        return (
            self.min_difficulty + (self.max_difficulty - self.min_difficulty) * smoothed
        )

    def _get_available_for_stage(self) -> List[str]:
        """Get strategies available at current curriculum stage."""
        progress = self.get_curriculum_progress()
        available = []
        for name in self.available_strategies:
            weight = self.strategy_weights.get(name)
            if weight is None:
                available.append(name)
            elif progress >= weight.min_curriculum_stage:
                available.append(name)
        return available if available else self.available_strategies

    def _compute_sampling_weights(self, available: List[str]) -> List[float]:
        """Compute sampling weights based on difficulty target and diversity."""
        target_diff = self.get_target_difficulty()
        weights = []

        # Track total usage for diversity bonus
        total_usage = sum(self._strategy_usage.values()) + 1

        for name in available:
            sw = self.strategy_weights.get(name)
            if sw is None:
                base = 1.0
                diff = 0.5
            else:
                base = sw.base_weight
                diff = sw.difficulty

            # Weight based on how close to target difficulty
            # Prefer strategies near target, but don't exclude others
            diff_delta = abs(diff - target_diff)
            diff_weight = max(0.1, 1.0 - diff_delta)

            # Diversity bonus for underused strategies
            usage_ratio = self._strategy_usage.get(name, 0) / total_usage
            diversity_weight = 1.0 + self.diversity_bonus * (
                1.0 - usage_ratio * len(available)
            )

            weights.append(base * diff_weight * diversity_weight)

        # Normalize
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(available)] * len(available)

        return weights

    def sample_lineup(self, num_opponents: int) -> List[str]:
        """
        Sample a diverse opponent lineup for the current curriculum stage.

        Args:
            num_opponents: Number of opponent slots to fill

        Returns:
            List of strategy names for each opponent slot
        """
        if num_opponents <= 0:
            return []

        # Check cache
        if (
            self.cache_interval > 0
            and self._cached_lineup is not None
            and len(self._cached_lineup) == num_opponents
            and self._reset_count % self.cache_interval != 0
        ):
            return self._cached_lineup

        available = self._get_available_for_stage()
        if not available:
            return ["random"] * num_opponents

        weights = self._compute_sampling_weights(available)
        lineup: List[str] = []

        # Sample with diversity enforcement
        used_in_lineup: set[str] = set()

        for i in range(num_opponents):
            # Adjust weights to encourage diversity
            adjusted_weights = weights.copy()

            if self.force_diversity and len(used_in_lineup) < min(len(available), 2):
                # Reduce weight of already-used strategies
                for j, name in enumerate(available):
                    if name in used_in_lineup:
                        adjusted_weights[j] *= 0.3

                # Renormalize
                total = sum(adjusted_weights)
                if total > 0:
                    adjusted_weights = [w / total for w in adjusted_weights]

            # Sample
            choice = self._rng.choices(available, weights=adjusted_weights, k=1)[0]
            lineup.append(choice)
            used_in_lineup.add(choice)

        # Update usage tracking
        for name in lineup:
            self._strategy_usage[name] = self._strategy_usage.get(name, 0) + 1

        self._cached_lineup = lineup
        return lineup

    def advance(self) -> None:
        """Advance the reset counter (call after each env.reset())."""
        self._reset_count += 1

    def get_lineup(self, num_opponents: int) -> List[str]:
        """
        Get opponent lineup and advance counter.

        This is the main interface - combines sample_lineup + advance.
        """
        lineup = self.sample_lineup(num_opponents)
        self.advance()
        return lineup

    def should_log(self, interval: int = 10000) -> bool:
        """Check if we should log lineup info at this reset."""
        return self._reset_count > 0 and self._reset_count % interval == 0

    def get_stats(self) -> Dict:
        """Get current sampler statistics for logging."""
        return {
            "reset_count": self._reset_count,
            "curriculum_progress": self.get_curriculum_progress(),
            "target_difficulty": self.get_target_difficulty(),
            "strategy_usage": dict(self._strategy_usage),
            "available_strategies": self._get_available_for_stage(),
        }

    def log_stats(self, interval: int = 50000) -> None:
        """Log statistics at regular intervals."""
        if self.should_log(interval):
            stats = self.get_stats()
            logger.info(
                f"Opponent sampler at reset {stats['reset_count']}: "
                f"progress={stats['curriculum_progress']:.2%}, "
                f"target_diff={stats['target_difficulty']:.2f}, "
                f"available={stats['available_strategies']}"
            )


def create_default_sampler(
    strategies: Sequence[str],
    curriculum_timesteps: int = 50_000_000,
    seed: Optional[int] = None,
) -> OpponentLineupSampler:
    """
    Factory function to create a sampler with sensible defaults.

    Args:
        strategies: List of available strategy names
        curriculum_timesteps: Total timesteps for full curriculum progression
        seed: Random seed for reproducibility

    Returns:
        Configured OpponentLineupSampler instance
    """
    sampler = OpponentLineupSampler(
        available_strategies=list(strategies),
        curriculum_total_timesteps=curriculum_timesteps,
        min_difficulty=0.15,  # Start with mostly easy opponents
        max_difficulty=0.75,  # End with challenging mix
        force_diversity=True,
        diversity_bonus=0.4,
        cache_interval=1000,  # Resample every 1000 resets
    )

    if seed is not None:
        sampler.set_seed(seed)

    return sampler
