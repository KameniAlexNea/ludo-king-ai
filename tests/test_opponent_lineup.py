"""Tests for the opponent lineup sampling utilities."""

import pytest

from ludo_rl.utils.opponent_lineup import (
    DEFAULT_STRATEGY_WEIGHTS,
    OpponentLineupSampler,
    StrategyWeight,
    create_default_sampler,
)


class TestStrategyWeight:
    """Tests for StrategyWeight dataclass."""

    def test_default_values(self):
        """StrategyWeight should have sensible defaults."""
        sw = StrategyWeight("test")
        assert sw.name == "test"
        assert sw.base_weight == 1.0
        assert sw.difficulty == 0.5
        assert sw.min_curriculum_stage == 0.0

    def test_custom_values(self):
        """StrategyWeight should accept custom values."""
        sw = StrategyWeight(
            "hard", base_weight=2.0, difficulty=0.9, min_curriculum_stage=0.5
        )
        assert sw.name == "hard"
        assert sw.base_weight == 2.0
        assert sw.difficulty == 0.9
        assert sw.min_curriculum_stage == 0.5


class TestDefaultStrategyWeights:
    """Tests for the DEFAULT_STRATEGY_WEIGHTS configuration."""

    def test_random_is_easiest(self):
        """Random strategy should have difficulty 0.0."""
        assert DEFAULT_STRATEGY_WEIGHTS["random"].difficulty == 0.0

    def test_all_strategies_have_weights(self):
        """All default strategies should be configured."""
        expected = {
            "random",
            "killer",
            "hoarder",
            "homebody",
            "cautious",
            "defensive",
            "hybrid",
        }
        assert expected <= set(DEFAULT_STRATEGY_WEIGHTS.keys())

    def test_difficulty_ordering(self):
        """Difficulties should generally increase from random to hybrid."""
        assert (
            DEFAULT_STRATEGY_WEIGHTS["random"].difficulty
            < DEFAULT_STRATEGY_WEIGHTS["killer"].difficulty
        )
        assert (
            DEFAULT_STRATEGY_WEIGHTS["killer"].difficulty
            < DEFAULT_STRATEGY_WEIGHTS["hybrid"].difficulty
        )


class TestOpponentLineupSampler:
    """Tests for OpponentLineupSampler class."""

    @pytest.fixture
    def basic_sampler(self):
        """Create a basic sampler with common strategies."""
        return OpponentLineupSampler(
            available_strategies=["random", "killer", "defensive"],
            curriculum_total_resets=100,
        )

    @pytest.fixture
    def seeded_sampler(self):
        """Create a sampler with fixed seed for reproducibility."""
        sampler = OpponentLineupSampler(
            available_strategies=["random", "killer", "defensive", "cautious"],
            curriculum_total_resets=1000,
        )
        sampler.set_seed(42)
        return sampler

    # --- Initialization Tests ---

    def test_init_default_weights(self, basic_sampler):
        """Sampler should initialize with default weights for known strategies."""
        assert "random" in basic_sampler.strategy_weights
        assert basic_sampler.strategy_weights["random"].difficulty == 0.0

    def test_init_unknown_strategy_gets_medium_difficulty(self):
        """Unknown strategies should get medium difficulty (0.5)."""
        sampler = OpponentLineupSampler(
            available_strategies=["random", "my_custom_strategy"],
        )
        assert sampler.strategy_weights["my_custom_strategy"].difficulty == 0.5

    def test_init_usage_tracking(self, basic_sampler):
        """Usage tracking should be initialized for all strategies."""
        for name in basic_sampler.available_strategies:
            assert name in basic_sampler._strategy_usage
            assert basic_sampler._strategy_usage[name] == 0

    # --- Curriculum Progress Tests ---

    def test_get_curriculum_progress_initial(self, basic_sampler):
        """Initial progress should be 0."""
        assert basic_sampler.get_curriculum_progress() == 0.0

    def test_get_curriculum_progress_after_advance(self, basic_sampler):
        """Progress should increase after advance()."""
        basic_sampler.advance()
        assert basic_sampler.get_curriculum_progress() == 0.01  # 1/100

    def test_get_curriculum_progress_capped_at_one(self, basic_sampler):
        """Progress should be capped at 1.0."""
        basic_sampler._reset_count = 200  # More than total
        assert basic_sampler.get_curriculum_progress() == 1.0

    def test_get_curriculum_progress_zero_total(self):
        """Progress should be 1.0 when curriculum_total_resets is 0."""
        sampler = OpponentLineupSampler(
            available_strategies=["random"],
            curriculum_total_resets=0,
        )
        assert sampler.get_curriculum_progress() == 1.0

    # --- Target Difficulty Tests ---

    def test_get_target_difficulty_initial(self, basic_sampler):
        """Initial difficulty should be min_difficulty."""
        assert basic_sampler.get_target_difficulty() == basic_sampler.min_difficulty

    def test_get_target_difficulty_increases(self, basic_sampler):
        """Target difficulty should increase with progress."""
        initial = basic_sampler.get_target_difficulty()
        basic_sampler._reset_count = 50
        mid = basic_sampler.get_target_difficulty()
        basic_sampler._reset_count = 100
        final = basic_sampler.get_target_difficulty()

        assert initial < mid < final

    def test_get_target_difficulty_final(self, basic_sampler):
        """Final difficulty should approach max_difficulty."""
        basic_sampler._reset_count = basic_sampler.curriculum_total_resets
        final = basic_sampler.get_target_difficulty()
        assert abs(final - basic_sampler.max_difficulty) < 0.01

    # --- Available Strategies Tests ---

    def test_get_available_for_stage_initial(self):
        """Only strategies with min_curriculum_stage=0 should be available initially."""
        sampler = OpponentLineupSampler(
            available_strategies=["random", "killer", "hybrid"],
            curriculum_total_resets=100,
        )
        available = sampler._get_available_for_stage()
        assert "random" in available
        assert "killer" in available
        # hybrid has min_curriculum_stage=0.3, so not available at start
        assert "hybrid" not in available

    def test_get_available_for_stage_late(self):
        """All strategies should be available late in training."""
        sampler = OpponentLineupSampler(
            available_strategies=["random", "killer", "hybrid"],
            curriculum_total_resets=100,
        )
        sampler._reset_count = 100  # 100% progress
        available = sampler._get_available_for_stage()
        assert "random" in available
        assert "killer" in available
        assert "hybrid" in available

    def test_get_available_fallback(self):
        """Should return all strategies if none available (edge case)."""
        sampler = OpponentLineupSampler(
            available_strategies=["late_strategy"],
            strategy_weights={
                "late_strategy": StrategyWeight(
                    "late_strategy", min_curriculum_stage=1.0
                )
            },
            curriculum_total_resets=100,
        )
        # At 0% progress, no strategies available, should fallback
        available = sampler._get_available_for_stage()
        assert "late_strategy" in available

    # --- Sampling Tests ---

    def test_sample_lineup_returns_correct_count(self, seeded_sampler):
        """sample_lineup should return the requested number of opponents."""
        lineup = seeded_sampler.sample_lineup(3)
        assert len(lineup) == 3

    def test_sample_lineup_zero_opponents(self, seeded_sampler):
        """sample_lineup with 0 opponents should return empty list."""
        assert seeded_sampler.sample_lineup(0) == []

    def test_sample_lineup_negative_opponents(self, seeded_sampler):
        """sample_lineup with negative opponents should return empty list."""
        assert seeded_sampler.sample_lineup(-1) == []

    def test_sample_lineup_diversity(self, seeded_sampler):
        """With force_diversity=True, lineup should have at least 2 different strategies."""
        # Sample many times to check diversity
        for _ in range(10):
            seeded_sampler._cached_lineup = None  # Force resample
            lineup = seeded_sampler.sample_lineup(3)
            unique = set(lineup)
            assert len(unique) >= 2, f"Lineup {lineup} lacks diversity"

    def test_sample_lineup_no_diversity(self):
        """With force_diversity=False, homogeneous lineups are possible."""
        sampler = OpponentLineupSampler(
            available_strategies=["random"],  # Only one strategy
            force_diversity=False,
            curriculum_total_resets=100,
        )
        lineup = sampler.sample_lineup(3)
        assert lineup == ["random", "random", "random"]

    def test_sample_lineup_uses_cache(self, seeded_sampler):
        """Lineup should be cached and reused within cache_interval."""
        seeded_sampler.cache_interval = 10
        lineup1 = seeded_sampler.sample_lineup(3)
        seeded_sampler._reset_count = 1  # Not at interval boundary
        lineup2 = seeded_sampler.sample_lineup(3)
        assert lineup1 == lineup2

    def test_sample_lineup_cache_invalidated_at_interval(self, seeded_sampler):
        """Lineup should be resampled at cache_interval boundaries."""
        seeded_sampler.cache_interval = 10
        seeded_sampler._cached_lineup = ["old", "lineup", "here"]
        seeded_sampler._reset_count = 10  # At interval boundary

        lineup = seeded_sampler.sample_lineup(3)
        assert lineup != ["old", "lineup", "here"]

    def test_sample_lineup_updates_usage(self, basic_sampler):
        """sample_lineup should update strategy usage tracking."""
        initial_usage = sum(basic_sampler._strategy_usage.values())
        basic_sampler.sample_lineup(3)
        final_usage = sum(basic_sampler._strategy_usage.values())
        assert final_usage == initial_usage + 3

    # --- Advance Tests ---

    def test_advance_increments_counter(self, basic_sampler):
        """advance() should increment reset counter."""
        assert basic_sampler._reset_count == 0
        basic_sampler.advance()
        assert basic_sampler._reset_count == 1
        basic_sampler.advance()
        assert basic_sampler._reset_count == 2

    # --- get_lineup Tests ---

    def test_get_lineup_returns_lineup(self, seeded_sampler):
        """get_lineup should return a valid lineup."""
        lineup = seeded_sampler.get_lineup(3)
        assert len(lineup) == 3
        for name in lineup:
            assert name in seeded_sampler.available_strategies

    def test_get_lineup_advances_counter(self, basic_sampler):
        """get_lineup should advance the counter."""
        assert basic_sampler._reset_count == 0
        basic_sampler.get_lineup(3)
        assert basic_sampler._reset_count == 1

    # --- Logging Tests ---

    def test_should_log_at_interval(self, basic_sampler):
        """should_log should return True at logging intervals."""
        basic_sampler._reset_count = 10000
        assert basic_sampler.should_log(interval=10000)

    def test_should_log_not_at_interval(self, basic_sampler):
        """should_log should return False when not at interval."""
        basic_sampler._reset_count = 10001
        assert not basic_sampler.should_log(interval=10000)

    def test_should_log_not_at_zero(self, basic_sampler):
        """should_log should return False at reset 0."""
        basic_sampler._reset_count = 0
        assert not basic_sampler.should_log(interval=10000)

    def test_get_stats_returns_dict(self, basic_sampler):
        """get_stats should return a dict with expected keys."""
        stats = basic_sampler.get_stats()
        assert "reset_count" in stats
        assert "curriculum_progress" in stats
        assert "target_difficulty" in stats
        assert "strategy_usage" in stats
        assert "available_strategies" in stats

    # --- Seed Tests ---

    def test_set_seed_reproducibility(self):
        """Same seed should produce same lineups."""
        sampler1 = OpponentLineupSampler(
            available_strategies=["random", "killer", "defensive", "cautious"],
            curriculum_total_resets=1000,
            cache_interval=0,  # No caching
        )
        sampler1.set_seed(12345)

        sampler2 = OpponentLineupSampler(
            available_strategies=["random", "killer", "defensive", "cautious"],
            curriculum_total_resets=1000,
            cache_interval=0,
        )
        sampler2.set_seed(12345)

        for _ in range(5):
            lineup1 = sampler1.sample_lineup(3)
            lineup2 = sampler2.sample_lineup(3)
            assert lineup1 == lineup2


class TestCreateDefaultSampler:
    """Tests for the create_default_sampler factory function."""

    def test_returns_sampler(self):
        """Factory should return an OpponentLineupSampler."""
        sampler = create_default_sampler(["random", "killer"])
        assert isinstance(sampler, OpponentLineupSampler)

    def test_uses_provided_strategies(self):
        """Factory should use the provided strategies."""
        strategies = ["random", "killer", "defensive"]
        sampler = create_default_sampler(strategies)
        assert sampler.available_strategies == strategies

    def test_uses_provided_curriculum_resets(self):
        """Factory should use the provided curriculum_resets."""
        sampler = create_default_sampler(["random"], curriculum_resets=500_000)
        assert sampler.curriculum_total_resets == 500_000

    def test_uses_provided_seed(self):
        """Factory should set seed when provided."""
        sampler1 = create_default_sampler(["random", "killer"], seed=42)
        sampler2 = create_default_sampler(["random", "killer"], seed=42)

        lineup1 = sampler1.sample_lineup(3)
        lineup2 = sampler2.sample_lineup(3)
        assert lineup1 == lineup2

    def test_default_settings(self):
        """Factory should set sensible defaults."""
        sampler = create_default_sampler(["random", "killer"])
        assert sampler.force_diversity is True
        assert sampler.diversity_bonus == 0.4
        assert sampler.cache_interval == 1000
        assert sampler.min_difficulty == 0.15
        assert sampler.max_difficulty == 0.75


class TestCurriculumProgression:
    """Integration tests for curriculum progression behavior."""

    def test_early_training_prefers_easier_strategies(self):
        """Early in training, easier strategies should be sampled more often."""
        sampler = OpponentLineupSampler(
            available_strategies=["random", "killer", "defensive"],
            curriculum_total_resets=1000,
            cache_interval=0,
        )
        sampler.set_seed(42)

        # Sample many lineups at early stage
        sampler._reset_count = 0
        early_counts = {"random": 0, "killer": 0, "defensive": 0}
        for _ in range(100):
            for name in sampler.sample_lineup(3):
                early_counts[name] += 1

        # random (difficulty=0.0) should appear frequently early
        assert early_counts["random"] > early_counts["defensive"]

    def test_late_training_allows_harder_strategies(self):
        """Late in training, harder strategies should be available."""
        sampler = OpponentLineupSampler(
            available_strategies=["random", "killer", "hybrid"],
            curriculum_total_resets=100,
            cache_interval=0,
        )
        sampler.set_seed(42)

        # At 0% progress, hybrid not available
        assert "hybrid" not in sampler._get_available_for_stage()

        # At 50% progress, hybrid still not available (min_curriculum_stage=0.3)
        sampler._reset_count = 50
        assert "hybrid" in sampler._get_available_for_stage()

    def test_diversity_maintained_throughout(self):
        """Diversity should be maintained throughout training."""
        sampler = OpponentLineupSampler(
            available_strategies=["random", "killer", "defensive", "cautious"],
            curriculum_total_resets=100,
            force_diversity=True,
            cache_interval=0,
        )
        sampler.set_seed(42)

        # Check diversity at different stages
        for progress in [0, 25, 50, 75, 100]:
            sampler._reset_count = progress
            lineup = sampler.sample_lineup(3)
            assert len(set(lineup)) >= 2, f"Diversity failed at progress={progress}"
