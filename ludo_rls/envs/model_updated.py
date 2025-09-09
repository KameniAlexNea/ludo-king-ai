from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ludo.constants import Colors

from rl_base.envs.model import BaseEnvConfig, BaseRewardConfig, ObservationConfig


@dataclass
class RewardConfig(BaseRewardConfig):
    """Balanced reward configuration for strategic Ludo play (self-play tuned)."""

    # Major events (highest priority)
    # Win reward boosted: with single-seat Option B we need stronger sparse signal
    win: float = 60.0  # Game victory (boosted from 30.0)
    lose: float = -60.0  # Negative terminal when agent loses (symmetric to win)
    finish_token: float = 8.0  # Finishing tokens
    capture: float = 6.0  # Capturing opponents
    got_captured: float = -6.0  # Own token captured

    # Action rewards (moderate)
    extra_turn: float = 2.0  # Rolling a 6 (useful but not dominant)

    # Strategic positioning (light shaping)
    home_progress_bonus: float = 3.0
    blocking_bonus: float = 2.0
    safety_bonus: float = 2.0
    home_approach_bonus: float = 0.5

    # Small continuous signals
    progress_scale: float = 0.05
    diversity_bonus: float = 0.25

    # Penalties / pacing
    illegal_action: float = -8.0
    illegal_masked_scale: float = 0.25  # Scale when mask auto-corrects
    time_penalty: float = -0.01

    # Risk modulation (unused currently)
    use_probabilistic_rewards: bool = False
    risk_weight: float = 0.3
    # Slightly stronger discouragement of long unresolved games
    draw_penalty: float = -2.0  # Applied if game truncates without a winner (was -0.5)


@dataclass
class EnvConfig(BaseEnvConfig):
    agent_color: str = Colors.RED
    max_turns: int = 1000
    reward_cfg: RewardConfig = field(default_factory=RewardConfig)
    obs_cfg: ObservationConfig = field(default_factory=ObservationConfig)
    use_action_mask: bool = True  # When True, invalid chosen actions are auto-corrected without large penalty
    randomize_training_color: bool = True  # Re-sample training seat each reset
