from __future__ import annotations

from dataclasses import dataclass, field

from ludo.constants import Colors
from rl_base.envs.model import BaseEnvConfig, BaseRewardConfig, ObservationConfig


@dataclass
class RewardConfig(BaseRewardConfig):
    """Balanced reward configuration for strategic Ludo play (self-play tuned)."""

    # Only new attributes not in BaseRewardConfig
    illegal_masked_scale: float = 0.25  # Scale when mask auto-corrects
    draw_penalty: float = -2.0  # Applied if game truncates without a winner (was -0.5)


@dataclass
class EnvConfig(BaseEnvConfig):
    agent_color: str = Colors.RED
    max_turns: int = 1000
    reward_cfg: RewardConfig = field(default_factory=RewardConfig)
    obs_cfg: ObservationConfig = field(default_factory=ObservationConfig)
    use_action_mask: bool = True  # When True, invalid chosen actions are auto-corrected without large penalty
    randomize_training_color: bool = True  # Re-sample training seat each reset
