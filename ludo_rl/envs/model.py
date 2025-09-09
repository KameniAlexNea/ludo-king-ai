from __future__ import annotations

from dataclasses import dataclass, field

from ludo.constants import Colors
from rl_base.envs.model import (
    BaseEnvConfig,
    BaseRewardConfig,
    ObservationConfig,
    OpponentCurriculumConfig,
    OpponentsConfig,
)


@dataclass
class RewardConfig(BaseRewardConfig):
    """Balanced reward configuration for strategic Ludo play."""

    pass  # Uses all the defaults from BaseRewardConfig


@dataclass
class EnvConfig(BaseEnvConfig):
    agent_color: str = Colors.RED
    max_turns: int = 1000
    reward_cfg: RewardConfig = field(default_factory=RewardConfig)
    obs_cfg: ObservationConfig = field(default_factory=ObservationConfig)
    opponents: OpponentsConfig = field(default_factory=OpponentsConfig)
    opponent_curriculum: OpponentCurriculumConfig = field(
        default_factory=OpponentCurriculumConfig
    )
    # If True, environment will randomize the agent's seat/color each reset
    randomize_agent_seat: bool = True
