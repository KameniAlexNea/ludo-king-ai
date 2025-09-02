"""RL training package for Ludo.

Provides:
- Gymnasium-compatible environments (single-agent + self-play variants)
- Observation / reward shaping utilities
- Opponent strategy factory wrappers
- Training helpers for Stable-Baselines3 and RLlib

Entry points:
    from ludo_rl.envs import LudoGymEnv
"""

from .envs.ludo_env import LudoGymEnv  # noqa: F401
