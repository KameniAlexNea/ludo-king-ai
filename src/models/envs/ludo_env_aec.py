"""Compatibility wrapper that re-exports the multi-agent Ludo environment."""

from __future__ import annotations

import os

__path__ = [os.path.join(os.path.dirname(__file__), "ludo_env_aec")]

from .ludo_env_aec.opponent_pool import OpponentPoolManager
from .ludo_env_aec.raw_env import env, raw_env
from .ludo_env_aec.turn_based_env import TurnBasedSelfPlayEnv

__all__ = [
    "env",
    "raw_env",
    "OpponentPoolManager",
    "TurnBasedSelfPlayEnv",
]
