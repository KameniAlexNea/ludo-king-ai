"""Utilities for selecting and loading PPO models for tournaments.

This module replaces legacy multi-seat/single-seat specific loaders and returns
an engine Strategy implementation (FrozenPolicyStrategy) wrapping a loaded
MaskablePPO policy.

Main entry points:
  - select_best_ppo_model(models_dir, preference, explicit=None)
  - load_ppo_policy(models_dir, preference, device='cpu') -> (model, name)
  - load_ppo_strategy(..., game=game_instance) -> FrozenPolicyStrategy

If you need to attach the PPO policy to multiple games, call
`load_ppo_policy` once and then build strategies per game using
`build_frozen_strategy`.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

from ludo_engine.core import LudoGame, PlayerColor
from ludo_engine.models import ALL_COLORS
from sb3_contrib import MaskablePPO

from ludo_rl.config import EnvConfig
from ludo_rl.ludo_env.observation import ObservationBuilder
from ludo_rl.strategies.frozen_policy_strategy import FrozenPolicyStrategy

_STEP_PATTERN = re.compile(r"(?:^|_)(\d+)(?:_|$)")


def _list_model_files(models_dir: str) -> List[str]:
    return [
        f
        for f in os.listdir(models_dir)
        if f.endswith(".zip") and not f.startswith(".")
    ]


def select_best_ppo_model(
    models_dir: str, model_preference: str = "final", explicit: Optional[str] = None
) -> str:
    """Select a PPO model basename (without .zip).

    Order logic:
      - If `explicit` provided and exists, return it.
      - Else follow preference chain (best|final|steps) with fallbacks.
    """
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory '{models_dir}' not found")

    model_files = _list_model_files(models_dir)
    if not model_files:
        raise FileNotFoundError(f"No .zip models in '{models_dir}'")

    bases = [f[:-4] for f in model_files]

    if explicit:
        if explicit in bases:
            return explicit
        raise FileNotFoundError(
            f"Explicit model '{explicit}' not found in: {', '.join(bases)}"
        )

    pref = model_preference.lower()
    if pref == "best":
        order = ["best", "final", "steps"]
    elif pref == "final":
        order = ["final", "best", "steps"]
    elif pref == "steps":
        order = ["steps", "best", "final"]
    else:
        raise ValueError("model_preference must be one of: best|final|steps")

    def find_tag(tag: str) -> Optional[str]:
        tag_lower = tag.lower()
        for b in bases:
            if tag_lower in b.lower():
                return b
        return None

    for p in order:
        if p in ("best", "final"):
            cand = find_tag(p)
            if cand:
                return cand
        elif p == "steps":
            step_pairs: List[Tuple[int, str]] = []
            for b in bases:
                m = _STEP_PATTERN.search(b)
                if not m:
                    continue
                try:
                    step = int(m.group(1))
                except Exception:
                    continue
                step_pairs.append((step, b))
            if step_pairs:
                step_pairs.sort(key=lambda x: x[0], reverse=True)
                return step_pairs[0][1]

    # fallback: first in list
    return bases[0]


def load_ppo_policy(
    models_dir: str,
    model_preference: str = "final",
    explicit: Optional[str] = None,
    device: str = "cpu",
):
    """Load a MaskablePPO model and return (model, basename)."""
    model_name = select_best_ppo_model(models_dir, model_preference, explicit)
    model_path = os.path.join(models_dir, f"{model_name}.zip")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found")
    try:
        model = MaskablePPO.load(model_path, device=device)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to load PPO model '{model_path}': {e}") from e
    return model, model_name


def build_frozen_strategy(
    model: MaskablePPO,
    game: LudoGame,
    agent_color: PlayerColor = PlayerColor.RED,
    env_cfg: Optional[EnvConfig] = None,
    player_name: str = "ppo",
    deterministic: bool = True,
):
    """Create a FrozenPolicyStrategy bound to a specific game instance.

    You can reuse the same loaded model across many games by calling this per game.
    """
    cfg = env_cfg or EnvConfig()
    obs_builder = ObservationBuilder(cfg, game, agent_color)
    # The StrategyFactory expects a strategy instance with .decide()
    strat = FrozenPolicyStrategy(model.policy, obs_builder, deterministic=deterministic)
    # Adjust its public name if desired
    strat.name = player_name
    return strat


def load_ppo_strategy(
    env_kind: str,
    models_dir: str,
    player_name: str = "ppo",
    agent_color: PlayerColor = PlayerColor.RED,
    model_preference: str = "final",
    game: Optional[LudoGame] = None,
    deterministic: bool = True,
    device: str = "cpu",
):
    """Backward-compatible loader that returns a FrozenPolicyStrategy.

    Parameters mirror the legacy interface plus:
      - game: required to build the observation context. If None, a temporary
        LudoGame is constructed (NOT recommended for actual tournament play).
    """
    model, model_name = load_ppo_policy(
        models_dir=models_dir, model_preference=model_preference, device=device
    )
    if game is None:
        # logger.warning(
        #     "No game instance provided to load_ppo_strategy; creating a temporary LudoGame.\n"
        #     "Attach a proper strategy built via build_frozen_strategy(model, actual_game, ...) for real matches."
        # )
        game = LudoGame(ALL_COLORS)
    strategy = build_frozen_strategy(
        model=model,
        game=game,
        agent_color=agent_color,
        env_cfg=EnvConfig(max_turns=500, seed=42),
        player_name=player_name,
        deterministic=deterministic,
    )
    return strategy


__all__ = [
    "select_best_ppo_model",
    "load_ppo_policy",
    "build_frozen_strategy",
    "load_ppo_strategy",
]
