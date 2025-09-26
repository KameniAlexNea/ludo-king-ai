"""Utilities for selecting and loading PPO models for tournaments.

This module replaces legacy multi-seat/single-seat specific loaders and returns
an engine Strategy implementation (FrozenPolicyStrategy) wrapping a loaded
MaskablePPO policy.

Main entry points:
  - select_best_ppo_model(models_dir, preference, explicit=None)
  - load_ppo_policy(models_dir, preference, device='cpu') -> (model, name, vec_normalize)
  - load_ppo_strategy(..., game=game_instance) -> FrozenPolicyStrategy

If you need to attach the PPO policy to multiple games, call
`load_ppo_policy` once and then build strategies per game using
`build_frozen_strategy`.
"""

import os
import re
from typing import List, Optional, Tuple

from ludo_engine.core import LudoGame, PlayerColor
from ludo_engine.models import ALL_COLORS
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from ludo_rl.config import EnvConfig
from ludo_rl.ludo_env.ludo_env import LudoRLEnv
from ludo_rl.ludo_env.observation import ObservationBuilder
from ludo_rl.strategies.frozen_policy_strategy import FrozenPolicyStrategy
from ludo_rl.utils.move_utils import MoveUtils

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
    if os.path.isfile(models_dir):
        # If a file was provided, assume it's the model path and return its basename
        base = os.path.basename(models_dir)
        if base.endswith(".zip"):
            return base[:-4]
        return base
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
) -> Tuple[MaskablePPO, str, Optional[VecNormalize]]:
    """Load a MaskablePPO model and its VecNormalize stats, return (model, basename, vec_normalize)."""
    model_name = select_best_ppo_model(models_dir, model_preference, explicit)
    if not os.path.isdir(models_dir):
        models_dir = os.path.dirname(models_dir)

    model_path = os.path.join(models_dir, f"{model_name}.zip")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found")

    # Load the model
    try:
        model = MaskablePPO.load(model_path, device=device)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to load PPO model '{model_path}': {e}") from e

    # Try to load VecNormalize stats
    vecnormalize_path = os.path.join(models_dir, f"{model_name}_vecnormalize.pkl")
    vec_normalize = None

    if os.path.exists(vecnormalize_path):
        try:
            # Create a dummy env to load VecNormalize (it needs the env structure)
            env_cfg = EnvConfig()
            dummy_env = DummyVecEnv(
                [
                    lambda: ActionMasker(
                        LudoRLEnv(env_cfg), MoveUtils.get_action_mask_for_env
                    )
                ]
            )
            dummy_env = VecMonitor(dummy_env)

            # Load the VecNormalize with the dummy env
            vec_normalize = VecNormalize.load(vecnormalize_path, dummy_env)
            vec_normalize.training = False  # Set to evaluation mode
            vec_normalize.norm_reward = False  # Match training setup

            # Close the dummy env since we only needed it for loading
            dummy_env.close()

        except Exception as e:
            print(
                f"Warning: Could not load VecNormalize stats from {vecnormalize_path}: {e}"
            )
            vec_normalize = None
    else:
        print(f"Warning: VecNormalize file not found at {vecnormalize_path}")

    return model, model_name, vec_normalize


def build_frozen_strategy(
    model: MaskablePPO,
    game: LudoGame,
    agent_color: PlayerColor = PlayerColor.RED,
    env_cfg: Optional[EnvConfig] = None,
    player_name: str = "ppo",
    deterministic: bool = True,
    vec_normalize: Optional[VecNormalize] = None,
):
    """Create a FrozenPolicyStrategy bound to a specific game instance.

    You can reuse the same loaded model across many games by calling this per game.
    """
    cfg = env_cfg or EnvConfig()
    obs_builder = ObservationBuilder(cfg, game, agent_color)

    strat = FrozenPolicyStrategy(
        model.policy,
        obs_builder,
        deterministic=deterministic,
        obs_normalizer=vec_normalize,
    )
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
    max_turns=500,
):
    """Load PPO strategy with proper VecNormalize handling."""
    # Load model and normalization stats
    model, model_name, vec_normalize = load_ppo_policy(
        models_dir=models_dir, model_preference=model_preference, device=device
    )

    if game is None:
        game = LudoGame(ALL_COLORS)

    strategy = build_frozen_strategy(
        model=model,
        game=game,
        agent_color=agent_color,
        env_cfg=EnvConfig(max_turns=max_turns, seed=42),
        player_name=player_name,
        deterministic=deterministic,
        vec_normalize=vec_normalize,
    )
    return strategy


__all__ = [
    "select_best_ppo_model",
    "load_ppo_policy",
    "build_frozen_strategy",
    "load_ppo_strategy",
]
