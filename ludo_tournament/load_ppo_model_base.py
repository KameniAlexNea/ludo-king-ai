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

import glob
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
from ludo_rl.ludo_env.ludo_env_selfplay import LudoRLEnvSelfPlay
from ludo_rl.ludo_env.ludo_env_hybrid import LudoRLEnvHybrid
from ludo_rl.ludo_env.observation import ObservationBuilder
from ludo_rl.strategies.frozen_policy_strategy import FrozenPolicyStrategy
from ludo_rl.utils.move_utils import MoveUtils

_STEP_PATTERN = re.compile(r"(?:^|_)(\d+)(?:_|$)")


def _create_env_for_vecnormalize(env_cfg: EnvConfig, env_kind: str = "classic"):
    """Create the appropriate environment for VecNormalize loading."""
    if env_kind == "selfplay":
        env = LudoRLEnvSelfPlay(env_cfg)
    elif env_kind == "hybrid":
        env = LudoRLEnvHybrid(env_cfg)
    elif env_kind == "classic":
        env = LudoRLEnv(env_cfg)
    else:
        raise ValueError(f"Invalid env_kind '{env_kind}'. Must be 'classic', 'selfplay', or 'hybrid'")
    return ActionMasker(env, MoveUtils.get_action_mask_for_env)


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
    env_kind: str = "classic",
) -> Tuple[MaskablePPO, str, Optional[VecNormalize]]:
    """Load a MaskablePPO model and its VecNormalize stats, return (model, basename, vec_normalize)."""
    if env_kind not in ["classic", "selfplay", "hybrid"]:
        raise ValueError(f"Invalid env_kind '{env_kind}'. Must be 'classic', 'selfplay', or 'hybrid'")
    
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
    vec_normalize = None
    
    # Find all VecNormalize files
    vecnormalize_files = glob.glob(os.path.join(models_dir, "*vecnormalize*_steps.pkl"))
    
    if not vecnormalize_files:
        raise FileNotFoundError(f"No VecNormalize files found in {models_dir}")
    
    # Check if model name contains steps
    if "_" in model_name and any(char.isdigit() for char in model_name):
        # Extract steps from model name (simple approach)
        parts = model_name.split("_")
        steps = None
        for part in parts:
            if part.isdigit():
                steps = part
                break
        if not steps:
            raise ValueError(f"Could not extract steps from model name '{model_name}'")
        
        # Look for exact match
        target_file = f"ppo_ludo_vecnormalize_{steps}_steps.pkl"
        vecnormalize_path = os.path.join(models_dir, target_file)
        if not os.path.exists(vecnormalize_path):
            available_files = [os.path.basename(f) for f in vecnormalize_files]
            raise FileNotFoundError(f"VecNormalize file '{target_file}' not found for model '{model_name}'. Available: {available_files}")
    else:
        # For non-step models, use highest step file
        try:
            vecnormalize_path = max(vecnormalize_files, key=lambda x: int(os.path.basename(x).split("_")[-2]))
        except (ValueError, IndexError) as e:
            raise RuntimeError(f"Failed to find highest step VecNormalize file: {e}")
    
    # Load VecNormalize
    try:
        env_cfg = EnvConfig()
        dummy_env = DummyVecEnv([lambda: _create_env_for_vecnormalize(env_cfg, env_kind)])
        dummy_env = VecMonitor(dummy_env)
        
        vec_normalize = VecNormalize.load(vecnormalize_path, dummy_env)
        vec_normalize.training = False
        vec_normalize.norm_reward = False
        dummy_env.close()
    except Exception as e:
        raise RuntimeError(f"Failed to load VecNormalize from '{vecnormalize_path}': {e}")

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
    if env_kind not in ["classic", "selfplay", "hybrid"]:
        raise ValueError(f"Invalid env_kind '{env_kind}'. Must be 'classic', 'selfplay', or 'hybrid'")
    # Load model and normalization stats
    model, model_name, vec_normalize = load_ppo_policy(
        models_dir=models_dir, model_preference=model_preference, device=device, env_kind=env_kind
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
