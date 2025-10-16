"""Utilities for selecting and loading PPO models for tournaments.

This module replaces legacy multi-seat/single-seat specific loaders and returns
an engine Strategy implementation (FrozenPolicyStrategy) wrapping a loaded
MaskablePPO policy.

Main entry points:
    - select_best_ppo_model(models_dir, preference, explicit=None)
    - load_ppo_policy(models_dir, preference, device='cpu') -> (model, name, vec_normalize, env_config)
    - load_ppo_strategy(..., game=game_instance) -> FrozenPolicyStrategy

If you need to attach the PPO policy to multiple games, call
`load_ppo_policy` once and then build strategies per game using
`build_frozen_strategy`.
"""

import copy
import glob
import os
import re
from typing import List, Optional, Tuple

import gymnasium as gym
from ludo_engine.core import LudoGame, PlayerColor
from ludo_engine.models import ALL_COLORS
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from ludo_rl.config import EnvConfig
from ludo_rl.ludo_env.ludo_env import LudoRLEnv
from ludo_rl.ludo_env.ludo_env_hybrid import LudoRLEnvHybrid
from ludo_rl.ludo_env.ludo_env_selfplay import LudoRLEnvSelfPlay
from ludo_rl.ludo_env.observation import (
    ContinuousObservationBuilder,
    DiscreteObservationBuilder,
)
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
        raise ValueError(
            f"Invalid env_kind '{env_kind}'. Must be 'classic', 'selfplay', or 'hybrid'"
        )
    return ActionMasker(env, MoveUtils.get_action_mask_for_env)


def _list_model_files(models_dir: str) -> List[str]:
    return [
        f
        for f in os.listdir(models_dir)
        if f.endswith(".zip") and not f.startswith(".")
    ]


def _infer_env_config_from_model(model: MaskablePPO) -> EnvConfig:
    """Infer the EnvConfig used during training from the saved model metadata."""

    env_cfg = EnvConfig()
    obs_space = getattr(model, "observation_space", None)

    include_dice_one_hot = env_cfg.obs.include_dice_one_hot
    discrete = env_cfg.obs.discrete

    try:  # Detect discrete extractor usage directly from the policy if available
        from ludo_rl.features.multidiscrete_extractor import (
            MultiDiscreteFeatureExtractor,
        )
    except Exception:  # pragma: no cover - optional dependency during evaluation
        MultiDiscreteFeatureExtractor = None  # type: ignore[assignment]

    if obs_space is not None:
        if isinstance(obs_space, gym.spaces.Dict):
            # Determine if any component uses MultiDiscrete (discrete observation pipeline)
            discrete = any(
                isinstance(component_space, gym.spaces.MultiDiscrete)
                for component_space in obs_space.spaces.values()
            )

            dice_space = obs_space.spaces.get("dice")
            if isinstance(dice_space, gym.spaces.Box):
                include_dice_one_hot = dice_space.shape[-1] == 6
            elif isinstance(dice_space, gym.spaces.MultiDiscrete):
                include_dice_one_hot = False
        elif isinstance(obs_space, gym.spaces.MultiDiscrete):
            discrete = True
            include_dice_one_hot = False
        elif isinstance(obs_space, gym.spaces.Box):
            discrete = False
            if obs_space.shape:
                include_dice_one_hot = obs_space.shape[-1] == 6

    if MultiDiscreteFeatureExtractor is not None and isinstance(
        model.policy.features_extractor, MultiDiscreteFeatureExtractor
    ):
        discrete = True

    env_cfg.obs.include_dice_one_hot = include_dice_one_hot
    env_cfg.obs.discrete = discrete
    return env_cfg


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
) -> Tuple[MaskablePPO, str, Optional[VecNormalize], EnvConfig]:
    """Load MaskablePPO policy, VecNormalize stats (if any), and inferred EnvConfig."""
    # Normalize aliases
    if env_kind == "single-seat":
        env_kind = "classic"
    if env_kind not in ["classic", "selfplay", "hybrid"]:
        raise ValueError(
            f"Invalid env_kind '{env_kind}'. Must be 'classic', 'selfplay', or 'hybrid'"
        )

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

    env_cfg = _infer_env_config_from_model(model)

    # Try to load VecNormalize stats if available; continue without if not found
    vec_normalize: Optional[VecNormalize] = None

    # Find all VecNormalize files (support multiple naming patterns)
    vecnormalize_files = glob.glob(
        os.path.join(os.path.dirname(model_path), "*vecnormalize*.pkl")
    )

    vecnormalize_path: Optional[str] = None
    if vecnormalize_files:
        # Prefer files with explicit step numbers matching model if any
        chosen: Optional[str] = None
        # Try to extract step int from model name
        m = _STEP_PATTERN.search(model_name)
        target_step = int(m.group(1)) if m else None
        if target_step is not None:
            for f in vecnormalize_files:
                nums = [int(x) for x in re.findall(r"(\d+)", os.path.basename(f))]
                if target_step in nums:
                    chosen = f
                    break
        if chosen is None:
            # fallback to latest by max numeric token in filename
            def last_number(path: str) -> int:
                nums = [int(x) for x in re.findall(r"(\d+)", os.path.basename(path))]
                return nums[-1] if nums else -1

            chosen = max(vecnormalize_files, key=last_number)
        vecnormalize_path = chosen

    if vecnormalize_path is not None and os.path.exists(vecnormalize_path):
        try:
            dummy_env = DummyVecEnv(
                [lambda: _create_env_for_vecnormalize(env_cfg, env_kind)]
            )
            dummy_env = VecMonitor(dummy_env)
            vec_normalize = VecNormalize.load(vecnormalize_path, dummy_env)
            vec_normalize.training = False
            vec_normalize.norm_reward = False
            dummy_env.close()
        except Exception:
            # If loading VecNormalize fails, proceed without normalization
            vec_normalize = None

    return model, model_name, vec_normalize, env_cfg


def build_frozen_strategy(
    model: MaskablePPO,
    game: LudoGame,
    agent_color: PlayerColor = PlayerColor.RED,
    env_cfg: Optional[EnvConfig] = None,
    player_name: str = "ppo",
    deterministic: bool = False,
    vec_normalize: Optional[VecNormalize] = None,
):
    """Create a FrozenPolicyStrategy bound to a specific game instance.

    You can reuse the same loaded model across many games by calling this per game.
    """
    cfg = env_cfg or EnvConfig()

    discrete_obs = bool(cfg.obs and getattr(cfg.obs, "discrete", False))
    if discrete_obs:
        vec_normalize = None  # Never normalize categorical features
        obs_builder = DiscreteObservationBuilder(cfg, game, agent_color)
    else:
        obs_builder = ContinuousObservationBuilder(cfg, game, agent_color)

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
    deterministic: bool = False,
    device: str = "cpu",
    max_turns=500,
):
    """Load PPO strategy with proper VecNormalize handling."""
    if env_kind == "single-seat":
        env_kind = "classic"
    if env_kind not in ["classic", "selfplay", "hybrid"]:
        raise ValueError(
            f"Invalid env_kind '{env_kind}'. Must be 'classic', 'selfplay', or 'hybrid'"
        )
    # Load model and normalization stats
    model, model_name, vec_normalize, inferred_env_cfg = load_ppo_policy(
        models_dir=models_dir,
        model_preference=model_preference,
        device=device,
        env_kind=env_kind,
    )

    if game is None:
        game = LudoGame(ALL_COLORS)

    env_cfg = copy.deepcopy(inferred_env_cfg)
    env_cfg.max_turns = max_turns
    env_cfg.seed = 42

    strategy = build_frozen_strategy(
        model=model,
        game=game,
        agent_color=agent_color,
        env_cfg=env_cfg,
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
