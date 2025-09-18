import glob
import os
import pickle

import numpy as np
import torch
from ludo_engine.models import AIDecisionContext, GameConstants, ValidMove
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

from rl_base.envs.model import BaseEnvConfig


class BasePPOStrategy:
    """Base PPO strategy wrapper using shared ObservationBuilder for feature logic.

    This avoids duplicating normalization / feature engineering. We project the provided
    game context into a lightweight internal game snapshot (dummy env's game) and then
    delegate vector construction to `ObservationBuilder._build_observation`.

    Only action masking + (optional) deterministic selection remain here.
    """

    def __init__(
        self,
        model_path: str,
        model_name: str,
        env_config: BaseEnvConfig | None = None,
        deterministic: bool = True,
        maskable: bool = False,
    ):
        self.model_name = model_name
        self.model = (MaskablePPO if maskable else PPO).load(model_path, device="cpu")
        # Ensure policy on CPU for deterministic test environment
        try:
            self.model.policy.to("cpu")
        except Exception:
            pass
        self.env_cfg = env_config or BaseEnvConfig()

        # Subclasses should implement this to create the appropriate environment
        self.dummy_env = self._create_dummy_env()

        # Build once to establish obs dimension
        obs, _ = self.dummy_env.reset(seed=self.env_cfg.seed)
        self.obs_dim = obs.shape[0]
        self.obs_builder = self.dummy_env.obs_builder
        self.deterministic = deterministic
        self.description = f"PPO Model: {model_name} (obs_dim={self.obs_dim})"

        # Attempt to load VecNormalize stats (obs_rms) from sibling files
        self.obs_rms = None
        try:
            model_dir = os.path.dirname(model_path)
            stats_candidates = sorted(
                glob.glob(os.path.join(model_dir, "*vecnormalize*.pkl"))
            )
            if stats_candidates:
                with open(stats_candidates[-1], "rb") as f:  # latest
                    vn = pickle.load(f)
                    # SB3 VecNormalize stores running mean/var in obs_rms
                    self.obs_rms = getattr(vn, "obs_rms", None)
        except Exception:
            # Silent fallback – strategy still works without normalization
            self.obs_rms = None

    def _create_dummy_env(self):
        """Create the dummy environment. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _create_dummy_env")

    # --- Helpers -----------------------------------------------------------------
    def _build_action_mask(self, valid_moves: list[ValidMove]) -> np.ndarray:
        mask = np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=np.float32)
        for mv in valid_moves:
            tid = mv.token_id
            if isinstance(tid, int) and 0 <= tid < GameConstants.TOKENS_PER_PLAYER:
                mask[tid] = 1.0
        return mask

    def _inject_context_into_dummy_game(
        self, ctx: AIDecisionContext
    ) -> tuple[int, int]:
        """Mutate dummy game to reflect context player token positions.

        Returns (turn_count, dice_value) extracted from context.
        """
        return ctx.current_situation.turn_count, ctx.current_situation.dice_value

    # --- Public API --------------------------------------------------------------
    def decide(self, game_context: AIDecisionContext) -> int:
        valid_moves = game_context.valid_moves
        if not valid_moves:
            return 0  # nothing to do
        turn_count, dice_val = (
            game_context.current_situation.turn_count,
            game_context.current_situation.dice_value,
        )
        # Build observation via canonical builder
        obs = self.obs_builder._build_observation(turn_count, dice_val)
        if obs.shape[0] != self.obs_dim:
            # Defensive: reshape/pad if feature flags changed externally
            if obs.shape[0] < self.obs_dim:
                obs = np.pad(obs, (0, self.obs_dim - obs.shape[0]))
            else:
                obs = obs[: self.obs_dim]
        # Apply observation normalization if stats loaded
        if self.obs_rms is not None:
            try:
                obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
                obs = np.clip(obs, -10.0, 10.0)
            except Exception:
                pass
        mask = self._build_action_mask(valid_moves)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_tensor)
            try:
                probs = dist.distribution.probs.squeeze(0).cpu().numpy()
            except Exception:  # backwards compatibility (logits path)
                probs = dist.distribution.logits.softmax(-1).squeeze(0).cpu().numpy()
        masked = probs * mask
        if masked.sum() <= 0:  # degenerate (no prob mass on valid actions)
            return int(np.argmax(mask))
        if self.deterministic:
            return int(np.argmax(masked))
        masked /= masked.sum()
        return int(np.random.choice(len(masked), p=masked))
