import glob
import os
import pickle

import numpy as np
import torch
from stable_baselines3 import PPO

from ludo.constants import GameConstants

from ..envs.model import BaseEnvConfig


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
    ):
        self.model_name = model_name
        self.model = PPO.load(model_path, device="cpu")
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
    def _build_action_mask(self, valid_moves: list[dict]) -> np.ndarray:
        mask = np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=np.float32)
        for mv in valid_moves:
            tid = mv.get("token_id")
            if isinstance(tid, int) and 0 <= tid < GameConstants.TOKENS_PER_PLAYER:
                mask[tid] = 1.0
        return mask

    def _inject_context_into_dummy_game(self, ctx: dict) -> tuple[int, int]:
        """Mutate dummy game to reflect context player token positions.

        Returns (turn_count, dice_value) extracted from context.
        """
        players_ctx = ctx.get("players") or []
        # Map colors for quick lookup; Colors enums carry .value
        by_color = {}
        for p in players_ctx:
            col = p.get("color")
            col_val = col.value if hasattr(col, "value") else col
            by_color[col_val] = p
        for game_player in self.dummy_env.game.players:
            gcol_val = game_player.color.value
            p_ctx = by_color.get(gcol_val)
            if not p_ctx:
                continue
            toks_ctx = p_ctx.get("tokens", [])
            for i, tok in enumerate(game_player.tokens):
                if i < len(toks_ctx):
                    tok.position = toks_ctx[i].get("position", -1)
                else:
                    tok.position = -1
        game_info = ctx.get("game_info", {})
        dice_val = ctx.get("dice_value") or game_info.get("dice_value") or 0
        turn_count = game_info.get("turn_count", ctx.get("turn_count", 0))
        return int(turn_count), int(dice_val)

    # --- Public API --------------------------------------------------------------
    def decide(self, game_context: dict) -> int:  # noqa: C901 (kept small already)
        valid_moves = game_context.get("valid_moves", [])
        if not valid_moves:
            return 0  # nothing to do
        turn_count, dice_val = self._inject_context_into_dummy_game(game_context)
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
