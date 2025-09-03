import glob
import os
import pickle

import numpy as np
import torch
from stable_baselines3 import PPO

from ludo.constants import Colors, GameConstants

from .envs.builders.observation_builder import ObservationBuilder
from .envs.ludo_env import EnvConfig, LudoGymEnv


class PPOStrategy:
    """Single-seat PPO strategy wrapper using shared ObservationBuilder.

    Mirrors classic implementation: build dummy env once, inject token positions
    from provided context, construct observation via builder, apply optional
    VecNormalize stats, and choose action (deterministic or stochastic).
    """

    def __init__(
        self,
        model_path: str,
        model_name: str,
        env_config: EnvConfig | None = None,
        deterministic: bool = True,
    ):
        self.model_name = model_name
        self.model = PPO.load(model_path)
        self.env_cfg = env_config or EnvConfig()
        if self.env_cfg.agent_color != Colors.RED:
            raise ValueError(
                f"PPOStrategy expects agent_color RED (training seat); got {self.env_cfg.agent_color}"
            )
        self.dummy_env = LudoGymEnv(self.env_cfg)
        obs, _ = self.dummy_env.reset(seed=self.env_cfg.seed)
        self.obs_dim = obs.shape[0]
        self.obs_builder: ObservationBuilder = self.dummy_env.obs_builder
        self.deterministic = deterministic
        self.description = f"PPO Model: {model_name} (obs_dim={self.obs_dim})"
        # VecNormalize stats attempt
        self.obs_rms = None
        try:
            model_dir = os.path.dirname(model_path)
            stats_candidates = sorted(
                glob.glob(os.path.join(model_dir, "*vecnormalize*.pkl"))
            )
            if stats_candidates:
                with open(stats_candidates[-1], "rb") as f:
                    vn = pickle.load(f)
                    self.obs_rms = getattr(vn, "obs_rms", None)
        except Exception:
            self.obs_rms = None

    # Helpers
    def _build_action_mask(self, valid_moves: list[dict]) -> np.ndarray:
        mask = np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=np.float32)
        for mv in valid_moves:
            tid = mv.get("token_id")
            if isinstance(tid, int) and 0 <= tid < GameConstants.TOKENS_PER_PLAYER:
                mask[tid] = 1.0
        return mask

    def _inject_context(self, ctx: dict) -> tuple[int, int]:
        players_ctx = ctx.get("players") or []
        by_color = {}
        for p in players_ctx:
            col = p.get("color")
            col_val = col.value if hasattr(col, "value") else col
            by_color[col_val] = p
        for game_player in self.dummy_env.game.players:
            p_ctx = by_color.get(game_player.color.value)
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

    # Public
    def decide(self, game_context: dict) -> int:
        valid_moves = game_context.get("valid_moves", [])
        if not valid_moves:
            return 0
        turn_count, dice_val = self._inject_context(game_context)
        obs = self.obs_builder._build_observation(turn_count, dice_val)
        if obs.shape[0] != self.obs_dim:
            if obs.shape[0] < self.obs_dim:
                obs = np.pad(obs, (0, self.obs_dim - obs.shape[0]))
            else:
                obs = obs[: self.obs_dim]
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
            except Exception:
                probs = dist.distribution.logits.softmax(-1).squeeze(0).cpu().numpy()
        masked = probs * mask
        if masked.sum() <= 0:
            return int(np.argmax(mask))
        if self.deterministic:
            return int(np.argmax(masked))
        masked /= masked.sum()
        return int(np.random.choice(len(masked), p=masked))
