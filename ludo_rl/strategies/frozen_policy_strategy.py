from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
from ludo_engine.models import AIDecisionContext, GameConstants, ValidMove
from ludo_engine.strategies.base import Strategy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from ludo_rl.ludo_env.observation import ObservationBuilder


class FrozenPolicyStrategy(Strategy):
    """Strategy that uses a frozen SB3 policy to select actions.

    If no policy is provided, falls back to random valid move.
    """

    def __init__(
        self,
        policy: Optional[MaskableActorCriticPolicy],
        obs_builder: ObservationBuilder,
        deterministic: bool = True,
    ):
        super().__init__(
            "FrozenPolicy",
            "Strategy powered by a frozen PPO policy (falls back to random)",
        )
        self.policy = policy
        self.obs_builder = obs_builder
        self.deterministic = deterministic

    @staticmethod
    def _build_action_mask(valid_moves: list[ValidMove]) -> np.ndarray:
        mask = np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=np.float32)
        for mv in valid_moves:
            tid = mv.token_id
            if isinstance(tid, int) and 0 <= tid < GameConstants.TOKENS_PER_PLAYER:
                mask[tid] = 1.0
        return mask

    def decide(self, game_context: AIDecisionContext) -> int:  # type: ignore[override]
        valid_moves = game_context.valid_moves
        if not valid_moves:
            return 0

        # If no policy was provided, pick a random valid move
        if self.policy is None:
            return random.choice(valid_moves).token_id

        # Build observation from the episode game (via builder) using turn and dice from context
        turn_count = game_context.current_situation.turn_count
        dice_val = game_context.current_situation.dice_value
        obs = self.obs_builder.build(turn_count, dice_val)

        # Compute distribution from policy, derive probs, and apply mask
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist = self.policy.get_distribution(obs_tensor)  # type: ignore[attr-defined]
            try:
                probs = dist.distribution.probs.squeeze(0).cpu().numpy()
            except Exception:
                probs = dist.distribution.logits.softmax(-1).squeeze(0).cpu().numpy()
        mask = self._build_action_mask(valid_moves)
        masked = probs * mask
        if masked.sum() <= 0:
            # degenerate fallback
            return int(np.argmax(mask))
        if self.deterministic:
            return int(np.argmax(masked))
        masked /= masked.sum()
        return int(np.random.choice(len(masked), p=masked))
