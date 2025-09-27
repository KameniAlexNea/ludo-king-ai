from typing import Optional

import numpy as np
import torch
from ludo_engine.models import AIDecisionContext, GameConstants, ValidMove
from ludo_engine.strategies.base import Strategy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.vec_env import VecNormalize

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
        obs_normalizer: Optional[VecNormalize] = None,
    ):
        super().__init__(
            "FrozenPolicy",
            "Strategy powered by a frozen PPO policy (falls back to random)",
        )
        self.policy = policy
        self.obs_builder = obs_builder
        self.deterministic = deterministic
        self.obs_normalizer = obs_normalizer

    @staticmethod
    def _build_action_mask(valid_moves: list[ValidMove]) -> np.ndarray:
        mask = np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=np.float32)
        for mv in valid_moves:
            tid = mv.token_id
            mask[tid] = 1
        return mask

    def decide(self, game_context: AIDecisionContext) -> int:  # type: ignore[override]
        valid_moves = game_context.valid_moves
        if not valid_moves:
            return 0

        # If no policy was provided, pick a random valid move
        if self.policy is None:
            raise ValueError("Policy is None, cannot decide action.")

        # Build observation from the episode game (via builder) using turn and dice from context
        turn_count = game_context.current_situation.turn_count
        dice_val = game_context.current_situation.dice_value
        obs = self.obs_builder.build(turn_count, dice_val)

        if self.obs_normalizer is not None:
            obs = self.obs_normalizer.normalize_obs(obs)

        # Compute distribution from policy, derive probs, and apply mask
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask = self._build_action_mask(valid_moves)
        with torch.no_grad():
            dist = self.policy.get_distribution(obs_tensor, mask)  # type: ignore[attr-defined]
            token_id = dist.get_actions(deterministic=self.deterministic).item()
        valid_token_ids = [mv.token_id for mv in valid_moves]
        if token_id not in valid_token_ids:
            # Log warning and fall back to random valid move
            token_id = np.random.choice(valid_token_ids)
        return token_id
