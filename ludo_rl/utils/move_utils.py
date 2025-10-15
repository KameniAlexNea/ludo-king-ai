from typing import List

import numpy as np
from ludo_engine.models import GameConstants, ValidMove


class MoveUtils:
    @staticmethod
    def action_masks(valid_moves: List[ValidMove]) -> np.ndarray:
        """Create an action mask from a list of valid moves."""
        if valid_moves is None:
            raise ValueError("valid_moves is None, cannot create action mask.")
        mask = np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=np.int32)
        if valid_moves:
            valid = {m.token_id for m in valid_moves}
            for i in range(GameConstants.TOKENS_PER_PLAYER):
                if i in valid:
                    mask[i] = 1
        return mask.astype(bool)

    @staticmethod
    def get_action_mask_for_env(env) -> np.ndarray:
        """Get action mask for a given environment instance, handling exceptions gracefully."""
        return MoveUtils.action_masks(getattr(env, "pending_valid_moves", None))
