from __future__ import annotations

from typing import List

import numpy as np
from ludo_engine.models import GameConstants, ValidMove


class MoveUtils:
    @staticmethod
    def action_mask(valid_moves: List[ValidMove] | None) -> np.ndarray:
        mask = np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=np.int8)
        if valid_moves:
            valid = {m.token_id for m in valid_moves}
            for i in range(GameConstants.TOKENS_PER_PLAYER):
                if i in valid:
                    mask[i] = 1
        return mask.astype(bool)
