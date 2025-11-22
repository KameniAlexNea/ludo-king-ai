from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Dict, Optional

import numpy as np
from sb3_contrib import MaskablePPO

from ..ludo_king.config import config
from .base import BaseStrategy, BaseStrategyConfig
from .types import MoveOption, StrategyContext

if TYPE_CHECKING:
    from ..ludo_king.game import Game


@dataclass(slots=True)
class RLStrategyConfig(BaseStrategyConfig):
    model: MaskablePPO
    deterministic: bool = True

    def sample(self, rng=None) -> Dict[str, object]:  # noqa: D401 - interface impl
        return {"model": self.model, "deterministic": self.deterministic}


class RLStrategy(BaseStrategy):
    """Delegate move selection to a pre-trained MaskablePPO policy."""

    name = "rl"
    config: ClassVar[Optional[BaseStrategyConfig]] = None

    def __init__(self, model: MaskablePPO, deterministic: bool = True) -> None:
        self.model = model
        self.deterministic = deterministic
        self.model.policy.set_training_mode(False)
        self._hist_T = int(config.HISTORY_LENGTH)
        # History buffers (agent-relative) to mirror env/simulator
        self._pos_hist: Optional[np.ndarray] = None  # (T,16) int64
        self._dice_hist: Optional[np.ndarray] = None  # (T,) int64
        self._mask_hist: Optional[np.ndarray] = None  # (T,16) bool
        self._player_hist: Optional[np.ndarray] = None  # (T,) int64
        self._token_colors: Optional[np.ndarray] = None  # (16,) int64
        self._token_exists_mask: Optional[np.ndarray] = None  # (16,) bool
        self._hist_len: int = 0
        self._hist_ptr: int = 0
        self._agent_index: Optional[int] = None

    @classmethod
    def configure(cls, model: MaskablePPO, *, deterministic: bool = True) -> None:
        return cls(model=model, deterministic=deterministic)

    @classmethod
    def configure_from_path(
        cls, model_path: str, *, device: str = "cpu", deterministic: bool = True
    ) -> None:
        model = MaskablePPO.load(model_path, device=device)
        model.policy.set_training_mode(False)
        return cls.configure(model=model, deterministic=deterministic)

    # --- History maintenance API ---
    def _ensure_buffers(self, game: "Game") -> None:
        if self._pos_hist is not None:
            return
        # Find this strategy's player index and agent color
        agent_index = None
        for i, pl in enumerate(game.players):
            if getattr(pl, "strategy", None) is self:
                agent_index = i
                break
        if agent_index is None:
            agent_index = 0
        self._agent_index = agent_index
        agent_color = int(game.players[agent_index].color)
        # Allocate buffers
        self._pos_hist = np.zeros((self._hist_T, 16), dtype=np.int64)
        self._dice_hist = np.zeros((self._hist_T,), dtype=np.int64)
        self._mask_hist = np.zeros((self._hist_T, 16), dtype=np.bool_)
        self._player_hist = np.zeros((self._hist_T,), dtype=np.int64)
        # Static per-agent arrays
        self._token_colors = game.board.token_colors(agent_color)
        self._token_exists_mask = game.board.token_exists_mask(agent_color)
        self._hist_len = 0
        self._hist_ptr = 0

    def update_history(self, game: "Game", mover_index: int, dice_roll: int) -> None:  # type: ignore[override]
        self._ensure_buffers(game)
        assert self._pos_hist is not None and self._dice_hist is not None
        assert self._mask_hist is not None and self._player_hist is not None
        # Snapshot agent-relative positions after the move
        agent_color = int(game.players[self._agent_index or 0].color)
        frame_pos = game.board.all_token_positions(agent_color)
        i = self._hist_ptr
        self._pos_hist[i, :] = frame_pos
        self._dice_hist[i] = int(dice_roll)
        self._mask_hist[i, :] = self._token_exists_mask  # type: ignore[index]
        self._player_hist[i] = int(mover_index)
        self._hist_ptr = (self._hist_ptr + 1) % self._hist_T
        self._hist_len = min(self._hist_len + 1, self._hist_T)

    def _build_observation(self, current_dice: int) -> Dict[str, np.ndarray]:
        # If no history yet, return zeros with correct shapes
        T = self._hist_T
        if self._pos_hist is None or self._hist_len == 0:
            return {
                "positions": np.zeros((T, 16), dtype=np.int64),
                "dice_history": np.zeros((T,), dtype=np.int64),
                "token_mask": np.zeros((T, 16), dtype=np.bool_),
                "player_history": np.zeros((T,), dtype=np.int64),
                "token_colors": np.zeros((16,), dtype=np.int64),
                "current_dice": np.asarray([int(current_dice)], dtype=np.int64),
            }
        k = self._hist_len
        out_pos = np.zeros((T, 16), dtype=np.int64)
        out_dice = np.zeros((T,), dtype=np.int64)
        out_mask = np.zeros((T, 16), dtype=np.bool_)
        out_player = np.zeros((T,), dtype=np.int64)
        start = (self._hist_ptr - k) % T
        if start + k <= T:
            out_pos[T - k : T, :] = self._pos_hist[start : start + k, :]
            out_dice[T - k : T] = self._dice_hist[start : start + k]
            out_mask[T - k : T, :] = self._mask_hist[start : start + k, :]
            out_player[T - k : T] = self._player_hist[start : start + k]
        else:
            first = T - start
            out_pos[T - k : T - k + first, :] = self._pos_hist[start:T, :]
            out_pos[T - k + first : T, :] = self._pos_hist[0 : k - first, :]
            out_dice[T - k : T - k + first] = self._dice_hist[start:T]
            out_dice[T - k + first : T] = self._dice_hist[0 : k - first]
            out_mask[T - k : T - k + first, :] = self._mask_hist[start:T, :]
            out_mask[T - k + first : T, :] = self._mask_hist[0 : k - first, :]
            out_player[T - k : T - k + first] = self._player_hist[start:T]
            out_player[T - k + first : T] = self._player_hist[0 : k - first]
        return {
            "positions": out_pos,
            "dice_history": out_dice,
            "token_mask": out_mask,
            "player_history": out_player,
            "token_colors": (
                self._token_colors
                if self._token_colors is not None
                else np.zeros((16,), dtype=np.int64)
            ),
            "current_dice": np.asarray([int(current_dice)], dtype=np.int64),
        }

    def select_move(self, ctx: StrategyContext) -> Optional[MoveOption]:  # type: ignore[override]
        if not np.any(ctx.action_mask):
            return None

        # Prefer history-based observation if available
        observation = self._build_observation(ctx.dice_roll)

        action, _ = self.model.predict(
            observation,
            action_masks=ctx.action_mask.astype(bool),
            deterministic=self.deterministic,
        )

        action_id = int(np.asarray(action).item())
        if action_id < 0 or action_id >= len(ctx.action_mask):
            return next(ctx.iter_legal(), None)

        if not ctx.action_mask[action_id]:
            return next(ctx.iter_legal(), None)

        for move in ctx.moves:
            if move.piece_id == action_id:
                return move

        return next(ctx.iter_legal(), None)
