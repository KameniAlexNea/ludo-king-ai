from __future__ import annotations

from typing import Optional

from ludo_engine.models import MoveResult

from ludo_rl.config import EnvConfig


class RewardCalculator:
    """Encapsulates reward computation for a single environment step.

    Inputs:
    - res: MoveResult from executing the agent's move
    - illegal: whether the chosen action was illegal and auto-corrected
    - cfg: EnvConfig to access RewardConfig weights
    - game_over: whether the overall game is over (and the agent did not win)

    Output:
    - scalar reward (float)
    """

    def compute(
        self, res: MoveResult, illegal: bool, cfg: EnvConfig, game_over: bool
    ) -> float:
        r = 0.0

        # Event rewards
        if getattr(res, "captured_tokens", None):
            r += cfg.reward.capture * len(res.captured_tokens)
        if getattr(res, "finished_token", False):
            r += cfg.reward.finish_token

        # Constraint penalties
        if illegal:
            r += cfg.reward.illegal_action

        # Step penalty
        r += cfg.reward.time_penalty

        # Terminal outcomes
        if getattr(res, "game_won", False):
            r += cfg.reward.win
        elif game_over:
            r += cfg.reward.lose

        return r
