from __future__ import annotations


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
        self,
        res: MoveResult,
        illegal: bool,
        cfg: EnvConfig,
        game_over: bool,
        captured_by_opponents: int = 0,
        extra_turn: bool = False,
    ) -> float:
        r = 0.0

        # Event rewards
        if res.captured_tokens:
            r += cfg.reward.capture * len(res.captured_tokens)
        if res.finished_token:
            r += cfg.reward.finish_token

        # Constraint penalties
        if illegal:
            r += cfg.reward.illegal_action

        # Step penalty
        r += cfg.reward.time_penalty

        # Opponent effects during the full turn (after agent acted)
        if captured_by_opponents > 0:
            r += cfg.reward.got_captured * int(captured_by_opponents)

        # Agent bonuses
        if extra_turn:
            r += cfg.reward.extra_turn

        # Terminal outcomes
        if res.game_won:
            r += cfg.reward.win
        elif game_over:
            r += cfg.reward.lose

        return r
