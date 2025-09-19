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
            capture_base = cfg.reward.capture * len(res.captured_tokens)
            # Apply scaling (annealed externally by modifying cfg.reward.capture_reward_scale)
            capture_base *= cfg.reward.capture_reward_scale
            r += capture_base
            # Optional: capture choice bonus if shaping enabled and multiple moves existed handled upstream
            if cfg.reward.enable_capture_shaping:
                r += cfg.reward.capture_choice_bonus
        elif cfg.reward.enable_capture_shaping:
            # If capture shaping enabled, we may want to penalize declining a capture opportunity.
            # This requires the environment (or caller) to set a flag on MoveResult; fall back gracefully.
            # We expect an attribute like `declined_capture` injected upstream when a capture was available but not taken.
            declined = getattr(res, "declined_capture", False)
            if declined:
                r += cfg.reward.decline_capture_penalty

        if res.finished_token:
            finish_val = cfg.reward.finish_token * cfg.reward.finish_reward_scale
            if cfg.reward.enable_progressive_finish:
                # Assume MoveResult may contain index of finished token order (0-based) as `finish_order_index`
                order_idx = getattr(res, "finish_order_index", None)
                if order_idx is not None:
                    mults = cfg.reward.finish_multipliers
                    if 0 <= order_idx < len(mults):
                        finish_val *= mults[order_idx]
            r += finish_val

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
