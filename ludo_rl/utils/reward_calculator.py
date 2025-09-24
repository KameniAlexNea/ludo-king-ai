from typing import Optional

from ludo_engine.core import Player
from ludo_engine.models import GameConstants, MoveResult, PlayerColor, BoardConstants

from ludo_rl.config import EnvConfig


def token_progress(pos: int, start_pos: int) -> float:
    if pos == GameConstants.HOME_POSITION:
        return 0.0
    if pos >= GameConstants.HOME_COLUMN_START:
        home_steps = (
            pos - GameConstants.HOME_COLUMN_START + 1
        )
        return (GameConstants.MAIN_BOARD_SIZE + home_steps) / float(
            GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE
        )
    # on main board: forward distance from start to current pos
    if pos >= start_pos:
        steps = pos - start_pos
    else:
        steps = GameConstants.MAIN_BOARD_SIZE - start_pos + pos
    return steps / float(GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE)


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
        extra_turn: bool = 0,
        winner: Optional[Player] = None,
        agent_color: Optional[PlayerColor] = None,
        home_tokens: int = 0,
    ) -> float:
        r = 0.0

        # Progress reward
        if agent_color is not None:
            start_pos = BoardConstants.START_POSITIONS[agent_color]
            progress_old = token_progress(res.old_position, start_pos)
            progress_new = token_progress(res.new_position, start_pos)
            delta = progress_new - progress_old
            if delta > 0:
                r += cfg.reward.progress_scale * delta

        # Safe zone reward
        if not BoardConstants.is_safe_position(res.old_position) and BoardConstants.is_safe_position(res.new_position):
            r += cfg.reward.safe_zone_reward

        # Event rewards
        if res.captured_tokens:
            capture_base = cfg.reward.capture * len(res.captured_tokens)
            # Apply scaling (annealed externally by modifying cfg.reward.capture_reward_scale)
            capture_base *= cfg.reward.capture_reward_scale
            r += capture_base
            # Optional: capture choice bonus if shaping enabled and multiple moves existed handled upstream
            if cfg.reward.enable_capture_shaping:
                r += cfg.reward.capture_choice_bonus
        # No else branch: we no longer assume any declined-capture tagging attribute on MoveResult.

        # Finish event: consider either engine flag or positional equality safeguard
        # Finish event: rely strictly on known dataclass fields and position equality safeguard.
        if res.finished_token or res.new_position == GameConstants.FINISH_POSITION:
            finish_val = cfg.reward.finish_token * cfg.reward.finish_reward_scale
            # Progressive finish removed: it relied on an attribute not in MoveResult dataclass.
            r += finish_val

        # Constraint penalties
        if illegal:
            r += cfg.reward.illegal_action

        # Step penalty
        r += cfg.reward.time_penalty

        # Opponent effects during the full turn (after agent acted)
        if captured_by_opponents > 0:
            r += cfg.reward.got_captured * int(captured_by_opponents)
            if home_tokens == 0:
                r += cfg.reward.all_captured

        if (
            res.old_position == GameConstants.HOME_POSITION
            and res.new_position != res.old_position
        ):
            r += cfg.reward.exit_start

        # Agent bonuses
        if extra_turn:
            r += cfg.reward.extra_turn

        # Terminal outcomes
        if winner is not None and agent_color is not None:
            if winner.color == agent_color:
                r += cfg.reward.win
            else:
                r += cfg.reward.lose
        elif game_over:
            r += cfg.reward.draw

        return r
