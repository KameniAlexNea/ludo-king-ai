from ludo_engine import LudoGame
from ludo_engine.models import BoardConstants, GameConstants, MoveResult, PlayerColor

from ludo_rl.config import EnvConfig


def token_progress_pos(pos: int, start_pos: int) -> int:
    if pos == GameConstants.HOME_POSITION:
        return 0
    if pos >= GameConstants.HOME_COLUMN_START:
        home_steps = pos - GameConstants.HOME_COLUMN_START + 1
        return GameConstants.MAIN_BOARD_SIZE + home_steps
    # on main board: forward distance from start to current pos
    if pos >= start_pos:
        steps = pos - start_pos
    else:
        steps = GameConstants.MAIN_BOARD_SIZE - start_pos + pos
    return steps + 1


def token_progress(pos: int, start_pos: int) -> float:
    return token_progress_pos(pos, start_pos) / float(
        GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE
    )


class SparseRewardCalculator:
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
        game: LudoGame,
        agent_color: PlayerColor,
        move: MoveResult,
        cfg: EnvConfig,
        return_breakdown: bool = False,
        is_illegal: bool = False,
    ) -> tuple[float, dict]:
        """Compute reward and return a breakdown dict of contributions.

        Returns (reward, breakdown) where breakdown maps component names to floats.
        """
        breakdown = {
            "progress": 0.0,
            "safe_zone": 0.0,
            "capture": 0.0,
            "capture_choice": 0.0,
            "finish": 0.0,
            "illegal": 0.0,
            "time_penalty": 0.0,
            "got_captured": 0.0,
            "all_captured": 0.0,
            "exit_start": 0.0,
            "extra_turn": 0.0,
            "terminal": 0.0,
        }

        player_positions = game.get_player_from_color(agent_color).player_positions()
        home_tokens = sum(
            1 for pos in player_positions if pos == GameConstants.HOME_POSITION
        )
        finished_tokens = sum(
            1 for pos in player_positions if pos == GameConstants.FINISH_POSITION
        )
        nhome_column_tokens = sum(
            1
            for pos in player_positions
            if GameConstants.HOME_POSITION < pos < GameConstants.HOME_COLUMN_START
        )

        reward = 0.0

        # Progress reward
        if agent_color is not None:
            start_pos = BoardConstants.START_POSITIONS[agent_color]
            progress_old = token_progress_pos(move.old_position, start_pos)
            progress_new = token_progress_pos(move.new_position, start_pos)
            delta = progress_new - progress_old
            if delta > 0:
                val = cfg.reward.progress_scale * delta
                breakdown["progress"] += val
                reward += val

        # Safe zone reward
        if not BoardConstants.is_safe_position(
            move.old_position
        ) and BoardConstants.is_safe_position(move.new_position):
            val = cfg.reward.safe_zone_reward
            breakdown["safe_zone"] += val
            reward += val

        # Event rewards
        if move.captured_tokens:
            capture_base = cfg.reward.capture * len(move.captured_tokens)
            # capture_base *= cfg.reward.capture_reward_scale
            breakdown["capture"] += capture_base
            reward += capture_base

        # Finish event
        if move.new_position == GameConstants.FINISH_POSITION:
            finish_val = cfg.reward.finish_token
            breakdown["finish"] += finish_val
            reward += finish_val

        # Constraint penalties
        if is_illegal:
            breakdown["illegal"] += cfg.reward.illegal_action
            reward += cfg.reward.illegal_action

        # Step penalty
        breakdown["time_penalty"] += cfg.reward.time_penalty
        reward += cfg.reward.time_penalty

        # Opponent effects during the full turn (after agent acted)
        if len(move.captured_tokens) > 0:
            val = cfg.reward.got_captured * int(len(move.captured_tokens))
            breakdown["got_captured"] += val
            reward += val
            if GameConstants.TOKENS_PER_PLAYER - finished_tokens == home_tokens:
                breakdown["all_captured"] += cfg.reward.all_captured
                reward += cfg.reward.all_captured

        # Home exit reward: only grant if agent already has at least one other
        # token active on the board (i.e. home_tokens < TOKENS_PER_PLAYER - 1).
        if (
            move.old_position == GameConstants.HOME_POSITION
            and move.new_position != move.old_position
            and nhome_column_tokens > 1
        ):
            # home_tokens counts how many are still at home; reward exit only
            # when there's another token already out (diversity/backup token).
            breakdown["exit_start"] += cfg.reward.exit_start
            reward += cfg.reward.exit_start

        if move.extra_turn:
            breakdown["extra_turn"] += cfg.reward.extra_turn
            reward += cfg.reward.extra_turn

        # Terminal outcomes
        if game.winner is not None and agent_color is not None:
            if game.winner.color == agent_color:
                breakdown["terminal"] += cfg.reward.win
                reward += cfg.reward.win
            else:
                breakdown["terminal"] += cfg.reward.lose
                reward += cfg.reward.lose
        elif game.game_over:
            breakdown["terminal"] += cfg.reward.draw
            reward += cfg.reward.draw

        return (reward, breakdown) if return_breakdown else reward
