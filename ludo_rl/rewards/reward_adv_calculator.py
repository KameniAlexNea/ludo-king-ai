from typing import Dict, Any
from ludo_engine import LudoGame, Player
from ludo_engine.models import BoardConstants, GameConstants, MoveResult, PlayerColor
from ludo_rl.config import EnvConfig


class AdvancedRewardCalculator:
    """Advanced reward calculator that incorporates strategic Ludo opportunities and risks.

    This calculator goes beyond simple events to reward/penalize based on:
    - Opportunity utilization (captures, exits taken vs available)
    - Risk management (vulnerable token changes)
    - Strategic positioning (safe zones, blocking, diversity)
    - Long-term planning (progress efficiency, opponent pressure)
    """

    def __init__(self):
        self.prev_opportunities = {}  # Track previous episode opportunities for comparison

    def compute(
        self,
        game: LudoGame,
        agent_color: PlayerColor,
        move: MoveResult,
        cfg: EnvConfig,
        episode_info: Dict[str, Any],
        return_breakdown: bool = False,
        is_illegal: bool = False,
    ) -> tuple[float, dict]:
        """Compute advanced reward with strategic components.

        Args:
            game: Current game state
            agent_color: Agent's color
            move: Result of the agent's move
            cfg: Environment config
            episode_info: Episode tracking info (opportunities, etc.)
            return_breakdown: Whether to return component breakdown
            is_illegal: Whether move was illegal

        Returns:
            (reward, breakdown) tuple
        """
        breakdown = {
            # Basic components (from sparse)
            "progress": 0.0,
            "safe_zone": 0.0,
            "capture": 0.0,
            "finish": 0.0,
            "illegal": 0.0,
            "time_penalty": 0.0,
            "got_captured": 0.0,
            "all_captured": 0.0,
            "exit_start": 0.0,
            "diversity_bonus": 0.0,
            "extra_turn": 0.0,
            "terminal": 0.0,
            # Advanced strategic components
            "capture_opportunity_taken": 0.0,
            "capture_opportunity_missed": 0.0,
            "exit_opportunity_taken": 0.0,
            "exit_opportunity_missed": 0.0,
            "vulnerability_reduction": 0.0,
            "vulnerability_increase": 0.0,
            "blocking_bonus": 0.0,
            "progress_efficiency": 0.0,
            "opponent_pressure_relief": 0.0,
        }

        reward = 0.0
        agent_player = game.get_player_from_color(agent_color)

        # Basic components (similar to sparse calculator)
        reward += self._compute_basic_components(game, agent_color, move, cfg, breakdown, is_illegal)

        # Advanced strategic components
        reward += self._compute_opportunity_rewards(episode_info, cfg, breakdown)
        reward += self._compute_risk_management(game, agent_color, move, cfg, breakdown)
        reward += self._compute_strategic_positioning(game, agent_color, move, cfg, breakdown)
        reward += self._compute_long_term_planning(game, agent_color, move, cfg, episode_info, breakdown)

        return (float(reward), breakdown) if return_breakdown else reward

    def _compute_basic_components(self, game: LudoGame, agent_color: PlayerColor, move: MoveResult, cfg: EnvConfig, breakdown: dict, is_illegal: bool) -> float:
        """Compute basic reward components (similar to SparseRewardCalculator)."""
        reward = 0.0

        # Progress reward
        if agent_color is not None:
            start_pos = BoardConstants.START_POSITIONS[agent_color]
            progress_old = self._token_progress_pos(move.old_position, start_pos)
            progress_new = self._token_progress_pos(move.new_position, start_pos)
            delta = progress_new - progress_old
            if delta > 0:
                val = cfg.reward.progress_scale * delta
                breakdown["progress"] += val
                reward += val

        # Safe zone reward
        if (
            move.old_position != move.new_position
            and move.new_position in BoardConstants.STAR_SQUARES
        ):
            val = cfg.reward.safe_zone_reward
            breakdown["safe_zone"] += val
            reward += val

        # Event rewards
        if move.captured_tokens:
            capture_base = cfg.reward.capture * len(move.captured_tokens)
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

        # Opponent effects
        if len(move.captured_tokens) > 0:
            val = cfg.reward.got_captured * int(len(move.captured_tokens))
            breakdown["got_captured"] += val
            reward += val

            home_tokens = sum(1 for pos in game.get_player_from_color(agent_color).player_positions()
                            if pos == GameConstants.HOME_POSITION)
            finished_tokens = sum(1 for pos in game.get_player_from_color(agent_color).player_positions()
                                if pos == GameConstants.FINISH_POSITION)
            if GameConstants.TOKENS_PER_PLAYER - finished_tokens == home_tokens:
                breakdown["all_captured"] += cfg.reward.all_captured
                reward += cfg.reward.all_captured

        # Home exit reward
        if (
            move.old_position == GameConstants.HOME_POSITION
            and move.new_position != move.old_position
        ):
            breakdown["exit_start"] += cfg.reward.exit_start
            reward += cfg.reward.exit_start

        # Diversity bonus
        nhome_column_tokens = sum(
            1 for pos in game.get_player_from_color(agent_color).player_positions()
            if GameConstants.HOME_POSITION < pos < GameConstants.HOME_COLUMN_START
        )
        if nhome_column_tokens > 1:
            breakdown["diversity_bonus"] += cfg.reward.diversity_bonus * nhome_column_tokens
            reward += cfg.reward.diversity_bonus * nhome_column_tokens

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

        return reward

    def _compute_opportunity_rewards(self, episode_info: dict, cfg: EnvConfig, breakdown: dict) -> float:
        """Reward based on opportunity utilization."""
        reward = 0.0

        # Capture opportunities
        capture_available = episode_info.get('episode_capture_ops_available', 0)
        capture_taken = episode_info.get('episode_capture_ops_taken', 0)
        if capture_available > 0:
            capture_ratio = capture_taken / capture_available
            # Reward taking captures, penalize missing them
            taken_reward = capture_taken * 0.5  # Bonus for each taken
            missed_penalty = (capture_available - capture_taken) * -0.3  # Penalty for missed
            breakdown["capture_opportunity_taken"] += taken_reward
            breakdown["capture_opportunity_missed"] += missed_penalty
            reward += taken_reward + missed_penalty

        # Exit opportunities
        exit_available = episode_info.get('episode_home_exit_ops_available', 0)
        exit_taken = episode_info.get('episode_home_exit_ops_taken', 0)
        if exit_available > 0:
            exit_ratio = exit_taken / exit_available
            # Reward taking exits, penalize missing them (more than captures since exits are fundamental)
            taken_reward = exit_taken * 0.4
            missed_penalty = (exit_available - exit_taken) * -0.2
            breakdown["exit_opportunity_taken"] += taken_reward
            breakdown["exit_opportunity_missed"] += missed_penalty
            reward += taken_reward + missed_penalty

        # Finish opportunities
        finish_available = episode_info.get('episode_finish_ops_available', 0)
        finish_taken = episode_info.get('episode_finish_ops_taken', 0)
        if finish_available > 0:
            finish_ratio = finish_taken / finish_available
            taken_reward = finish_taken * 0.6  # High reward for finishing
            missed_penalty = (finish_available - finish_taken) * -0.1
            breakdown["finish_opportunity_taken"] = taken_reward
            breakdown["finish_opportunity_missed"] = missed_penalty
            reward += taken_reward + missed_penalty

        return reward

    def _compute_risk_management(self, game: LudoGame, agent_color: PlayerColor, move: MoveResult, cfg: EnvConfig, breakdown: dict) -> float:
        """Reward for managing risk (vulnerable tokens)."""
        reward = 0.0

        # Calculate vulnerable tokens before and after move
        vuln_before = self._count_vulnerable_tokens(game, agent_color, before_move=True, move=move)
        vuln_after = self._count_vulnerable_tokens(game, agent_color, before_move=False, move=move)

        vuln_delta = vuln_before - vuln_after
        if vuln_delta > 0:
            # Reduced vulnerability - good
            val = vuln_delta * 0.3
            breakdown["vulnerability_reduction"] += val
            reward += val
        elif vuln_delta < 0:
            # Increased vulnerability - bad
            val = vuln_delta * -0.4  # Stronger penalty
            breakdown["vulnerability_increase"] += val
            reward += val

        return reward

    def _compute_strategic_positioning(self, game: LudoGame, agent_color: PlayerColor, move: MoveResult, cfg: EnvConfig, breakdown: dict) -> float:
        """Reward for strategic positioning (blocking, safe zones, etc.)."""
        reward = 0.0

        # Blocking bonus: reward if move blocks opponent progress
        if self._move_blocks_opponent(game, agent_color, move):
            val = 0.2
            breakdown["blocking_bonus"] += val
            reward += val

        # Additional safe zone bonus if multiple tokens in safe zones
        safe_tokens = sum(1 for pos in game.get_player_from_color(agent_color).player_positions()
                         if pos in BoardConstants.STAR_SQUARES)
        if safe_tokens > 1:
            val = (safe_tokens - 1) * 0.1  # Bonus for each additional safe token
            breakdown["safe_zone"] += val  # Add to existing safe_zone
            reward += val

        return reward

    def _compute_long_term_planning(self, game: LudoGame, agent_color: PlayerColor, move: MoveResult, cfg: EnvConfig, episode_info: dict, breakdown: dict) -> float:
        """Reward for long-term planning aspects."""
        reward = 0.0

        # Progress efficiency: reward if progress is made efficiently (not wasting moves)
        if move.old_position != move.new_position:
            start_pos = BoardConstants.START_POSITIONS[agent_color]
            progress_made = self._token_progress_pos(move.new_position, start_pos) - self._token_progress_pos(move.old_position, start_pos)
            if progress_made > 0:
                # Efficiency bonus based on dice roll utilization
                dice_value = game.dice_value if hasattr(game, 'dice_value') else 6  # Assume max if unknown
                efficiency = min(progress_made / dice_value, 1.0)  # How much of dice used for progress
                val = efficiency * 0.1
                breakdown["progress_efficiency"] += val
                reward += val

        # Opponent pressure relief: reward if opponents have fewer active tokens after move
        opp_active_before = episode_info.get('opponents_active_before', 3)  # Assume 3 if not tracked
        opp_active_after = sum(1 for p in game.players if p.color != agent_color and
                              any(pos != GameConstants.HOME_POSITION for pos in p.player_positions()))
        pressure_relief = opp_active_before - opp_active_after
        if pressure_relief > 0:
            val = pressure_relief * 0.15
            breakdown["opponent_pressure_relief"] += val
            reward += val

        return reward

    def _token_progress_pos(self, pos: int, start_pos: int) -> int:
        """Calculate token progress position (copied from reward_calculator)."""
        if pos == GameConstants.HOME_POSITION:
            return 0
        if pos >= GameConstants.HOME_COLUMN_START:
            home_steps = pos - GameConstants.HOME_COLUMN_START + 1
            return GameConstants.MAIN_BOARD_SIZE + home_steps
        if pos >= start_pos:
            steps = pos - start_pos
        else:
            steps = GameConstants.MAIN_BOARD_SIZE - start_pos + pos
        return steps + 1

    def _count_vulnerable_tokens(self, game: LudoGame, agent_color: PlayerColor, before_move=True, move=None) -> int:
        """Count how many of agent's tokens are vulnerable to capture."""
        agent_positions = game.get_player_from_color(agent_color).player_positions()
        if not before_move and move:
            # Adjust for the move
            agent_positions = [move.new_position if pos == move.old_position else pos for pos in agent_positions]

        vulnerable = 0
        for pos in agent_positions:
            if pos == GameConstants.HOME_POSITION or pos == GameConstants.FINISH_POSITION:
                continue
            # Check if any opponent can capture this position
            for player in game.players:
                if player.color == agent_color:
                    continue
                opp_positions = player.player_positions()
                # Simple check: if opponent has token that can reach this pos with exact dice
                for opp_pos in opp_positions:
                    if opp_pos != GameConstants.HOME_POSITION and opp_pos != GameConstants.FINISH_POSITION:
                        # Calculate distance (simplified)
                        dist = (pos - opp_pos) % GameConstants.MAIN_BOARD_SIZE
                        if dist <= 6 and dist > 0:  # Can reach with dice 1-6
                            vulnerable += 1
                            break
        return vulnerable

    def _move_blocks_opponent(self, game: LudoGame, agent_color: PlayerColor, move: MoveResult) -> bool:
        """Check if the move blocks opponent progress."""
        # Simple check: if new position is just ahead of opponent token
        for player in game.players:
            if player.color == agent_color:
                continue
            opp_positions = player.player_positions()
            for opp_pos in opp_positions:
                if opp_pos != GameConstants.HOME_POSITION and opp_pos != GameConstants.FINISH_POSITION:
                    # If agent's new pos is one step ahead of opponent
                    if (move.new_position - opp_pos) % GameConstants.MAIN_BOARD_SIZE == 1:
                        return True
        return False