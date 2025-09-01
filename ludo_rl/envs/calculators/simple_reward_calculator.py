"""Simple deterministic reward calculation utilities for LudoGymEnv.

This module provides a simplified, non-probabilistic reward function
that scales rewards based on game state, encouraging strategic play
without micromanaging every single action. Incorporates improvements
from gemine suggestion including better token progress calculation
and non-linear reward scaling.
"""

from typing import Dict, List, Optional

from ludo.constants import BoardConstants, GameConstants
from ludo.game import LudoGame
from ludo.player import Player
from ludo.token import Token

from ..model import SimpleRewardConstants as RewardConstants
from ..model import EnvConfig



class SimpleRewardCalculator:
    """Handles simple deterministic reward computation without probabilistic modifiers."""

    def __init__(self, cfg: EnvConfig, game: LudoGame, agent_color: str):
        self.cfg = cfg
        self.game = game
        self.agent_color = agent_color
        self._threat_cache = {}  # Cache for threat calculations
        self._cache_turn = -1
        for opp_player in self.game.players:
            if opp_player.color.value == self.agent_color:
                self.agent_player = opp_player

    def _get_game_phase(self) -> str:
        """Determine current game phase for non-linear reward scaling."""
        agent_player = self._get_agent_player()
        if not agent_player:
            return "early"

        finished_tokens = sum(1 for token in agent_player.tokens if token.position >= GameConstants.FINISH_POSITION)
        tokens_in_home_column = sum(1 for token in agent_player.tokens if BoardConstants.is_home_column_position(token.position))

        if finished_tokens >= RewardConstants.GAME_PHASE_ENDGAME_FINISHED:
            return "endgame"
        elif finished_tokens >= RewardConstants.GAME_PHASE_LATE_FINISHED or tokens_in_home_column >= RewardConstants.GAME_PHASE_LATE_HOME_TOKENS:
            return "late"
        elif any(token.position >= 0 for token in agent_player.tokens):
            return "mid"
        else:
            return "early"

    def _get_agent_player(self) -> Optional[Player]:
        """Get the agent player object."""
        for player in self.game.players:
            if player.color.value == self.agent_color:
                return player
        return None

    def _get_opponent_threat_level(self) -> float:
        """Calculate how close opponents are to winning (0.0 to 1.0) with caching."""
        current_turn = getattr(self.game, "turn_count", 0)

        # Invalidate cache if game state changed
        if current_turn != self._cache_turn:
            self._threat_cache.clear()
            self._cache_turn = current_turn

        cache_key = "opponent_threat"
        if cache_key in self._threat_cache:
            return self._threat_cache[cache_key]

        max_opponent_progress = 0.0

        for player in self.game.players:
            if player.color.value == self.agent_color:
                continue

            finished_tokens = sum(1 for token in player.tokens if token.position >= GameConstants.FINISH_POSITION)
            if finished_tokens >= 3:
                threat = RewardConstants.CRITICAL_THREAT_THRESHOLD  # Critical threat
            elif finished_tokens >= 2:
                threat = RewardConstants.URGENT_THREAT_THRESHOLD
            elif finished_tokens >= 1:
                threat = RewardConstants.OPPONENT_THREAT_FINISHED_1
            else:
                # Check if opponent has tokens in home column
                home_tokens = sum(1 for token in player.tokens if BoardConstants.is_home_column_position(token.position))
                if home_tokens >= 2:
                    threat = RewardConstants.OPPONENT_THREAT_HOME_TOKENS_2
                else:
                    threat = RewardConstants.OPPONENT_THREAT_NONE

            max_opponent_progress = max(max_opponent_progress, threat)

        self._threat_cache[cache_key] = max_opponent_progress
        return max_opponent_progress

    def _get_token_progress(self, token: Token) -> float:
        """Calculate a token's normalized progress towards its finish line (0.0 to 1.0)."""
        return self._get_token_progress_at_position(token, token.position)

    def _get_token_progress_at_position(self, token: Token, position: int) -> float:
        """Calculate a token's normalized progress at a specific position (0.0 to 1.0)."""
        if position >= GameConstants.FINISH_POSITION:
            return 1.0
        if position < 0:
            return 0.0

        # Total number of spaces a token must travel from home to finish
        total_path_length = GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE

        # Calculate steps from start to current position
        player_start_pos = BoardConstants.START_POSITIONS.get(token.player_color)
        if player_start_pos is None:
            return 0.0

        current_steps = 0
        if position < GameConstants.HOME_COLUMN_START:
            # On main board
            # Steps from start position to current position
            if position >= player_start_pos:
                current_steps = position - player_start_pos
            else:
                # Wrap around the board
                current_steps = (GameConstants.MAIN_BOARD_SIZE - player_start_pos) + position
        else:
            # In home column
            current_steps = (GameConstants.MAIN_BOARD_SIZE + (position - GameConstants.HOME_COLUMN_START))

        return min(current_steps / total_path_length, 1.0)

    def _compute_progress_reward(self, move_res: Dict) -> float:
        """
        Reward is non-linear and scales with a token's progress.
        Moving a token closer to the finish line gives a higher reward
        as the token gets closer to the goal.
        """
        rcfg = self.cfg.reward_cfg
        token: Optional[Token] = move_res.get("token")
        if token is None or not isinstance(token, Token):
            return 0.0

        target_pos = move_res.get("target_position")
        if target_pos is None:
            return 0.0

        initial_progress = self._get_token_progress(token)
        final_progress = self._get_token_progress_at_position(token, target_pos)

        delta_progress = final_progress - initial_progress

        # Non-linear scaling: reward is proportional to the square of final progress
        # This heavily rewards moves closer to the finish line
        reward = delta_progress * (final_progress ** RewardConstants.PROGRESS_REWARD_EXPONENT) * rcfg.progress_scale
        return reward

    def _compute_safety_reward(self, move_res: Dict) -> float:
        """Reward moving to safe positions with state-dependent scaling."""
        target_pos = move_res.get("target_position")
        source_pos = move_res.get("source_position")

        if target_pos is None:
            return 0.0

        # Only reward if moving from dangerous to safe position
        if (
            source_pos is not None
            and not BoardConstants.is_safe_position(source_pos)
            and BoardConstants.is_safe_position(target_pos)
        ):
            # Calculate threat level based on nearby opponent tokens
            threat_level = self._calculate_threat_level(target_pos)

            if threat_level > RewardConstants.LOW_THREAT_THRESHOLD:  # Only if there was some threat
                # Non-linear scaling: higher reward for higher threat
                safety_multiplier = min(RewardConstants.SAFETY_MULTIPLIER_MAX,
                                      RewardConstants.SAFETY_MULTIPLIER_BASE + threat_level ** RewardConstants.SAFETY_MULTIPLIER_EXPONENT)
                return self.cfg.reward_cfg.safety_bonus * safety_multiplier

        return 0.0

    def _calculate_threat_level(self, position: int) -> float:
        """Calculate threat level at a position based on opponent proximity with caching."""
        if position < 0 or BoardConstants.is_safe_position(position):
            return 0.0

        current_turn = getattr(self.game, "turn_count", 0)

        # Invalidate cache if game state changed
        if current_turn != self._cache_turn:
            self._threat_cache.clear()
            self._cache_turn = current_turn

        cache_key = f"position_threat_{position}"
        if cache_key in self._threat_cache:
            return self._threat_cache[cache_key]

        threat = 0.0
        for opp_player in self.game.players:
            if opp_player.color.value == self.agent_color:
                continue

            for token in opp_player.tokens:
                if token.position < 0 or BoardConstants.is_home_column_position(token.position):
                    continue

                # Calculate distance to opponent token
                distance = abs(token.position - position)
                if distance <= RewardConstants.MAX_THREAT_DISTANCE:  # Within threat distance
                    threat += 1.0 / (distance + 1)  # Closer = higher threat

        normalized_threat = min(1.0, threat / RewardConstants.THREAT_NORMALIZATION_FACTOR)
        self._threat_cache[cache_key] = normalized_threat
        return normalized_threat

    def _compute_capture_reward(self, move_res: Dict) -> float:
        """
        Reward for capturing an opponent with non-linear scaling based on game state and threat level.
        """
        rcfg = self.cfg.reward_cfg
        captured_tokens = move_res.get("captured_tokens", [])
        if not captured_tokens:
            return 0.0

        game_phase = self._get_game_phase()
        threat_level = self._get_opponent_threat_level()

        # Base capture reward scales with game phase
        base_capture = {
            "early": RewardConstants.CAPTURE_EARLY,
            "mid": RewardConstants.CAPTURE_MID,
            "late": RewardConstants.CAPTURE_LATE,
            "endgame": RewardConstants.CAPTURE_ENDGAME
        }.get(game_phase, RewardConstants.CAPTURE_EARLY)

        total_capture_reward = 0.0
        for captured_token in captured_tokens:
            # Calculate the captured token's progress before capture
            progress_before_capture = self._get_token_progress(captured_token)

            # Base reward with progress scaling
            capture_reward = base_capture * (RewardConstants.SAFETY_MULTIPLIER_BASE + progress_before_capture ** RewardConstants.CAPTURE_PROGRESS_EXPONENT)

            # Bonus for capturing tokens close to winning (in home column)
            if hasattr(captured_token, 'position') and BoardConstants.is_home_column_position(captured_token.position):
                home_progress = captured_token.position - GameConstants.HOME_COLUMN_START
                capture_reward *= (RewardConstants.HOME_COLUMN_CAPTURE_BASE + RewardConstants.HOME_COLUMN_CAPTURE_INCREMENT * home_progress)  # Higher bonus for tokens closer to finish

            # Threat-based multiplier
            if threat_level > RewardConstants.CRITICAL_THREAT_THRESHOLD:
                capture_reward *= RewardConstants.CAPTURE_THREAT_MULTIPLIER_HIGH  # Double reward when opponents are close to winning
            elif threat_level > RewardConstants.MEDIUM_THREAT_THRESHOLD:
                capture_reward *= RewardConstants.CAPTURE_THREAT_MULTIPLIER_MEDIUM

            total_capture_reward += capture_reward

        return total_capture_reward

    def _compute_got_captured_penalty(self, move_res: Dict, token_positions_before: Optional[List[int]] = None) -> float:
        """
        Penalty for being captured with non-linear scaling based on game state.
        """
        rcfg = self.cfg.reward_cfg

        # New approach using token_positions_before if available
        if token_positions_before is not None:
            agent_player = self._get_agent_player()
            if not agent_player:
                return 0.0

            current_positions = [token.position for token in agent_player.tokens]
            captured_tokens = 0
            captured_from_home_column = 0

            # Count tokens that were captured (moved from positive position to -1)
            for before_pos, current_pos in zip(token_positions_before, current_positions):
                if before_pos >= 0 and current_pos < 0:
                    captured_tokens += 1
                    if BoardConstants.is_home_column_position(before_pos):
                        captured_from_home_column += 1

            if captured_tokens == 0:
                return 0.0
        else:
            # Fallback to old approach
            if not move_res.get("was_captured"):
                return 0.0
            captured_tokens = 1
            captured_from_home_column = 0

        game_phase = self._get_game_phase()
        threat_level = self._get_opponent_threat_level()

        # Base penalty scales with game phase (more painful later in game)
        base_penalty = {
            "early": RewardConstants.GOT_CAPTURED_EARLY,
            "mid": RewardConstants.GOT_CAPTURED_MID,
            "late": RewardConstants.GOT_CAPTURED_LATE,
            "endgame": RewardConstants.GOT_CAPTURED_ENDGAME
        }.get(game_phase, RewardConstants.GOT_CAPTURED_MID)

        total_penalty = base_penalty * captured_tokens

        # Extra penalty for losing tokens from home column
        if captured_from_home_column > 0:
            home_penalty = RewardConstants.HOME_COLUMN_CAPTURE_PENALTY * captured_from_home_column
            if game_phase == "endgame":
                home_penalty *= RewardConstants.HOME_COLUMN_CAPTURE_PENALTY_ENDGAME_MULTIPLIER  # Catastrophic in endgame
            total_penalty += home_penalty

        # Increase penalty when we're behind
        if threat_level > RewardConstants.REWARD_THREAT_THRESHOLD_MODERATE:
            total_penalty *= (RewardConstants.SAFETY_MULTIPLIER_BASE + threat_level)

        return total_penalty

    def _compute_moving_reward(self, move_res: Dict) -> float:
        """Reward for making valid moves, scaled by progress and safety with threat modulation."""
        if not move_res.get("moved"):
            return 0.0

        target_pos = move_res.get("target_position")
        source_pos = move_res.get("source_position")
        if target_pos is None:
            return 0.0

        base_reward = 0.0
        game_phase = self._get_game_phase()
        threat_level = self._get_opponent_threat_level()

        # Reward getting tokens out of home
        if source_pos is not None and source_pos < 0 and target_pos >= 0:
            if game_phase == "early":
                base_reward = RewardConstants.MOVING_EXIT_HOME_EARLY
            elif game_phase == "mid":
                base_reward = RewardConstants.MOVING_EXIT_HOME_MID
            else:
                base_reward = RewardConstants.MOVING_EXIT_HOME_LATE

        # Reward entering home column
        elif (source_pos is not None and source_pos < GameConstants.HOME_COLUMN_START
              and target_pos >= GameConstants.HOME_COLUMN_START):
            if game_phase == "late" or game_phase == "endgame":
                base_reward = RewardConstants.MOVING_ENTER_HOME_LATE
            else:
                base_reward = RewardConstants.MOVING_ENTER_HOME_EARLY

        # Reward progress in home column (exponentially increasing)
        elif BoardConstants.is_home_column_position(target_pos):
            home_progress = target_pos - GameConstants.HOME_COLUMN_START
            if game_phase == "endgame":
                base_reward = RewardConstants.MOVING_HOME_PROGRESS_ENDGAME * (RewardConstants.HOME_PROGRESS_EXPONENT_ENDGAME ** home_progress)  # Exponential scaling
            else:
                base_reward = RewardConstants.MOVING_HOME_PROGRESS_EARLY * (RewardConstants.HOME_PROGRESS_EXPONENT_EARLY ** home_progress)

        # Regular board movement (minimal reward)
        else:
            base_reward = RewardConstants.MOVING_REGULAR

        # Scale by opponent threat level
        if threat_level > RewardConstants.REWARD_THREAT_THRESHOLD_HIGH:
            base_reward *= (RewardConstants.SAFETY_MULTIPLIER_BASE + threat_level * RewardConstants.MOVING_THREAT_MULTIPLIER)

        # Bonus for safe moves
        if BoardConstants.is_safe_position(target_pos):
            base_reward += RewardConstants.MOVING_SAFE_BONUS

        # Bonus for progress
        token: Optional[Token] = move_res.get("token")
        progress_value = self._get_token_progress(token) if token else 0.0
        progress_bonus = progress_value * RewardConstants.MOVING_PROGRESS_BONUS_SCALE

        # Non-linear scaling for significant progress
        if progress_value > RewardConstants.PROGRESS_SIGNIFICANT_MULTIPLIER:
            progress_bonus *= RewardConstants.MOVING_PROGRESS_SIGNIFICANT_MULTIPLIER

        return base_reward + progress_bonus

    def compute_progress_reward(
        self, progress_before: float, progress_after: float
    ) -> float:
        """Compute progress-based reward with non-linear scaling."""
        delta = progress_after - progress_before
        if abs(delta) < RewardConstants.PROGRESS_SIGNIFICANT_THRESHOLD:  # Only significant progress
            return 0.0

        # Non-linear scaling: more reward for larger progress steps
        scaled_delta = delta * (RewardConstants.SAFETY_MULTIPLIER_BASE + abs(delta) * RewardConstants.PROGRESS_REWARD_QUADRATIC_SCALE)  # Quadratic scaling
        reward = scaled_delta * self.cfg.reward_cfg.progress_scale

        # Cap extreme values
        return max(RewardConstants.PROGRESS_REWARD_CAP_MIN, min(RewardConstants.PROGRESS_REWARD_CAP_MAX, reward))

    def compute_comprehensive_reward(
        self,
        move_res: Dict,
        progress_delta: float,
        extra_turn: bool,
        diversity_bonus: bool,
        illegal_action: bool,
        reward_components: List[float],
        token_positions_before: Optional[List[int]] = None,
    ) -> float:
        """Comprehensive deterministic reward system with non-linear scaling."""
        rcfg = self.cfg.reward_cfg
        total_reward = 0.0

        # Major event rewards (highest priority)
        capture_reward = self._compute_capture_reward(move_res)
        total_reward += capture_reward
        reward_components.append(capture_reward)  # capture

        got_captured_penalty = self._compute_got_captured_penalty(move_res, token_positions_before)
        total_reward += got_captured_penalty
        reward_components.append(got_captured_penalty)  # got_captured

        # Non-linear progress reward
        progress_reward = self._compute_progress_reward(move_res)
        total_reward += progress_reward
        reward_components.append(progress_reward)

        if move_res.get("token_finished"):
            game_phase = self._get_game_phase()
            threat_level = self._get_opponent_threat_level()

            finish_reward = {
                "early": RewardConstants.FINISH_EARLY,
                "mid": RewardConstants.FINISH_MID,
                "late": RewardConstants.FINISH_LATE,
                "endgame": RewardConstants.FINISH_ENDGAME
            }.get(game_phase, RewardConstants.FINISH_EARLY)

            # Urgent finish bonus when opponents are threatening
            if threat_level > RewardConstants.URGENT_THREAT_THRESHOLD:
                finish_reward *= RewardConstants.FINISH_THREAT_MULTIPLIER_URGENT
            elif threat_level > RewardConstants.REWARD_THREAT_THRESHOLD_HIGH:
                finish_reward *= RewardConstants.FINISH_THREAT_MULTIPLIER_HIGH

            total_reward += finish_reward
            reward_components.append(finish_reward)  # finish

        # Moderate event rewards
        if extra_turn:
            threat_level = self._get_opponent_threat_level()
            extra_turn_reward = rcfg.extra_turn
            if threat_level > RewardConstants.REWARD_THREAT_THRESHOLD_HIGH:
                extra_turn_reward *= (RewardConstants.SAFETY_MULTIPLIER_BASE + threat_level * RewardConstants.EXTRA_TURN_THREAT_MULTIPLIER)
            total_reward += extra_turn_reward
            reward_components.append(extra_turn_reward)  # extra_turn

        # Strategic rewards
        safety_reward = self._compute_safety_reward(move_res)
        total_reward += safety_reward
        reward_components.append(safety_reward)  # safety

        moving_reward = self._compute_moving_reward(move_res)
        total_reward += moving_reward
        reward_components.append(moving_reward)  # moving

        # Blocking reward
        blocking_reward = self._compute_blocking_reward(move_res)
        total_reward += blocking_reward
        reward_components.append(blocking_reward)  # blocking

        # Small bonuses and penalties
        if diversity_bonus:
            diversity_reward = rcfg.diversity_bonus
            total_reward += diversity_reward
            reward_components.append(diversity_reward)  # diversity

        if illegal_action:
            total_reward += rcfg.illegal_action  # Don't modulate penalties
            reward_components.append(rcfg.illegal_action)  # illegal

        # Progress reward
        if abs(progress_delta) > RewardConstants.PROGRESS_SIGNIFICANT_THRESHOLD:
            progress_reward = self.compute_progress_reward(0.0, progress_delta)
            total_reward += progress_reward
            reward_components.append(progress_reward)  # progress

        # Time penalty (very small, just to encourage efficiency)
        total_reward += rcfg.time_penalty
        reward_components.append(rcfg.time_penalty)  # time

        return total_reward

    def _is_blocked_by_position(self, token_position: int, blocking_position: int) -> bool:
        """Check if a token is blocked by another position."""
        if token_position < 0 or blocking_position < 0:
            return False

        # Direct blocking (same position)
        if token_position == blocking_position:
            return True

        # Check if blocking position creates a barrier
        # For Ludo, blocking typically occurs at the same position
        # But we can extend this for strategic blocking patterns
        return abs(token_position - blocking_position) <= RewardConstants.BLOCKING_DISTANCE

    def _compute_blocking_reward(self, move_res: Dict) -> float:
        """Compute reward for blocking opponent tokens."""
        reward = 0.0

        # Get the token that was moved
        token = move_res.get("token")
        if token is None or not isinstance(token, Token):
            return 0.0

        target_pos = move_res.get("target_position")
        if target_pos is None:
            return 0.0

        # Reward for blocking opponent home entry
        # Home entry positions are the positions right before entering home column
        home_entries = BoardConstants.HOME_COLUMN_ENTRIES
        if target_pos in home_entries.values():
            reward += RewardConstants.BLOCKING_HOME_ENTRY_REWARD

        # Reward for stacking on own tokens (creates stronger blocks)
        own_tokens_at_position = sum(1 for t in self.agent_player.tokens
                                   if t.position == target_pos and t.id != token.id)
        if own_tokens_at_position > 0:
            reward += RewardConstants.STACKING_REWARD * own_tokens_at_position

        # Reward for blocking opponent advancement
        for opp_player in self.game.players:
            if opp_player.color.value == self.agent_color:
                continue

            for opp_token in opp_player.tokens:
                if opp_token.position < 0:
                    continue

                # Check if opponent token is blocked by our new position
                if self._is_blocked_by_position(opp_token.position, target_pos):
                    reward += RewardConstants.BLOCKING_ADVANCEMENT_REWARD

        return reward

    def get_terminal_reward(
        self, agent_player: Player, opponents: list[Player]
    ) -> float:
        """Compute terminal rewards with non-linear scaling based on final state."""
        if agent_player.has_won():
            # Win reward scales with how decisive the victory was
            agent_finished = sum(1 for token in agent_player.tokens if token.position >= GameConstants.FINISH_POSITION)
            max_opponent_finished = max(
                sum(1 for token in opp.tokens if token.position >= GameConstants.FINISH_POSITION)
                for opp in opponents
            )

            # Bonus for decisive wins (finishing with opponents having fewer tokens)
            win_margin = agent_finished - max_opponent_finished
            base_win_reward = self.cfg.reward_cfg.win

            if win_margin >= RewardConstants.WIN_MARGIN_DOMINANT:
                return base_win_reward * RewardConstants.WIN_MULTIPLIER_DOMINANT  # Dominant victory
            elif win_margin >= RewardConstants.WIN_MARGIN_CLEAR:
                return base_win_reward * RewardConstants.WIN_MULTIPLIER_CLEAR  # Clear victory
            else:
                return base_win_reward  # Close victory

        elif any(opp.has_won() for opp in opponents):
            # Loss penalty scales with how badly we lost
            agent_finished = sum(1 for token in agent_player.tokens if token.position >= GameConstants.FINISH_POSITION)
            winner_finished = GameConstants.TOKENS_TO_WIN  # Winner has all tokens finished

            loss_margin = winner_finished - agent_finished
            base_loss_penalty = self.cfg.reward_cfg.lose

            if loss_margin >= RewardConstants.LOSS_MARGIN_SHUTOUT:
                return base_loss_penalty * RewardConstants.LOSS_MULTIPLIER_SHUTOUT  # Shutout loss (0 tokens finished)
            elif loss_margin >= RewardConstants.LOSS_MARGIN_BAD:
                return base_loss_penalty * RewardConstants.LOSS_MULTIPLIER_BAD  # Bad loss
            else:
                return base_loss_penalty  # Close loss

        return 0.0
