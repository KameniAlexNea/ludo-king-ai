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

from ..model import EnvConfig


class SimpleRewardCalculator:
    """Handles simple deterministic reward computation without probabilistic modifiers."""

    def __init__(self, cfg: EnvConfig, game: LudoGame, agent_color: str):
        self.cfg = cfg
        self.game = game
        self.agent_color = agent_color

    def _get_game_phase(self) -> str:
        """Determine current game phase for non-linear reward scaling."""
        agent_player = self._get_agent_player()
        if not agent_player:
            return "early"

        finished_tokens = sum(1 for token in agent_player.tokens if token.position >= GameConstants.FINISH_POSITION)
        tokens_in_home_column = sum(1 for token in agent_player.tokens if BoardConstants.is_home_column_position(token.position))

        if finished_tokens >= 2:
            return "endgame"
        elif finished_tokens >= 1 or tokens_in_home_column >= 2:
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
        """Calculate how close opponents are to winning (0.0 to 1.0)."""
        max_opponent_progress = 0.0

        for player in self.game.players:
            if player.color.value == self.agent_color:
                continue

            finished_tokens = sum(1 for token in player.tokens if token.position >= GameConstants.FINISH_POSITION)
            if finished_tokens >= 3:
                return 1.0  # Critical threat
            elif finished_tokens >= 2:
                max_opponent_progress = max(max_opponent_progress, 0.8)
            elif finished_tokens >= 1:
                max_opponent_progress = max(max_opponent_progress, 0.5)
            else:
                # Check if opponent has tokens in home column
                home_tokens = sum(1 for token in player.tokens if BoardConstants.is_home_column_position(token.position))
                if home_tokens >= 2:
                    max_opponent_progress = max(max_opponent_progress, 0.3)

        return max_opponent_progress

    def _get_token_progress(self, token: Token) -> float:
        """Calculate a token's normalized progress towards its finish line (0.0 to 1.0)."""
        if token.position >= GameConstants.FINISH_POSITION:
            return 1.0
        if token.position < 0:
            return 0.0

        # Total number of spaces a token must travel from home to finish
        total_path_length = GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE

        # Calculate steps from start to current position
        player_start_pos = BoardConstants.START_POSITIONS.get(token.player_color)
        if player_start_pos is None:
            return 0.0

        current_steps = 0
        if token.position < GameConstants.HOME_COLUMN_START:
            # On main board
            # Steps from start position to current position
            if token.position >= player_start_pos:
                current_steps = token.position - player_start_pos
            else:
                # Wrap around the board
                current_steps = (GameConstants.MAIN_BOARD_SIZE - player_start_pos) + token.position
        else:
            # In home column
            current_steps = (GameConstants.MAIN_BOARD_SIZE + (token.position - GameConstants.HOME_COLUMN_START))

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

        initial_progress = self._get_token_progress(token)
        # Temporarily update token position to calculate progress after move
        temp_pos = token.position
        token.position = move_res.get("target_position")
        final_progress = self._get_token_progress(token)
        # Revert position to avoid side effects
        token.position = temp_pos

        delta_progress = final_progress - initial_progress

        # Non-linear scaling: reward is proportional to the square of final progress
        # This heavily rewards moves closer to the finish line
        reward = delta_progress * (final_progress ** 2) * rcfg.progress_scale
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

            if threat_level > 0.1:  # Only if there was some threat
                # Non-linear scaling: higher reward for higher threat
                safety_multiplier = min(2.0, 1.0 + threat_level ** 1.5)
                return self.cfg.reward_cfg.safety_bonus * safety_multiplier

        return 0.0

    def _calculate_threat_level(self, position: int) -> float:
        """Calculate threat level at a position based on opponent proximity."""
        if position < 0 or BoardConstants.is_safe_position(position):
            return 0.0

        threat = 0.0
        for opp_player in self.game.players:
            if opp_player.color.value == self.agent_color:
                continue

            for token in opp_player.tokens:
                if token.position < 0 or BoardConstants.is_home_column_position(token.position):
                    continue

                # Calculate distance to opponent token
                distance = abs(token.position - position)
                if distance <= 6:  # Within 6 spaces
                    threat += 1.0 / (distance + 1)  # Closer = higher threat

        return min(1.0, threat / 2.0)  # Normalize

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
            "early": 8.0,
            "mid": 12.0,
            "late": 18.0,
            "endgame": 25.0
        }.get(game_phase, 8.0)

        total_capture_reward = 0.0
        for captured_token in captured_tokens:
            # Calculate the captured token's progress before capture
            progress_before_capture = self._get_token_progress(captured_token)

            # Base reward with progress scaling
            capture_reward = base_capture * (1 + progress_before_capture ** 2)

            # Bonus for capturing tokens close to winning (in home column)
            if hasattr(captured_token, 'position') and BoardConstants.is_home_column_position(captured_token.position):
                home_progress = captured_token.position - GameConstants.HOME_COLUMN_START
                capture_reward *= (1.5 + 0.3 * home_progress)  # Higher bonus for tokens closer to finish

            # Threat-based multiplier
            if threat_level > 0.7:
                capture_reward *= 2.0  # Double reward when opponents are close to winning
            elif threat_level > 0.4:
                capture_reward *= 1.5

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
            "early": -3.0,
            "mid": -5.0,
            "late": -10.0,
            "endgame": -20.0
        }.get(game_phase, -5.0)

        total_penalty = base_penalty * captured_tokens

        # Extra penalty for losing tokens from home column
        if captured_from_home_column > 0:
            home_penalty = -15.0 * captured_from_home_column
            if game_phase == "endgame":
                home_penalty *= 3.0  # Catastrophic in endgame
            total_penalty += home_penalty

        # Increase penalty when we're behind
        if threat_level > 0.6:
            total_penalty *= (1.0 + threat_level)

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
                base_reward = 5.0
            elif game_phase == "mid":
                base_reward = 3.0
            else:
                base_reward = 2.0

        # Reward entering home column
        elif (source_pos is not None and source_pos < GameConstants.HOME_COLUMN_START
              and target_pos >= GameConstants.HOME_COLUMN_START):
            if game_phase == "late" or game_phase == "endgame":
                base_reward = 8.0
            else:
                base_reward = 4.0

        # Reward progress in home column (exponentially increasing)
        elif BoardConstants.is_home_column_position(target_pos):
            home_progress = target_pos - GameConstants.HOME_COLUMN_START
            if game_phase == "endgame":
                base_reward = 2.0 * (1.5 ** home_progress)  # Exponential scaling
            else:
                base_reward = 1.0 * (1.3 ** home_progress)

        # Regular board movement (minimal reward)
        else:
            base_reward = 0.5

        # Scale by opponent threat level
        if threat_level > 0.5:
            base_reward *= (1.0 + threat_level)

        # Bonus for safe moves
        if BoardConstants.is_safe_position(target_pos):
            base_reward += 0.05

        # Bonus for progress
        token: Optional[Token] = move_res.get("token")
        progress_value = self._get_token_progress(token) if token else 0.0
        progress_bonus = progress_value * 0.2

        # Non-linear scaling for significant progress
        if progress_value > 0.5:
            progress_bonus *= 1.5

        return base_reward + progress_bonus

    def compute_progress_reward(
        self, progress_before: float, progress_after: float
    ) -> float:
        """Compute progress-based reward with non-linear scaling."""
        delta = progress_after - progress_before
        if abs(delta) < 0.01:  # Only significant progress
            return 0.0

        # Non-linear scaling: more reward for larger progress steps
        scaled_delta = delta * (1 + abs(delta))  # Quadratic scaling
        reward = scaled_delta * self.cfg.reward_cfg.progress_scale

        # Cap extreme values
        return max(-5.0, min(5.0, reward))

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
                "early": 20.0,
                "mid": 30.0,
                "late": 50.0,
                "endgame": 100.0
            }.get(game_phase, 20.0)

            # Urgent finish bonus when opponents are threatening
            if threat_level > 0.8:
                finish_reward *= 3.0
            elif threat_level > 0.5:
                finish_reward *= 2.0

            total_reward += finish_reward
            reward_components.append(finish_reward)  # finish

        # Moderate event rewards
        if extra_turn:
            threat_level = self._get_opponent_threat_level()
            extra_turn_reward = rcfg.extra_turn
            if threat_level > 0.5:
                extra_turn_reward *= (1.0 + threat_level)
            total_reward += extra_turn_reward
            reward_components.append(extra_turn_reward)  # extra_turn

        # Strategic rewards
        safety_reward = self._compute_safety_reward(move_res)
        total_reward += safety_reward
        reward_components.append(safety_reward)  # safety

        moving_reward = self._compute_moving_reward(move_res)
        total_reward += moving_reward
        reward_components.append(moving_reward)  # moving

        # Small bonuses and penalties
        if diversity_bonus:
            diversity_reward = rcfg.diversity_bonus
            total_reward += diversity_reward
            reward_components.append(diversity_reward)  # diversity

        if illegal_action:
            total_reward += rcfg.illegal_action  # Don't modulate penalties
            reward_components.append(rcfg.illegal_action)  # illegal

        # Progress reward
        if abs(progress_delta) > 0.01:
            progress_reward = self.compute_progress_reward(0.0, progress_delta)
            total_reward += progress_reward
            reward_components.append(progress_reward)  # progress

        # Time penalty (very small, just to encourage efficiency)
        total_reward += rcfg.time_penalty
        reward_components.append(rcfg.time_penalty)  # time

        return total_reward

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

            if win_margin >= 3:
                return base_win_reward * 2.0  # Dominant victory
            elif win_margin >= 2:
                return base_win_reward * 1.5  # Clear victory
            else:
                return base_win_reward  # Close victory

        elif any(opp.has_won() for opp in opponents):
            # Loss penalty scales with how badly we lost
            agent_finished = sum(1 for token in agent_player.tokens if token.position >= GameConstants.FINISH_POSITION)
            winner_finished = GameConstants.TOKENS_TO_WIN  # Winner has all tokens finished

            loss_margin = winner_finished - agent_finished
            base_loss_penalty = self.cfg.reward_cfg.lose

            if loss_margin >= 4:
                return base_loss_penalty * 2.0  # Shutout loss (0 tokens finished)
            elif loss_margin >= 3:
                return base_loss_penalty * 1.5  # Bad loss
            else:
                return base_loss_penalty  # Close loss

        return 0.0
