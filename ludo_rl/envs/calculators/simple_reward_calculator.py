"""Simple deterministic reward calculation utilities for LudoGymEnv.

This module provides a minimal reward function that lets the agent learn
strategic play through experience, rather than hard-coding policy decisions.
"""

from typing import Dict, List, Optional

from ludo.constants import BoardConstants, GameConstants
from ludo.game import LudoGame
from ludo.player import Player
from ludo.token import Token

from ..model import SimpleRewardConstants as RewardConstants
from ..model import EnvConfig



class SimpleRewardCalculator:
    """Minimal reward calculator that lets the agent learn strategy through experience."""

    def __init__(self, cfg: EnvConfig, game: LudoGame, agent_color: str):
        self.cfg = cfg
        self.game = game
        self.agent_color = agent_color

    def _get_token_progress(self, token: Token, position: Optional[int] = None) -> float:
        """Calculate a token's normalized progress towards its finish line (0.0 to 1.0).

        Args:
            token: The token to calculate progress for
            position: Optional position to calculate progress at (defaults to token's current position)
        """
        pos = position if position is not None else token.position

        if pos >= GameConstants.FINISH_POSITION:
            return 1.0
        if pos < 0:
            return 0.0

        # Total number of spaces a token must travel from home to finish
        total_path_length = GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE

        # Calculate steps from start to current position
        player_start_pos = BoardConstants.START_POSITIONS.get(token.player_color)
        if player_start_pos is None:
            return 0.0

        current_steps = 0
        if pos < GameConstants.HOME_COLUMN_START:
            # On main board
            if pos >= player_start_pos:
                current_steps = pos - player_start_pos
            else:
                # Wrap around the board
                current_steps = (GameConstants.MAIN_BOARD_SIZE - player_start_pos) + pos
        else:
            # In home column
            current_steps = (GameConstants.MAIN_BOARD_SIZE + (pos - GameConstants.HOME_COLUMN_START))

        return min(current_steps / total_path_length, 1.0)

    def _compute_movement_reward(self, move_res: Dict) -> float:
        """Unified movement reward combining progress and safety incentives."""
        token: Optional[Token] = move_res.get("token")
        if token is None or not isinstance(token, Token):
            return 0.0

        target_pos = move_res.get("target_position")
        if target_pos is None:
            return 0.0

        # Calculate progress increase
        initial_progress = self._get_token_progress(token)
        final_progress = self._get_token_progress(token, target_pos)
        progress_delta = final_progress - initial_progress

        if progress_delta <= 0:
            return 0.0

        # Base progress reward
        reward = progress_delta * RewardConstants.TOKEN_PROGRESS_REWARD

        # Bonus for home column progress
        if BoardConstants.is_home_column_position(target_pos):
            reward *= RewardConstants.HOME_COLUMN_MULTIPLIER

        # Bonus for safe positions
        if BoardConstants.is_safe_position(target_pos):
            reward += RewardConstants.SAFE_POSITION_BONUS

        return reward

    def _compute_capture_reward(self, move_res: Dict) -> float:
        """Simple capture reward - let agent learn capture value."""
        captured_tokens = move_res.get("captured_tokens", [])
        return len(captured_tokens) * RewardConstants.CAPTURE_REWARD

    def _compute_got_captured_penalty(self, move_res: Dict, token_positions_before: Optional[List[int]] = None) -> float:
        """Simple capture penalty."""
        if token_positions_before is not None:
            # Count tokens that were captured
            current_positions = [token.position for token in self.game.players[0].tokens]  # Simplified
            captured_count = sum(1 for before, current in zip(token_positions_before, current_positions)
                               if before >= 0 and current < 0)
            return captured_count * RewardConstants.GOT_CAPTURED_PENALTY
        else:
            # Fallback
            return RewardConstants.GOT_CAPTURED_PENALTY if move_res.get("was_captured") else 0.0

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
        """Simple reward calculation - let the agent learn strategy."""
        total_reward = 0.0

        # Core rewards
        capture_reward = self._compute_capture_reward(move_res)
        total_reward += capture_reward
        reward_components.append(capture_reward)

        got_captured_penalty = self._compute_got_captured_penalty(move_res, token_positions_before)
        total_reward += got_captured_penalty
        reward_components.append(got_captured_penalty)

        # Unified movement reward (progress + safety)
        movement_reward = self._compute_movement_reward(move_res)
        total_reward += movement_reward
        reward_components.append(movement_reward)

        # Token finished bonus (simple)
        if move_res.get("token_finished"):
            total_reward += RewardConstants.TOKEN_PROGRESS_REWARD * 10  # Small bonus
            reward_components.append(RewardConstants.TOKEN_PROGRESS_REWARD * 10)

        # Extra turn bonus
        if extra_turn:
            total_reward += RewardConstants.EXTRA_TURN_BONUS
            reward_components.append(RewardConstants.EXTRA_TURN_BONUS)

        # Diversity bonus (keep as is)
        if diversity_bonus:
            diversity_reward = self.cfg.reward_cfg.diversity_bonus
            total_reward += diversity_reward
            reward_components.append(diversity_reward)

        # Illegal action penalty
        if illegal_action:
            total_reward += self.cfg.reward_cfg.illegal_action
            reward_components.append(self.cfg.reward_cfg.illegal_action)

        return total_reward

    def get_terminal_reward(self, agent_player: Player, opponents: list[Player]) -> float:
        """Simple terminal rewards."""
        if agent_player.has_won():
            return RewardConstants.WIN_REWARD
        elif any(opp.has_won() for opp in opponents):
            return RewardConstants.LOSS_PENALTY
        return 0.0
