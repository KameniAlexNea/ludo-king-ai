"""
Handles the reward calculation for the Ludo RL agent.
"""

from typing import Dict

from ..config import REWARDS


class RewardCalculator:
    """Calculates the reward for a given game state and action."""

    def calculate_reward(
        self, game_data: Dict, game_index: int, next_game_data: Dict = None
    ) -> float:
        """
        Calculate reward with better engineering and game context awareness.

        Args:
            game_data: Current game state data
            game_index: Index of current game in sequence
            next_game_data: Next game state data (if available)

        Returns:
            float: Calculated reward
        """
        outcome = game_data.get("outcome", {})
        context = game_data["game_context"]
        player_state = context["player_state"]
        current_situation = context["current_situation"]
        valid_moves = context.get("valid_moves", [])
        strategic_analysis = context.get("strategic_analysis", {})

        reward = 0.0

        # Basic success/failure with higher penalties for invalid moves
        if outcome.get("success", False):
            reward += REWARDS.SUCCESS
        else:
            reward += REWARDS.FAILS  # Stronger penalty for invalid moves

        # Strategic rewards with better scaling
        captured_tokens = outcome.get("captured_tokens", [])
        if captured_tokens:
            reward += REWARDS.CAPTURE * len(captured_tokens)

        if outcome.get("token_finished", False):
            reward += REWARDS.TOKEN_FINISHED

        if outcome.get("extra_turn", False):
            reward += REWARDS.EXTRA_TURN

        # Progress-based rewards with better scaling
        old_pos = outcome.get("old_position", -1)
        new_pos = outcome.get("new_position", -1)
        if outcome.get("success", False) and old_pos != -1 and new_pos != old_pos:
            if new_pos > old_pos:
                progress = (new_pos - old_pos) / 52.0
                reward += progress * REWARDS.PROGRESS_WEIGHT
            elif old_pos > 40 and new_pos < 10:  # Wrapped around the board
                progress = (52 - old_pos + new_pos) / 52.0
                reward += progress * REWARDS.PROGRESS_WEIGHT

        # Strategic value integration
        chosen_move_idx = game_data.get("chosen_move", 0)
        if chosen_move_idx < len(valid_moves):
            chosen_move = valid_moves[chosen_move_idx]
            strategic_value = chosen_move.get("strategic_value", 0)
            reward += strategic_value * REWARDS.STRATEGIC_WEIGHT

            # Safety and risk assessment
            if chosen_move.get("is_safe_move", True):
                reward += REWARDS.SAFETY_BONUS
            else:
                reward += REWARDS.RISK_PENALTY

            # Best move bonus
            best_move = strategic_analysis.get("best_strategic_move", {})
            if best_move and chosen_move.get("token_id") == best_move.get("token_id"):
                reward += REWARDS.BEST_MOVE

        # Game phase awareness
        active_tokens = player_state.get("active_tokens", 0)
        finished_tokens = player_state.get("finished_tokens", 0)
        tokens_in_home = player_state.get("tokens_in_home", 0)

        # First token out bonus
        if old_pos == -1 and new_pos >= 0 and active_tokens == 1:
            reward += REWARDS.FIRST_TOKEN_OUT

        # Game phase bonuses
        total_progress = active_tokens + finished_tokens
        if total_progress <= 1:  # Early game
            reward += REWARDS.EARLY_GAME_BONUS
        elif finished_tokens >= 2:  # End game
            reward += REWARDS.END_GAME_BONUS

        # Relative positioning reward
        opponents = context.get("opponents", [])
        if opponents:
            # Calculate relative advantage
            max_opp_finished = max(
                (opp.get("tokens_finished", 0) for opp in opponents), default=0
            )
            relative_advantage = finished_tokens - max_opp_finished
            reward += relative_advantage * REWARDS.RELATIVE_POSITION_WEIGHT

        # Game winning bonus
        if player_state.get("has_won", False):
            reward += REWARDS.WON

        # Turn momentum bonus
        consecutive_sixes = current_situation.get("consecutive_sixes", 0)
        if consecutive_sixes > 0:
            reward += min(consecutive_sixes, 2) * REWARDS.EXTRA_TURN * 0.5

        # Penalty for stalling in home when possible to move out
        if (
            current_situation.get("dice_value", 1) == 6
            and tokens_in_home > 0
            and chosen_move_idx < len(valid_moves)
            and valid_moves[chosen_move_idx].get("move_type") != "exit_home"
        ):
            reward -= 1.0  # Penalty for not using 6 to exit home

        return reward
