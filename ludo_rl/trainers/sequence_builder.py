"""
Builds training sequences from game data.
"""

from typing import Dict, List, Tuple

import numpy as np
from loguru import logger

from ..states import LudoStateEncoder
from .reward_calculator import RewardCalculator


class SequenceBuilder:
    """Builds training sequences from game data."""

    def __init__(self, game_data: List[Dict], encoder: LudoStateEncoder):
        self.game_data = game_data
        self.encoder = encoder
        self.reward_calculator = RewardCalculator()

    def create_training_sequences(self) -> List[List[Tuple]]:
        """
        Convert game data to training sequences with precise game boundary detection using exp_id.

        Uses exp_id (filename) to reliably identify game boundaries, which is much more accurate
        than timestamp or turn-based heuristics since all records from the same game file
        belong to the same game session.

        Returns:
            List[List[Tuple]]: List of game sequences, each containing (s, a, r, s', done) tuples
        """
        sequences = []
        current_sequence = []
        last_exp_id = None

        for i, game_data in enumerate(self.game_data):
            context = game_data["game_context"]
            current_player = context["current_situation"]["player_color"]
            current_exp_id = game_data.get("exp_id", "")

            # Game boundary detection using exp_id
            is_new_game = False
            if last_exp_id is not None and current_exp_id != last_exp_id:
                is_new_game = True

            if is_new_game and current_sequence:
                # Add sequence completion rewards
                self._add_sequence_completion_rewards(current_sequence)
                sequences.append(current_sequence)
                current_sequence = []

            # Encode current state
            state = self.encoder.encode_state(game_data)

            # Get action (use move index instead of token ID)
            action_idx = game_data.get("chosen_move", 0)
            valid_moves = context.get("valid_moves", [])
            if action_idx >= len(valid_moves):
                action_idx = 0

            # Calculate reward
            reward = self.reward_calculator.calculate_reward(game_data, i)

            # Determine next state and done flag
            next_state, done = self._get_next_state_and_done(i, current_player)

            current_sequence.append((state, action_idx, reward, next_state, done))

            # Update tracking variables
            last_exp_id = current_exp_id

        # Add the final sequence
        if current_sequence:
            self._add_sequence_completion_rewards(current_sequence)
            sequences.append(current_sequence)

        logger.info(f"Created {len(sequences)} training sequences")
        avg_length = np.mean([len(seq) for seq in sequences]) if sequences else 0
        logger.info(f"Average sequence length: {avg_length:.1f}")

        return sequences

    def _add_sequence_completion_rewards(self, sequence: List[Tuple]):
        """Add completion rewards to the sequence based on final outcome."""
        if not sequence:
            return

        # Get final state info from the last experience
        final_state, final_action, final_reward, _, _ = sequence[-1]

        # Add modest completion bonus/penalty
        completion_bonus = 0.0
        if final_reward > 50:  # Likely won the game
            completion_bonus = 10.0
        elif final_reward < -10:  # Poor performance
            completion_bonus = -5.0

        # Apply completion bonus to last few moves
        for i in range(max(0, len(sequence) - 3), len(sequence)):
            state, action, reward, next_state, done = sequence[i]
            sequence[i] = (
                state,
                action,
                reward + completion_bonus * 0.1,
                next_state,
                done,
            )

    def _get_next_state_and_done(
        self, current_index: int, current_player: str
    ) -> Tuple[np.ndarray, bool]:
        """Get next state and done flag with logic using exp_id."""
        current_exp_id = self.game_data[current_index].get("exp_id", "")

        # Look ahead to find next state for same player in same game
        for j in range(current_index + 1, min(current_index + 20, len(self.game_data))):
            next_data = self.game_data[j]
            next_exp_id = next_data.get("exp_id", "")

            # If we've moved to a different game, episode is done
            if next_exp_id != current_exp_id:
                break

            next_context = next_data["game_context"]
            next_player = next_context["current_situation"]["player_color"]

            if next_player == current_player:
                # Found next state for same player in same game
                next_state = self.encoder.encode_state(next_data)
                return next_state, False

        # No next state found for same player in same game - episode done
        current_state = self.encoder.encode_state(self.game_data[current_index])
        return current_state, True
