"""
Handles the evaluation of the trained model.
"""

from typing import Dict, List, Tuple

from ..model.dqn_model import LudoDQNAgent
from .sequence_builder import SequenceBuilder


class Evaluator:
    """Evaluates the trained model."""

    def __init__(self, agent: LudoDQNAgent, game_data: List[Dict], encoder):
        self.agent = agent
        self.game_data = game_data
        self.encoder = encoder

    def evaluate_model(self, test_sequences: List[List[Tuple]] = None) -> Dict:
        """
        Evaluate the trained model with comprehensive metrics.

        Args:
            test_sequences: Test sequences (if None, uses 10% of training data)

        Returns:
            Dict: Evaluation metrics
        """
        if test_sequences is None:
            # Use last 10% of sequences for testing
            sequence_builder = SequenceBuilder(self.game_data, self.encoder)
            all_sequences = sequence_builder.create_training_sequences()
            test_size = max(1, len(all_sequences) // 10)
            test_sequences = all_sequences[-test_size:]

        self.agent.set_eval_mode()

        total_reward = 0.0
        total_steps = 0
        correct_actions = 0
        # strategic_value_errors = []
        # move_type_accuracy = {"exit_home": 0, "advance": 0, "capture": 0, "finish": 0}
        # move_type_counts = {"exit_home": 0, "advance": 0, "capture": 0, "finish": 0}

        for sequence in test_sequences:
            sequence_reward = 0.0
            for state, true_action, reward, _, _ in sequence:
                # Create dummy valid moves for evaluation
                dummy_moves = [{"token_id": i, "strategic_value": 0} for i in range(4)]
                predicted_action = self.agent.act(state, dummy_moves)

                if predicted_action == true_action:
                    correct_actions += 1

                sequence_reward += reward
                total_steps += 1

            total_reward += sequence_reward

        accuracy = correct_actions / total_steps if total_steps > 0 else 0
        avg_reward_per_sequence = (
            total_reward / len(test_sequences) if test_sequences else 0
        )
        avg_reward_per_step = total_reward / total_steps if total_steps > 0 else 0

        self.agent.set_train_mode()

        return {
            "accuracy": accuracy,
            "avg_reward_per_sequence": avg_reward_per_sequence,
            "avg_reward_per_step": avg_reward_per_step,
            "total_test_steps": total_steps,
            "num_test_sequences": len(test_sequences),
            "total_reward": total_reward,
        }
