"""
Improved RL Player for Ludo using the enhanced DQN agent.
"""

from typing import Dict

from .config import TRAINING_CONFIG
from .model.dqn_model import LudoDQNAgent
from .state_encoder import LudoStateEncoder


class RLPlayer:
    """Enhanced RL Player that uses the improved DQN agent for move selection."""

    def __init__(self, model_path: str = None, name: str = "RLPlayer"):
        """
        Initialize the improved RL player.

        Args:
            model_path: Path to trained model file
            name: Player name
        """
        self.name = name
        self.encoder = LudoStateEncoder()

        # Initialize with improved agent
        self.agent = LudoDQNAgent(
            state_dim=self.encoder.state_dim,
            max_actions=4,
            lr=TRAINING_CONFIG.LEARNING_RATE,
            gamma=TRAINING_CONFIG.GAMMA,
            epsilon=0.0,  # No exploration during gameplay
            use_prioritized_replay=False,  # Not needed for inference
            use_double_dqn=TRAINING_CONFIG.USE_DOUBLE_DQN,
        )

        # Load model if provided
        if model_path:
            self.load_model(model_path)

        self.agent.set_eval_mode()

    def choose_move(self, game_state: Dict) -> int:
        """
        Choose the best move given the current game state.

        Args:
            game_state: Current game state with all necessary information

        Returns:
            int: Index of chosen move in valid_moves list
        """
        try:
            # Extract valid moves
            valid_moves = game_state.get("valid_moves", [])
            if not valid_moves:
                return 0

            # Encode the game state
            state = self.encoder.encode_state(game_state)

            # Get action from agent (no exploration)
            action_idx = self.agent.act(state, valid_moves)

            # Ensure action is valid
            if 0 <= action_idx < len(valid_moves):
                return action_idx
            else:
                return 0  # Fallback to first move

        except Exception as e:
            print(f"Error in RL player move selection: {e}")
            # Fallback to first valid move
            return 0

    def choose_move_with_analysis(self, game_state: Dict) -> Dict:
        """
        Choose move with detailed analysis and confidence scores.

        Args:
            game_state: Current game state

        Returns:
            Dict: Move choice with analysis
        """
        try:
            valid_moves = game_state.get("valid_moves", [])
            if not valid_moves:
                return {
                    "move_index": 0,
                    "confidence": 0.0,
                    "analysis": "No valid moves",
                }

            # Encode state
            state = self.encoder.encode_state(game_state)

            # Get Q-values for analysis
            state_tensor = (
                self.agent.q_network(
                    self.agent.q_network.network[0](
                        self.agent.device.type == "cuda" and state.cuda()
                        if hasattr(state, "cuda")
                        else state
                    )
                )
                .detach()
                .cpu()
                .numpy()
            )

            # Analyze each move
            move_analyses = []
            for i, move in enumerate(valid_moves):
                token_id = move.get("token_id", 0)
                q_value = (
                    state_tensor[min(token_id, 3)]
                    if len(state_tensor) > token_id
                    else 0.0
                )

                # Combine with strategic information
                strategic_value = move.get("strategic_value", 0)
                safety_score = 1.0 if move.get("is_safe_move", True) else 0.3
                capture_bonus = 0.5 if move.get("captures_opponent", False) else 0.0

                total_score = q_value + strategic_value * 0.1 + capture_bonus

                move_analyses.append(
                    {
                        "move_index": i,
                        "token_id": token_id,
                        "q_value": float(q_value),
                        "strategic_value": strategic_value,
                        "safety_score": safety_score,
                        "total_score": float(total_score),
                        "move_type": move.get("move_type", "unknown"),
                    }
                )

            # Choose best move
            best_move = max(move_analyses, key=lambda x: x["total_score"])
            best_idx = best_move["move_index"]

            # Calculate confidence (difference from second best)
            sorted_moves = sorted(
                move_analyses, key=lambda x: x["total_score"], reverse=True
            )
            if len(sorted_moves) > 1:
                confidence = (
                    sorted_moves[0]["total_score"] - sorted_moves[1]["total_score"]
                )
                confidence = min(1.0, max(0.0, confidence / 5.0))  # Normalize to 0-1
            else:
                confidence = 1.0

            return {
                "move_index": best_idx,
                "confidence": confidence,
                "analysis": {
                    "chosen_move": best_move,
                    "all_moves": move_analyses,
                    "reasoning": self._generate_reasoning(best_move, game_state),
                },
            }

        except Exception as e:
            print(f"Error in detailed move analysis: {e}")
            return {"move_index": 0, "confidence": 0.0, "analysis": f"Error: {str(e)}"}

    def _generate_reasoning(self, chosen_move: Dict, game_state: Dict) -> str:
        """Generate human-readable reasoning for the move choice."""
        reasoning_parts = []

        move_type = chosen_move.get("move_type", "unknown")
        q_value = chosen_move.get("q_value", 0)
        strategic_value = chosen_move.get("strategic_value", 0)

        # Base reasoning
        if move_type == "exit_home":
            reasoning_parts.append("Getting token out of home")
        elif move_type == "finish":
            reasoning_parts.append("Finishing token to win")
        elif move_type == "capture":
            reasoning_parts.append("Capturing opponent token")
        else:
            reasoning_parts.append("Advancing token")

        # Add confidence indicators
        if q_value > 1.0:
            reasoning_parts.append("(high confidence)")
        elif q_value > 0.5:
            reasoning_parts.append("(moderate confidence)")
        else:
            reasoning_parts.append("(low confidence)")

        # Add strategic reasoning
        if strategic_value > 10:
            reasoning_parts.append("with high strategic value")
        elif strategic_value > 5:
            reasoning_parts.append("with good strategic value")

        return " ".join(reasoning_parts)

    def load_model(self, model_path: str):
        """Load a trained model."""
        try:
            self.agent.load_model(model_path)
            self.agent.set_eval_mode()
            print(f"Loaded RL model from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")

    def get_player_info(self) -> Dict:
        """Get information about the RL player."""
        return {
            "name": self.name,
            "type": "RL_Agent",
            "model_info": {
                "state_dim": self.encoder.state_dim,
                "epsilon": self.agent.epsilon,
                "training_step": self.agent.training_step,
            },
        }

    def reset_game(self):
        """Reset player state for a new game."""
        # RL player doesn't need game-specific state reset
        pass

    def update_strategy(self, game_result: Dict):
        """Update strategy based on game result (not used in inference mode)."""
        pass


class LudoRLStrategy:
    """
    Strategy wrapper for the RL player to integrate with the existing strategy system.
    This follows the same pattern as other strategies in the codebase.
    """

    def __init__(self, model_path: str = None, name: str = "ImprovedRL-DQN"):
        """
        Initialize the RL strategy.

        Args:
            model_path: Path to trained model file
            name: Name for the strategy
        """
        self.name = name
        self.description = (
            "Improved Deep Q-Network RL agent with Dueling DQN and prioritized replay"
        )
        self.rl_player = RLPlayer(model_path, name)

    def decide(self, game_context: Dict) -> int:
        """
        Make a decision based on the game context.

        Args:
            game_context: Game context from get_ai_decision_context()

        Returns:
            int: Token ID to move
        """
        # Convert game context to the format expected by the improved player
        game_data = {"game_context": game_context, "chosen_move": 0}

        # Get move index from improved player
        move_index = self.rl_player.choose_move(game_data)

        # Convert move index to token ID
        valid_moves = game_context.get("valid_moves", [])
        if valid_moves and 0 <= move_index < len(valid_moves):
            return valid_moves[move_index]["token_id"]
        elif valid_moves:
            return valid_moves[0]["token_id"]
        else:
            return 0

    def __str__(self) -> str:
        return f"Strategy(name={self.name}, description={self.description})"

    def __repr__(self) -> str:
        return self.__str__()


def create_rl_strategy(
    model_path: str = None, name: str = "ImprovedRL-DQN"
) -> LudoRLStrategy:
    """
    Factory function to create an improved RL strategy.

    Args:
        model_path: Path to trained model file
        name: Name for the strategy

    Returns:
        LudoRLStrategy: Configured improved RL strategy
    """
    return LudoRLStrategy(model_path, name)
