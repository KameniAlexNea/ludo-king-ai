"""
RL Player implementation that integrates with the existing Ludo game system.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import os

from .state_encoder import LudoStateEncoder
from .dqn_model import LudoDQNAgent


class LudoRLPlayer:
    """RL-based Ludo player that integrates with the existing strategy system."""
    
    def __init__(self, model_path: str = None, name: str = "RL-DQN"):
        """
        Initialize the RL player.
        
        Args:
            model_path: Path to trained model file
            name: Name for the RL player strategy
        """
        self.name = name
        self.description = f"Deep Q-Network RL agent trained on Ludo game data"
        
        self.encoder = LudoStateEncoder()
        self.agent = LudoDQNAgent(self.encoder.state_dim)
        
        if model_path and os.path.exists(model_path):
            self.agent.load_model(model_path)
            self.agent.set_eval_mode()  # Disable exploration for gameplay
            print(f"Loaded RL model from {model_path}")
        else:
            print(f"Warning: No model loaded. Using untrained agent.")
            self.agent.set_eval_mode()
    
    def choose_move(self, game_context: Dict) -> int:
        """
        Choose a move based on the game context.
        
        This method is called by the game system and should return a token ID.
        
        Args:
            game_context: Game context from get_ai_decision_context()
            
        Returns:
            int: Token ID to move (0-3)
        """
        valid_moves = game_context.get('valid_moves', [])
        
        if not valid_moves:
            return 0  # No valid moves available
        
        # Convert game context to the format expected by state encoder
        game_data = {
            'game_context': game_context,
            'chosen_move': 0  # placeholder
        }
        
        try:
            # Encode the current state
            state = self.encoder.encode_state(game_data)
            
            # Get action from the RL agent
            action = self.agent.act(state, valid_moves)
            
            # Ensure the action corresponds to a valid move
            valid_token_ids = [move['token_id'] for move in valid_moves]
            
            if action in valid_token_ids:
                return action
            else:
                # Fallback: return the token ID of the first valid move
                return valid_token_ids[0]
                
        except Exception as e:
            print(f"Error in RL player decision making: {e}")
            # Fallback: return first valid token
            return valid_moves[0]['token_id']
    
    def get_strategy_name(self) -> str:
        """Get the strategy name for logging and identification."""
        return self.name
    
    def get_strategy_description(self) -> str:
        """Get the strategy description."""
        return self.description


class LudoRLStrategy:
    """
    Strategy wrapper for the RL player to integrate with the existing strategy system.
    This follows the same pattern as other strategies in the codebase.
    """
    
    def __init__(self, model_path: str = None, name: str = "RL-DQN"):
        """
        Initialize the RL strategy.
        
        Args:
            model_path: Path to trained model file
            name: Name for the strategy
        """
        self.name = name
        self.description = f"Deep Q-Network RL agent trained on Ludo game data"
        self.rl_player = LudoRLPlayer(model_path, name)
    
    def make_decision(self, game_context: Dict) -> int:
        """
        Make a decision based on the game context.
        
        Args:
            game_context: Game context from get_ai_decision_context()
            
        Returns:
            int: Token ID to move
        """
        return self.rl_player.choose_move(game_context)
    
    def __str__(self) -> str:
        return f"Strategy(name={self.name}, description={self.description})"
    
    def __repr__(self) -> str:
        return self.__str__()


def create_rl_strategy(model_path: str = None, name: str = "RL-DQN") -> LudoRLStrategy:
    """
    Factory function to create an RL strategy.
    
    Args:
        model_path: Path to trained model file
        name: Name for the strategy
        
    Returns:
        LudoRLStrategy: Configured RL strategy
    """
    return LudoRLStrategy(model_path, name)