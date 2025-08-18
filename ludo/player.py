"""
Player representation for Ludo game.
Each player has a color and controls 4 tokens.
"""

from enum import Enum
from typing import List, Dict
from .token import Token, TokenState


class PlayerColor(Enum):
    """Available player colors in Ludo."""
    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    YELLOW = "yellow"


class Player:
    """
    Represents a player in the Ludo game.
    """
    
    def __init__(self, color: PlayerColor, player_id: int):
        """
        Initialize a player with their color and 4 tokens.
        
        Args:
            color: Player's color (RED, BLUE, GREEN, YELLOW)
            player_id: Unique identifier for the player (0-3)
        """
        self.color = color
        self.player_id = player_id
        self.tokens: List[Token] = []
        
        # Create 4 tokens for this player
        for i in range(4):
            token = Token(
                token_id=i,
                player_color=color.value,
                state=TokenState.HOME
            )
            self.tokens.append(token)
        
        # Starting positions for each color on the board
        self.start_positions = {
            PlayerColor.RED: 1,
            PlayerColor.GREEN: 14,
            PlayerColor.YELLOW: 27,
            PlayerColor.BLUE: 40
        }
        
        self.start_position = self.start_positions[color]
    
    def get_movable_tokens(self, dice_value: int) -> List[Token]:
        """
        Get all tokens that can be moved with the given dice value.
        
        Args:
            dice_value: The value rolled on the dice (1-6)
            
        Returns:
            List[Token]: List of tokens that can make valid moves
        """
        movable_tokens = []
        
        for token in self.tokens:
            if token.can_move(dice_value, None):  # Simplified check for now
                movable_tokens.append(token)
        
        return movable_tokens
    
    def has_tokens_in_home(self) -> bool:
        """Check if player has any tokens still in home."""
        return any(token.is_in_home() for token in self.tokens)
    
    def has_active_tokens(self) -> bool:
        """Check if player has any tokens on the board."""
        return any(token.is_active() or token.is_in_home_column() for token in self.tokens)
    
    def get_finished_tokens_count(self) -> int:
        """Get the number of tokens that have reached the center."""
        return sum(1 for token in self.tokens if token.is_finished())
    
    def has_won(self) -> bool:
        """Check if player has won (all 4 tokens finished)."""
        return self.get_finished_tokens_count() == 4
    
    def can_move_any_token(self, dice_value: int) -> bool:
        """
        Check if player can move any token with the given dice value.
        
        Args:
            dice_value: The value rolled on the dice (1-6)
            
        Returns:
            bool: True if any token can be moved, False otherwise
        """
        return len(self.get_movable_tokens(dice_value)) > 0
    
    def move_token(self, token_id: int, dice_value: int) -> bool:
        """
        Move a specific token by token_id.
        
        Args:
            token_id: ID of the token to move (0-3)
            dice_value: The value rolled on the dice
            
        Returns:
            bool: True if move was successful, False otherwise
        """
        if token_id < 0 or token_id >= 4:
            return False
        
        token = self.tokens[token_id]
        return token.move(dice_value, self.start_position)
    
    def get_game_state(self) -> Dict:
        """
        Get the current game state for this player in a format suitable for AI.
        
        Returns:
            Dict: Player's current state including all token positions
        """
        return {
            'player_id': self.player_id,
            'color': self.color.value,
            'start_position': self.start_position,
            'tokens': [token.to_dict() for token in self.tokens],
            'tokens_in_home': sum(1 for token in self.tokens if token.is_in_home()),
            'active_tokens': sum(1 for token in self.tokens if token.is_active()),
            'tokens_in_home_column': sum(1 for token in self.tokens if token.is_in_home_column()),
            'finished_tokens': self.get_finished_tokens_count(),
            'has_won': self.has_won()
        }
    
    def get_possible_moves(self, dice_value: int) -> List[Dict]:
        """
        Get all possible moves for this player with the given dice value.
        This is particularly useful for AI decision making.
        
        Args:
            dice_value: The value rolled on the dice (1-6)
            
        Returns:
            List[Dict]: List of possible moves with details
        """
        possible_moves = []
        
        for token in self.tokens:
            if token.can_move(dice_value, None):
                target_position = token.get_target_position(dice_value, self.start_position)
                
                move_info = {
                    'token_id': token.token_id,
                    'current_position': token.position,
                    'current_state': token.state.value,
                    'target_position': target_position,
                    'move_type': self._get_move_type(token, dice_value),
                    'is_safe_move': self._is_safe_move(token, target_position),
                    'captures_opponent': False,  # Will be calculated by board
                    'strategic_value': self._calculate_strategic_value(token, dice_value)
                }
                
                possible_moves.append(move_info)
        
        return possible_moves
    
    def _get_move_type(self, token: Token, dice_value: int) -> str:
        """Determine the type of move being made."""
        if token.is_in_home() and dice_value == 6:
            return "exit_home"
        elif token.is_in_home_column():
            target = token.get_target_position(dice_value, self.start_position)
            if target == 57:
                return "finish"
            return "advance_home_column"
        else:
            return "advance_main_board"
    
    def _is_safe_move(self, token: Token, target_position: int) -> bool:
        """Check if the target position is a safe square."""
        # Safe squares are at positions: 1, 9, 14, 22, 27, 35, 40, 48 (star squares)
        # Plus each player's colored squares
        safe_squares = {1, 9, 14, 22, 27, 35, 40, 48}
        
        # Player's colored safe squares
        colored_safe = {
            PlayerColor.RED: {1, 2, 3, 4, 5, 6},
            PlayerColor.GREEN: {14, 15, 16, 17, 18, 19},
            PlayerColor.YELLOW: {27, 28, 29, 30, 31, 32},
            PlayerColor.BLUE: {40, 41, 42, 43, 44, 45}
        }
        
        if target_position in safe_squares:
            return True
        
        if target_position in colored_safe.get(self.color, set()):
            return True
        
        # Home column is always safe
        if target_position >= 52:
            return True
        
        return False
    
    def _calculate_strategic_value(self, token: Token, dice_value: int) -> float:
        """
        Calculate a strategic value for a move (simplified version).
        Higher values indicate better moves.
        """
        value = 0.0
        
        # Finishing a token is very valuable
        if token.is_in_home_column():
            target = token.get_target_position(dice_value, self.start_position)
            if target == 57:
                value += 100.0  # Finishing is highest priority
            else:
                value += 20.0   # Moving in home column is good
        
        # Exiting home is valuable when you roll a 6
        elif token.is_in_home() and dice_value == 6:
            value += 15.0
        
        # Moving active tokens forward
        elif token.is_active():
            value += dice_value  # Further movement is generally better
        
        return value
    
    def __str__(self) -> str:
        """String representation of the player."""
        return f"Player({self.color.value}, tokens: {[str(token) for token in self.tokens]})"
