"""
Board representation for Ludo game.
Manages the game board state and validates moves.
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from .player import Player, PlayerColor
from .token import Token


@dataclass
class Position:
    """Represents a position on the board."""

    index: int
    is_safe: bool = False
    is_star: bool = False
    color: Optional[str] = None

    def __post_init__(self):
        """Set safe and star properties based on position."""
        # Star squares (safe for all players)
        star_positions = {1, 9, 14, 22, 27, 35, 40, 48}
        if self.index in star_positions:
            self.is_star = True
            self.is_safe = True

        # Colored safe squares for each player
        colored_squares = {
            1: PlayerColor.RED.value,
            9: PlayerColor.RED.value,
            14: PlayerColor.GREEN.value,
            22: PlayerColor.GREEN.value,
            27: PlayerColor.YELLOW.value,
            35: PlayerColor.YELLOW.value,
            40: PlayerColor.BLUE.value,
            48: PlayerColor.BLUE.value,
        }

        if self.index in colored_squares:
            self.color = colored_squares[self.index]
            self.is_safe = True


class Board:
    """
    Represents the Ludo game board and manages token positions.
    """

    def __init__(self):
        """Initialize the board with 52 main positions plus home columns."""
        self.main_path_size = 52
        self.home_column_size = 6  # Positions 52-57 for each player

        # Initialize main board positions
        self.positions: Dict[int, Position] = {}
        for i in range(self.main_path_size):
            self.positions[i] = Position(i)

        # Track which tokens are at each position
        self.token_positions: Dict[int, List[Token]] = {}
        self.reset_token_positions()

        # Starting positions for each color
        self.start_positions = {
            PlayerColor.RED.value: 1,
            PlayerColor.GREEN.value: 14,
            PlayerColor.YELLOW.value: 27,
            PlayerColor.BLUE.value: 40,
        }

        # Home column entry positions for each color
        self.home_entries = {
            PlayerColor.RED.value: 0,  # After position 51, enters home column
            PlayerColor.GREEN.value: 13,
            PlayerColor.YELLOW.value: 26,
            PlayerColor.BLUE.value: 39,
        }

    def reset_token_positions(self):
        """Reset all token position tracking."""
        self.token_positions.clear()
        for i in range(self.main_path_size + self.home_column_size * 4):
            self.token_positions[i] = []

    def add_token(self, token: Token, position: int):
        """Add a token to a specific position on the board."""
        if position not in self.token_positions:
            self.token_positions[position] = []
        self.token_positions[position].append(token)

    def remove_token(self, token: Token, position: int):
        """Remove a token from a specific position on the board."""
        if position in self.token_positions and token in self.token_positions[position]:
            self.token_positions[position].remove(token)

    def get_tokens_at_position(self, position: int) -> List[Token]:
        """Get all tokens at a specific position."""
        return self.token_positions.get(position, [])

    def is_position_safe(self, position: int, player_color: str) -> bool:
        """Check if a position is safe for a given player color."""
        if position < 0 or position >= self.main_path_size:
            return True  # Home and home columns are safe

        board_position = self.positions.get(position)
        if not board_position:
            return True

        # Star squares are safe for everyone
        if board_position.is_star:
            return True

        # Colored squares are safe for matching color
        if board_position.color == player_color:
            return True

        return False

    def can_move_to_position(
        self, token: Token, target_position: int
    ) -> Tuple[bool, List[Token]]:
        """
        Check if a token can move to a target position.

        Returns:
            Tuple[bool, List[Token]]: (can_move, tokens_to_capture)
        """
        tokens_at_target = self.get_tokens_at_position(target_position)
        tokens_to_capture = []

        # No tokens at target position
        if not tokens_at_target:
            return True, []

        # Check if position is safe
        if self.is_position_safe(target_position, token.player_color):
            # Safe positions allow stacking with same color
            same_color_tokens = [
                t for t in tokens_at_target if t.player_color == token.player_color
            ]
            opponent_tokens = [
                t for t in tokens_at_target if t.player_color != token.player_color
            ]

            if opponent_tokens:
                # Can't land on opponent tokens in safe squares
                return False, []
            else:
                # Can stack with own tokens
                return True, []

        # Not a safe position
        opponent_tokens = [
            t for t in tokens_at_target if t.player_color != token.player_color
        ]
        same_color_tokens = [
            t for t in tokens_at_target if t.player_color == token.player_color
        ]

        if same_color_tokens:
            # Can stack with own tokens
            if opponent_tokens:
                # Can capture opponents when landing with own tokens
                tokens_to_capture = opponent_tokens
            return True, tokens_to_capture

        if opponent_tokens:
            # Can capture all opponent tokens at this position
            tokens_to_capture = opponent_tokens
            return True, tokens_to_capture

        return True, []

    def execute_move(
        self, token: Token, old_position: int, new_position: int
    ) -> List[Token]:
        """
        Execute a move on the board and return any captured tokens.

        Args:
            token: The token being moved
            old_position: Current position of the token
            new_position: Target position for the token

        Returns:
            List[Token]: List of captured tokens
        """
        captured_tokens = []

        # Remove token from old position
        if old_position >= 0:  # -1 means token was in home
            self.remove_token(token, old_position)

        # Check what happens at the new position
        can_move, tokens_to_capture = self.can_move_to_position(token, new_position)

        if not can_move:
            # Move is not valid, put token back
            if old_position >= 0:
                self.add_token(token, old_position)
            return []

        # Capture opponent tokens
        for captured_token in tokens_to_capture:
            self.remove_token(captured_token, new_position)
            from .token import TokenState

            captured_token.state = TokenState.HOME
            captured_token.position = -1
            captured_tokens.append(captured_token)

        # Place the moving token at new position
        self.add_token(token, new_position)

        return captured_tokens

    def get_board_state_for_ai(self, current_player: Player) -> Dict:
        """
        Get the current board state in a format suitable for AI analysis.

        Args:
            current_player: The player whose turn it is

        Returns:
            Dict: Complete board state information
        """
        board_state = {
            "current_player": current_player.color.value,
            "board_positions": {},
            "safe_positions": [],
            "star_positions": [],
            "player_start_positions": self.start_positions,
            "home_column_entries": self.home_entries,
        }

        # Map all token positions
        for position, tokens in self.token_positions.items():
            if tokens:  # Only include positions with tokens
                board_state["board_positions"][position] = [
                    {
                        "player_color": token.player_color,
                        "token_id": token.token_id,
                        "state": token.state.value,
                    }
                    for token in tokens
                ]

        # Add safe and star positions
        for pos_idx, position in self.positions.items():
            if position.is_safe:
                board_state["safe_positions"].append(pos_idx)
            if position.is_star:
                board_state["star_positions"].append(pos_idx)

        return board_state

    def get_position_info(self, position: int) -> Dict:
        """Get detailed information about a specific position."""
        if position < 0:
            return {"type": "home", "is_safe": True, "tokens": []}
        elif position < self.main_path_size:
            board_pos = self.positions.get(position, Position(position))
            return {
                "type": "main_board",
                "position": position,
                "is_safe": board_pos.is_safe,
                "is_star": board_pos.is_star,
                "color": board_pos.color,
                "tokens": [
                    token.to_dict() for token in self.get_tokens_at_position(position)
                ],
            }
        else:
            return {
                "type": "home_column",
                "position": position,
                "is_safe": True,
                "tokens": [
                    token.to_dict() for token in self.get_tokens_at_position(position)
                ],
            }

    def update_token_position(self, token: Token, old_position: int, new_position: int):
        """Update token position tracking on the board."""
        if old_position >= 0:
            self.remove_token(token, old_position)
        if new_position >= 0:
            self.add_token(token, new_position)

    def get_blocking_positions(self, player_color: str) -> Set[int]:
        """Get positions where this player is blocking opponents."""
        blocking_positions = set()

        for position, tokens in self.token_positions.items():
            if position < 0 or position >= self.main_path_size:
                continue  # Only check main board positions

            player_tokens = [t for t in tokens if t.player_color == player_color]
            if len(player_tokens) >= 2 and not self.is_position_safe(
                position, player_color
            ):
                # Multiple tokens of same player on non-safe square = blocking
                blocking_positions.add(position)

        return blocking_positions

    def __str__(self) -> str:
        """String representation of the board state."""
        result = "Board State:\n"
        for position in range(self.main_path_size):
            tokens = self.get_tokens_at_position(position)
            if tokens:
                result += f"Position {position}: {[str(token) for token in tokens]}\n"
        return result
