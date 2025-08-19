"""
Token representation for Ludo game.
Each player has 4 tokens that move around the board.
"""

from enum import Enum
from dataclasses import dataclass
from .constants import BoardConstants


class TokenState(Enum):
    """Possible states of a token."""

    HOME = "home"  # Token is in starting home area
    ACTIVE = "active"  # Token is on the main board path
    HOME_COLUMN = "home_column"  # Token is in the final home column
    FINISHED = "finished"  # Token has reached the center


@dataclass
class Token:
    """
    Represents a single token/piece in the Ludo game.
    """

    token_id: int  # 0, 1, 2, 3 for each player
    player_color: str
    state: TokenState = TokenState.HOME
    position: int = (
        -1
    )  # -1 means in home, 0-51 for board positions, 100-105 for home column

    def __post_init__(self):
        """Initialize token in home state."""
        if self.state == TokenState.HOME:
            self.position = -1

    def is_in_home(self) -> bool:
        """Check if token is still in home area."""
        return self.state == TokenState.HOME

    def is_active(self) -> bool:
        """Check if token is on the main board."""
        return self.state == TokenState.ACTIVE

    def is_in_home_column(self) -> bool:
        """Check if token is in the final home column."""
        return self.state == TokenState.HOME_COLUMN

    def is_finished(self) -> bool:
        """Check if token has reached the center."""
        return self.state == TokenState.FINISHED

    def can_move(self, dice_value: int, current_board_state) -> bool:
        """
        Check if this token can make a valid move with the given dice value.

        Args:
            dice_value: The value rolled on the dice (1-6)
            current_board_state: Current state of the board to check for blocks/captures

        Returns:
            bool: True if the token can move, False otherwise
        """
        if self.is_finished():
            return False

        if self.is_in_home():
            # Can only move out of home with a 6
            return dice_value == 6

        if self.is_in_home_column():
            # Must use exact count to move in home column
            target_position = self.position + dice_value
            return target_position <= 105  # Can't go beyond position 105 (finish)

        # Token is active on main board
        return True

    def get_target_position(self, dice_value: int, player_start_position: int) -> int:
        """
        Calculate the target position after moving with dice_value.

        Args:
            dice_value: The value rolled on the dice
            player_start_position: Starting position for this player's color

        Returns:
            int: Target position after the move
        """
        if self.is_in_home():
            if dice_value == 6:
                return player_start_position
            return -1  # Can't move

        if self.is_in_home_column():
            return self.position + dice_value

        # Active on main board - implement wraparound logic
        new_position = self.position + dice_value

        # Get the home entry position for this player
        home_entry_position = BoardConstants.HOME_COLUMN_ENTRIES[self.player_color]

        # Check if we should enter home column
        if self.position <= home_entry_position < new_position:
            # Enter home column at position 100 and advance
            overflow = new_position - home_entry_position - 1
            return 100 + overflow

        # Handle wraparound: after position 51, go to position 0 (except for red)
        if new_position > 51:
            if self.player_color == "red" and self.position <= 51:
                # Red enters home after position 51
                overflow = new_position - 52
                return 100 + overflow
            else:
                # All other colors wrap around to position 0
                return new_position - 52

        return new_position

    def move(self, dice_value: int, player_start_position: int) -> bool:
        """
        Move the token based on dice value.

        Args:
            dice_value: The value rolled on the dice
            player_start_position: Starting position for this player's color

        Returns:
            bool: True if move was successful, False otherwise
        """
        if not self.can_move(dice_value, None):  # Simplified check
            return False

        target_position = self.get_target_position(dice_value, player_start_position)

        if target_position == -1:
            return False

        # Update token state and position
        if self.is_in_home() and dice_value == 6:
            self.state = TokenState.ACTIVE
            self.position = player_start_position
        elif self.is_active():
            if 100 <= target_position <= 105:
                # Entering home column
                self.state = TokenState.HOME_COLUMN
                self.position = target_position
                if self.position == 105:  # Reached finish
                    self.state = TokenState.FINISHED
            else:
                # Normal move on main board
                self.position = target_position
        elif self.is_in_home_column():
            self.position = target_position
            if self.position == 105:  # Reached finish
                self.state = TokenState.FINISHED

        return True

    def to_dict(self) -> dict:
        """Convert token to dictionary for AI consumption."""
        return {
            "token_id": self.token_id,
            "player_color": self.player_color,
            "state": self.state.value,
            "position": self.position,
            "is_in_home": self.is_in_home(),
            "is_active": self.is_active(),
            "is_in_home_column": self.is_in_home_column(),
            "is_finished": self.is_finished(),
        }

    def __str__(self) -> str:
        """String representation of the token."""
        return f"Token({self.player_color}_{self.token_id}: {self.state.value} at {self.position})"
