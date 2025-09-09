"""
Token representation for Ludo game.
Each player has 4 tokens that move around the board.
"""

from dataclasses import dataclass
from enum import Enum

from .constants import BoardConstants, GameConstants


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
            # Can only leave home with exact EXIT_HOME_ROLL
            return dice_value == GameConstants.EXIT_HOME_ROLL

        if self.is_in_home_column():
            # Must not overshoot finish
            target_position = self.position + dice_value
            return target_position <= GameConstants.FINISH_POSITION

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
            if dice_value == GameConstants.EXIT_HOME_ROLL:
                return player_start_position
            return GameConstants.HOME_POSITION  # Can't move

        if self.is_in_home_column():
            return self.position + dice_value

        # Active on main board - unified logic
        current = self.position
        new_position = current + dice_value
        home_entry = BoardConstants.HOME_COLUMN_ENTRIES[self.player_color]

        # board_last_index = GameConstants.MAIN_BOARD_SIZE - 1

        # Normalize potential wrap for crossing beyond last board index
        # We need to detect crossing the home_entry square moving forward (circular path)

        def crosses_entry(start: int, end: int, entry: int) -> bool:
            """Return True if movement from start->end (forward with wrap) crosses the entry square strictly after leaving it.

            Crossing means entry is in the open interval (start, end] along forward movement path of length dice_value.
            """
            if start <= end:  # no wrap
                return start < entry <= end
            # wrapped segment: (start..board_last] U [0..end]
            return entry > start or entry <= end

        # Compute wrapped main-board landing before considering home column
        wrapped_main_pos = new_position % GameConstants.MAIN_BOARD_SIZE

        if crosses_entry(current, wrapped_main_pos, home_entry):
            # Steps taken after reaching entry square determine home column depth
            # Use iterative approach for clarity and to avoid off-by-one errors in complex formulas
            steps_after_entry = 0
            steps_taken = 0
            pos = current
            while steps_taken < dice_value:
                pos = (pos + 1) % GameConstants.MAIN_BOARD_SIZE
                steps_taken += 1
                if pos == home_entry:
                    # remaining steps go into home column
                    steps_after_entry = dice_value - steps_taken
                    break

            target_home_index = BoardConstants.HOME_COLUMN_START + steps_after_entry
            # Cannot exceed finish
            if target_home_index > GameConstants.FINISH_POSITION:
                return GameConstants.HOME_POSITION  # invalid move (overshoot)
            return target_home_index

        # Not entering home column: land on wrapped main-board position
        return wrapped_main_pos

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

        if target_position == GameConstants.HOME_POSITION:
            return False
        # Commit the already validated target movement
        self.commit_move(target_position, player_start_position)

        return True

    def commit_move(self, target_position: int, player_start_position: int):
        """Commit a precomputed, validated movement to target_position.

        This applies state transitions without recomputing legality. Intended
        for use by higher-level game logic after board validation to prevent
        duplicate target recomputation.
        """
        # Leaving home
        if self.is_in_home():
            # Only valid if target equals the player's start position
            if target_position == player_start_position:
                self.state = TokenState.ACTIVE
                self.position = target_position
            return

        # From active path
        if self.is_active():
            if BoardConstants.is_home_column_position(target_position):
                self.position = target_position
                self.state = (
                    TokenState.FINISHED
                    if target_position == GameConstants.FINISH_POSITION
                    else TokenState.HOME_COLUMN
                )
            else:
                self.position = target_position
            return

        # From home column
        if self.is_in_home_column():
            self.position = target_position
            if target_position == GameConstants.FINISH_POSITION:
                self.state = TokenState.FINISHED

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
