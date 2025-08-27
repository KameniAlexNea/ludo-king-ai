"""
Constants and configuration values for the Ludo game.
Centralized location for all game rules and board layout constants.
"""

from typing import Dict, Set


class GameConstants:
    """Core game constants and rules."""

    # Board dimensions
    MAIN_BOARD_SIZE = 52
    HOME_COLUMN_SIZE = 6
    TOKENS_PER_PLAYER = 4
    MAX_PLAYERS = 4

    # Dice
    DICE_MIN = 1
    DICE_MAX = 6
    EXIT_HOME_ROLL = 6

    # Win condition
    TOKENS_TO_WIN = 4

    # Special positions
    FINISH_POSITION = 105  # Final position in home column
    HOME_POSITION = -1  # Tokens start in home (-1)
    HOME_COLUMN_START = 100  # Start of home column positions

    # Normalization constants for RL environment
    DICE_NORMALIZATION_MEAN = 3.5  # (DICE_MIN + DICE_MAX) / 2
    HOME_COLUMN_DEPTH_SCALE = 5.0  # HOME_COLUMN_SIZE - 1
    POSITION_NORMALIZATION_FACTOR = 0.5  # For scaling positions to [0,1]
    TURN_INDEX_MAX_SCALE = 1.0  # Maximum normalized turn index
    BLOCKING_COUNT_NORMALIZATION = 6.0  # Maximum expected blocking positions

    # Opponent simulation
    MAX_OPPONENT_CHAIN_LENGTH = 20  # Safety cap for opponent turn chains


class BoardConstants:
    """Board layout and position constants."""

    # Star squares (safe for all players) - 0-indexed board
    STAR_SQUARES: Set[int] = {8, 21, 34, 47}

    # Starting positions for each color
    START_POSITIONS: Dict[str, int] = {
        "red": 1,
        "green": 14,
        "yellow": 27,
        "blue": 40,
    }

    # Last position before entering home column for each color
    HOME_COLUMN_ENTRIES: Dict[str, int] = {
        "red": 51,  # Red enters home after position 51
        "green": 12,  # Green enters home after position 12
        "yellow": 25,  # Yellow enters home after position 25
        "blue": 38,  # Blue enters home after position 38
    }

    # Starting positions are safe for everyone (all starting squares are safe)
    COLORED_SAFE_SQUARES: Dict[str, Set[int]] = {
        "red": {1},  # Only starting position (safe for everyone)
        "green": {14},  # Only starting position (safe for everyone)
        "yellow": {27},  # Only starting position (safe for everyone)
        "blue": {40},  # Only starting position (safe for everyone)
    }

    # All safe squares (combination of star squares and colored squares)
    @classmethod
    def get_all_safe_squares(cls) -> Set[int]:
        """Get all safe squares on the board."""
        all_safe = cls.STAR_SQUARES.copy()
        for color_squares in cls.COLORED_SAFE_SQUARES.values():
            all_safe.update(color_squares)
        return all_safe

    # Home column positions (100 to 105)
    HOME_COLUMN_START = 100
    HOME_COLUMN_END = 105
    FINISH_POSITION = 105  # Final position in home column

    @classmethod
    def is_home_column_position(cls, position: int) -> bool:
        """Check if a position is in any home column."""
        return cls.HOME_COLUMN_START <= position <= cls.HOME_COLUMN_END

    @classmethod
    def is_safe_position(cls, position: int, player_color: str = None) -> bool:
        """
        Check if a position is safe.

        Args:
            position: Board position to check
            player_color: Optional player color for color-specific safe squares

        Returns:
            bool: True if position is safe
        """
        # Home columns are always safe
        if cls.is_home_column_position(position):
            return True

        # Star squares are safe for everyone
        if position in cls.STAR_SQUARES:
            return True

        # Starting positions are safe for everyone (not just the owner)
        all_starting_positions = {1, 14, 27, 40}
        if position in all_starting_positions:
            return True

        return False


class StrategyConstants:
    """Constants for AI strategy calculations."""

    # Strategic values
    FINISH_TOKEN_VALUE = 100.0
    HOME_COLUMN_ADVANCE_VALUE = 20.0
    EXIT_HOME_VALUE = 15.0
    CAPTURE_BONUS = 25.0
    SAFE_MOVE_BONUS = 5.0
    RISKY_MOVE_PENALTY = 10.0

    # Threat levels
    HIGH_THREAT_THRESHOLD = 0.7
    MODERATE_THREAT_THRESHOLD = 0.4

    # Player progress thresholds
    SIGNIFICANTLY_BEHIND_THRESHOLD = 0.25
    SIGNIFICANTLY_AHEAD_THRESHOLD = 0.25


class Colors:
    """Player color constants."""

    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    BLUE = "blue"

    ALL_COLORS = [RED, GREEN, YELLOW, BLUE]

    @classmethod
    def is_valid_color(cls, color: str) -> bool:
        """Check if a color is valid."""
        return color in cls.ALL_COLORS


# Legacy constants for backward compatibility
SAFE_SQUARES = BoardConstants.STAR_SQUARES
START_POSITIONS = BoardConstants.START_POSITIONS
COLORED_SAFE_SQUARES = BoardConstants.COLORED_SAFE_SQUARES
