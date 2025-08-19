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
    FINISH_POSITION = 57
    HOME_POSITION = -1


class BoardConstants:
    """Board layout and position constants."""

    # Star squares (safe for all players)
    STAR_SQUARES: Set[int] = {1, 9, 14, 22, 27, 35, 40, 48}

    # Starting positions for each color
    START_POSITIONS: Dict[str, int] = {
        "red": 1,
        "green": 14,
        "yellow": 27,
        "blue": 40,
    }

    # Home column entry positions for each color
    HOME_COLUMN_ENTRIES: Dict[str, int] = {
        "red": 0,  # After position 51, enters home column
        "green": 13,
        "yellow": 26,
        "blue": 39,
    }

    # Colored safe squares for each player (including starting positions)
    COLORED_SAFE_SQUARES: Dict[str, Set[int]] = {
        "red": {1, 2, 3, 4, 5, 6},
        "green": {14, 15, 16, 17, 18, 19},
        "yellow": {27, 28, 29, 30, 31, 32},
        "blue": {40, 41, 42, 43, 44, 45},
    }

    # All safe squares (combination of star squares and colored squares)
    @classmethod
    def get_all_safe_squares(cls) -> Set[int]:
        """Get all safe squares on the board."""
        all_safe = cls.STAR_SQUARES.copy()
        for color_squares in cls.COLORED_SAFE_SQUARES.values():
            all_safe.update(color_squares)
        return all_safe

    # Home column positions (52-57 for each player, calculated dynamically)
    HOME_COLUMN_START = 52

    @classmethod
    def is_home_column_position(cls, position: int) -> bool:
        """Check if a position is in any home column."""
        return position >= cls.HOME_COLUMN_START

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

        # Color-specific safe squares
        if player_color and position in cls.COLORED_SAFE_SQUARES.get(
            player_color, set()
        ):
            return True

        return False


class StrategyConstants:
    """Constants for AI strategy calculations."""

    # Strategic values
    FINISH_TOKEN_VALUE = 100.0
    HOME_COLUMN_ADVANCE_VALUE = 20.0
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
