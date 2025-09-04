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
    HOME_COLUMN_START = GameConstants.HOME_COLUMN_START
    HOME_COLUMN_END = GameConstants.FINISH_POSITION
    FINISH_POSITION = GameConstants.FINISH_POSITION

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

        # Starting positions (from START_POSITIONS) are safe for everyone
        if position in cls.START_POSITIONS.values():
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

    # Weights for enhanced heuristic components
    FORWARD_PROGRESS_WEIGHT = 1.0
    ACCELERATION_WEIGHT = 0.1  # bonus per reduced remaining step (heuristic)
    SAFETY_BONUS = SAFE_MOVE_BONUS  # reuse base safe bonus, alias for clarity
    VULNERABILITY_PENALTY_WEIGHT = 8.0  # penalty if landing square likely capturable
    HOME_COLUMN_DEPTH_MULTIPLIER = 1.0  # scales depth-based home column value

    # Killer strategy specific weights
    KILLER_PROGRESS_WEIGHT = 2.0  # value of removing a progressed enemy token
    KILLER_THREAT_WEIGHT = 1.5  # weight for targeting leading opponent
    KILLER_CHAIN_BONUS = 10.0  # extra turn follow-up potential bonus
    KILLER_SAFE_LAND_BONUS = 4.0  # landing safely after capture
    KILLER_BLOCK_BONUS = 6.0  # forming/keeping a two-token block after move
    KILLER_RECAPTURE_PENALTY = 12.0  # risk if easily recaptured
    KILLER_WEAK_PREY_PENALTY = 5.0  # skip low-progress prey if risky
    KILLER_FUTURE_CAPTURE_WEIGHT = 3.0  # weight per potential future capture target in range after move

    # Cautious strategy specific thresholds
    CAUTIOUS_MAX_ALLOWED_THREAT = 0  # normal mode: avoid any threatened landing
    CAUTIOUS_LATE_GAME_ALLOWED_THREAT = 1  # relax slightly when behind late game
    CAUTIOUS_MIN_ACTIVE_TOKENS = 2  # ensure some board presence


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
