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

    # Home column positions (100 to 105)
    HOME_COLUMN_START = GameConstants.HOME_COLUMN_START
    HOME_COLUMN_END = GameConstants.FINISH_POSITION
    FINISH_POSITION = GameConstants.FINISH_POSITION

    # All safe squares (combination of star squares and colored squares)
    @classmethod
    def get_all_safe_squares(cls) -> Set[int]:
        """Get all safe squares on the board."""
        all_safe = cls.STAR_SQUARES.copy()
        for color_squares in cls.COLORED_SAFE_SQUARES.values():
            all_safe.update(color_squares)
        return all_safe

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
    KILLER_FUTURE_CAPTURE_WEIGHT = (
        3.0  # weight per potential future capture target in range after move
    )

    # Cautious strategy specific thresholds
    CAUTIOUS_MAX_ALLOWED_THREAT = 0  # normal mode: avoid any threatened landing
    CAUTIOUS_LATE_GAME_ALLOWED_THREAT = 1  # relax slightly when behind late game
    CAUTIOUS_MIN_ACTIVE_TOKENS = 2  # ensure some board presence

    # Optimist strategy weights
    OPTIMIST_HIGH_RISK_THRESHOLD = (
        10.0  # strategic value threshold to treat risky as high-value
    )
    OPTIMIST_RISK_REWARD_BONUS = 4.0
    OPTIMIST_CAPTURE_PROGRESS_WEIGHT = 1.2
    OPTIMIST_FUTURE_CAPTURE_WEIGHT = 2.0
    OPTIMIST_EXIT_EARLY_ACTIVE_TARGET = 3  # aim to have many tokens active
    OPTIMIST_STACK_BONUS = 2.5

    # Winner strategy weights
    WINNER_HOME_DEPTH_WEIGHT = 1.0
    WINNER_SAFE_CAPTURE_PROGRESS_WEIGHT = 1.0
    WINNER_EXIT_MIN_ACTIVE = (
        1  # ensure at least one token active before deprioritizing exits
    )

    # Defensive strategy weights (distinct from cautious: allows controlled presence & block formation)
    DEFENSIVE_MIN_ACTIVE_TOKENS = 2  # maintain at least this many tokens active
    DEFENSIVE_BLOCK_FORMATION_BONUS = (
        6.0  # reward forming or staying in a block (stack immunity)
    )
    DEFENSIVE_SAFE_CAPTURE_BONUS = 12.0  # only modest capture reward if safe
    DEFENSIVE_HOME_DEPTH_WEIGHT = 0.9  # slightly lower than winner; still important
    DEFENSIVE_EXIT_PRESSURE_THRESHOLD = (
        3  # if opponents have this many finished, increase urgency
    )
    DEFENSIVE_ALLOW_THREAT_DISTANCE = 2  # acceptable minimal incoming distance (tighter than cautious late-game relaxation)
    DEFENSIVE_MAX_THREAT_COUNT = (
        1  # tolerate at most one potential attacker when moving on main board
    )
    DEFENSIVE_REPOSITION_BONUS = (
        2.5  # small bonus for moving from threatened square to safer square
    )
    DEFENSIVE_AVOID_BREAKING_BLOCK_PENALTY = (
        5.0  # penalty if move breaks own protective block without benefit
    )

    # Balanced strategy adaptive weights (blends offensive/defensive heuristics)
    BALANCED_PROGRESS_WEIGHT = 1.1  # slight emphasis on steady advancement
    BALANCED_HOME_PRIORITY = (
        0.95  # weight for home column depth (between defensive and winner)
    )
    BALANCED_SAFE_CAPTURE_WEIGHT = (
        1.3  # reward safe capture a bit more than pure progress
    )
    BALANCED_RISK_TOLERANCE_MARGIN = 0.15  # dynamic shift toward aggression when behind
    BALANCED_MIN_ACTIVE_TARGET = 2  # maintain moderate board presence
    BALANCED_BLOCK_VALUE = 4.0  # value for maintaining/creating protective blocks
    BALANCED_FUTURE_CAPTURE_PROXIMITY = (
        5  # squares ahead scanned for future capture potential
    )
    BALANCED_FUTURE_CAPTURE_WEIGHT = 1.5
    BALANCED_THREAT_SOFT_CAP = (
        2  # tolerate up to 2 potential attackers when equal/behind
    )
    BALANCED_AHEAD_THREAT_CAP = 1  # stricter when ahead
    BALANCED_LATE_GAME_FINISH_PUSH = (
        2  # number of tokens finished by leader to trigger finish push
    )

    # Hybrid probabilistic strategy constants
    HYBRID_IMMEDIATE_RISK_WEIGHT = 0.55  # alpha for blending immediate vs horizon
    HYBRID_PROXIMITY_REF = 7
    HYBRID_PROXIMITY_PENALTY_CAP = 6.0
    HYBRID_CLUSTER_INCREMENT = 0.15
    HYBRID_IMPACT_BASE = 0.5
    HYBRID_IMPACT_PROGRESS_POWER = 1.2
    HYBRID_PROGRESS_POWER = 1.4
    HYBRID_PROGRESS_SCALE = 3.0
    HYBRID_HOME_DEPTH_FACTOR = 2.0
    HYBRID_CAPTURE_BASE = 2.0
    HYBRID_EXTRA_TURN_COEFF = 2.2
    HYBRID_EXTRA_TURN_PROGRESS_NORM = 3.5 / GameConstants.MAIN_BOARD_SIZE
    HYBRID_FUTURE_SAFETY_BONUS = 0.2
    HYBRID_SPREAD_ACTIVE_TARGET = 2
    HYBRID_SPREAD_BONUS = 0.5
    HYBRID_RISK_SUPPRESSION_COEFF = 0.7
    HYBRID_FINISH_BONUS = 4.2
    HYBRID_ADVANCE_HOME_BONUS = 1.9
    HYBRID_EXIT_HOME_BONUS = 1.1
    HYBRID_SAFE_LANDING_BONUS = 1.0
    HYBRID_COMPOSITE_RISK_POWER = 1.05
    HYBRID_LATE_GAME_PROGRESS_MULT = 1.15
    HYBRID_EARLY_GAME_PROGRESS_MULT = 0.9
    HYBRID_LEAD_FACTOR_STRONG = 0.8
    HYBRID_BEHIND_FACTOR_STRONG = -0.8

    # Weighted random strategy constants
    WEIGHTED_RANDOM_TEMP_EARLY = 1.4
    WEIGHTED_RANDOM_TEMP_MID = 1.0
    WEIGHTED_RANDOM_TEMP_LATE = 0.7
    WEIGHTED_RANDOM_PHASE_EARLY = 0.25
    WEIGHTED_RANDOM_PHASE_LATE = 0.65
    WEIGHTED_RANDOM_EPSILON = 0.05  # epsilon-uniform exploration
    WEIGHTED_RANDOM_DIVERSITY_LAMBDA = 0.15  # penalize recently used token ids
    WEIGHTED_RANDOM_CAPTURE_BONUS = 0.4
    WEIGHTED_RANDOM_SAFE_BONUS = 0.2
    WEIGHTED_RANDOM_RISK_THREAT_CAP = 3  # damp moves with many potential threats
    WEIGHTED_RANDOM_RISK_PENALTY = 0.5
    WEIGHTED_RANDOM_MIN_WEIGHT = 1e-4


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
