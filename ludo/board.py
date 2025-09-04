"""
Board representation for Ludo game.
Manages the game board state and validates moves.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .constants import BoardConstants
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
        if self.index in BoardConstants.STAR_SQUARES:
            self.is_star = True
            self.is_safe = True

        # Colored safe squares for each player
        for color, safe_positions in BoardConstants.COLORED_SAFE_SQUARES.items():
            if self.index in safe_positions:
                self.color = color
                self.is_safe = True
                break


class Board:
    """
    Represents the Ludo game board and manages token positions.
    """

    def __init__(self):
        """Initialize the board with 52 main positions plus home columns."""
        self.main_path_size = 52
        self.home_column_size = 6  # Positions 100-105 for home columns

        # Initialize main board positions (0-51)
        self.positions: Dict[int, Position] = {}
        for i in range(self.main_path_size):
            self.positions[i] = Position(i)

        # Initialize home column positions (100-105)
        for i in range(100, 106):  # 100, 101, 102, 103, 104, 105
            self.positions[i] = Position(i)

        # Optimized blocking positions tracking (initialize before reset_token_positions)
        self._blocking_positions_cache: Dict[str, Set[int]] = {}
        self._cache_valid = False

        # Track positions with multiple tokens by color for faster lookup
        self._multi_token_positions: Dict[str, Set[int]] = {
            PlayerColor.RED.value: set(),
            PlayerColor.GREEN.value: set(),
            PlayerColor.YELLOW.value: set(),
            PlayerColor.BLUE.value: set(),
        }

        # Track which tokens are at each position
        self.token_positions: Dict[int, List[Token]] = {}
        self.reset_token_positions()

        # Starting positions for each color
        self.start_positions = BoardConstants.START_POSITIONS

        # Home column entry positions for each color
        self.home_entries = BoardConstants.HOME_COLUMN_ENTRIES

    def reset_token_positions(self):
        """Reset all token position tracking."""
        self.token_positions.clear()

        # Initialize positions for main board (0-51)
        for i in range(self.main_path_size):
            self.token_positions[i] = []

        # Initialize positions for home columns (100-105)
        for i in range(100, 106):
            self.token_positions[i] = []

        # Reset cache and multi-token tracking
        self._invalidate_blocking_cache()
        for color_set in self._multi_token_positions.values():
            color_set.clear()

    def add_token(self, token: Token, position: int):
        """Add a token to a specific position on the board."""
        if position not in self.token_positions:
            self.token_positions[position] = []

        self.token_positions[position].append(token)

        # Update multi-token tracking for blocking optimization
        if 0 <= position < self.main_path_size:
            player_tokens_count = sum(
                1
                for t in self.token_positions[position]
                if t.player_color == token.player_color
            )
            if player_tokens_count >= 2:
                self._multi_token_positions[token.player_color].add(position)

        # Invalidate cache when board state changes
        self._invalidate_blocking_cache()

    def remove_token(self, token: Token, position: int):
        """Remove a token from a specific position on the board."""
        if position in self.token_positions and token in self.token_positions[position]:
            self.token_positions[position].remove(token)

            # Update multi-token tracking for blocking optimization
            if 0 <= position < self.main_path_size:
                player_tokens_count = sum(
                    1
                    for t in self.token_positions[position]
                    if t.player_color == token.player_color
                )
                if player_tokens_count < 2:
                    self._multi_token_positions[token.player_color].discard(position)

            # Invalidate cache when board state changes
            self._invalidate_blocking_cache()

    def get_tokens_at_position(self, position: int) -> List[Token]:
        """Get all tokens at a specific position."""
        return self.token_positions.get(position, [])

    def is_position_safe(self, position: int, player_color: str) -> bool:
        """Check if a position is safe for a given player color."""
        return BoardConstants.is_safe_position(position, player_color)

    def can_move_to_position(
        self, token: Token, target_position: int
    ) -> Tuple[bool, List[Token]]:
        """
        Check if a token can move to a target position.

        Returns:
            Tuple[bool, List[Token]]: (can_move, tokens_to_capture)
        """
        tokens_at_target = self.get_tokens_at_position(target_position)
        tokens_to_capture: List[Token] = []

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
            # Can stack with own tokens. If opponents present too, decide capture rule below.
            if opponent_tokens:
                # If opponent stack size >=2, it's protected (cannot capture a block)
                opponent_stack_counts = {}
                for ot in opponent_tokens:
                    opponent_stack_counts.setdefault(ot.player_color, 0)
                    opponent_stack_counts[ot.player_color] += 1
                protected = any(count >= 2 for count in opponent_stack_counts.values())
                if not protected:
                    tokens_to_capture = opponent_tokens
            return True, tokens_to_capture

        if opponent_tokens:
            # Single-opponent or mixed-color stack capture logic
            opponent_stack_counts = {}
            for ot in opponent_tokens:
                opponent_stack_counts.setdefault(ot.player_color, 0)
                opponent_stack_counts[ot.player_color] += 1
            # If ANY color has >=2 tokens here, square is blocked from capture
            if any(count >= 2 for count in opponent_stack_counts.values()):
                return False, []
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
        if position == -1:
            return {"type": "home", "is_safe": True, "tokens": []}
        elif 100 <= position <= 105:
            return {
                "type": "home_column",
                "position": position,
                "is_safe": True,
                "tokens": [
                    token.to_dict() for token in self.get_tokens_at_position(position)
                ],
            }
        elif 0 <= position < self.main_path_size:
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
                "type": "unknown",
                "position": position,
                "is_safe": False,
                "tokens": [],
            }

    def update_token_position(self, token: Token, old_position: int, new_position: int):
        """Update token position tracking on the board."""
        if old_position >= 0:
            self.remove_token(token, old_position)
        if new_position >= 0:
            self.add_token(token, new_position)

    def _invalidate_blocking_cache(self):
        """Invalidate the blocking positions cache."""
        self._cache_valid = False
        self._blocking_positions_cache.clear()

    def _rebuild_blocking_cache(self):
        """Rebuild the blocking positions cache for all players."""
        self._blocking_positions_cache.clear()

        for color in [
            PlayerColor.RED.value,
            PlayerColor.GREEN.value,
            PlayerColor.YELLOW.value,
            PlayerColor.BLUE.value,
        ]:
            self._blocking_positions_cache[color] = self._calculate_blocking_positions(
                color
            )

        self._cache_valid = True

    def _calculate_blocking_positions(self, player_color: str) -> Set[int]:
        """
        Calculate blocking positions for a specific player.
        Optimized version that only checks positions with multiple tokens.
        """
        blocking_positions = set()

        # Only check positions that potentially have multiple tokens of this color
        candidate_positions = self._multi_token_positions.get(player_color, set())

        for position in candidate_positions:
            # Double-check that position is still valid and has multiple tokens
            if 0 <= position < self.main_path_size and not self.is_position_safe(
                position, player_color
            ):
                player_tokens = [
                    t
                    for t in self.token_positions.get(position, [])
                    if t.player_color == player_color
                ]

                if len(player_tokens) >= 2:
                    blocking_positions.add(position)
                else:
                    # Clean up inconsistent state
                    self._multi_token_positions[player_color].discard(position)

        return blocking_positions

    def get_blocking_positions(self, player_color: str) -> Set[int]:
        """
        Get positions where this player is blocking opponents.

        This method is now optimized with caching and efficient data structures.
        Time complexity reduced from O(n*m) to O(1) for cached results,
        where n is number of positions and m is average tokens per position.
        """
        # Return cached result if available
        if self._cache_valid and player_color in self._blocking_positions_cache:
            return self._blocking_positions_cache[player_color].copy()

        # Rebuild entire cache if invalid (affects all players)
        if not self._cache_valid:
            self._rebuild_blocking_cache()
            return self._blocking_positions_cache.get(player_color, set()).copy()

        # Calculate for this specific player if not in cache
        blocking_positions = self._calculate_blocking_positions(player_color)
        self._blocking_positions_cache[player_color] = blocking_positions

        return blocking_positions.copy()

    def get_all_blocking_positions(self) -> Dict[str, Set[int]]:
        """
        Get blocking positions for all players at once.
        More efficient than calling get_blocking_positions for each player separately.
        """
        if not self._cache_valid:
            self._rebuild_blocking_cache()

        return {
            color: positions.copy()
            for color, positions in self._blocking_positions_cache.items()
        }

    def has_blocking_position(self, player_color: str, position: int) -> bool:
        """
        Quick check if a specific position is blocking for a player.
        More efficient than getting all blocking positions when you only need one.
        """
        if not (0 <= position < self.main_path_size):
            return False

        if self.is_position_safe(position, player_color):
            return False

        # Check if this position has multiple tokens of the same color
        player_tokens = [
            t
            for t in self.token_positions.get(position, [])
            if t.player_color == player_color
        ]

        return len(player_tokens) >= 2

    def __str__(self) -> str:
        """String representation of the board state."""
        result = "Board State:\n"
        for position in range(self.main_path_size):
            tokens = self.get_tokens_at_position(position)
            if tokens:
                result += f"Position {position}: {[str(token) for token in tokens]}\n"
        return result
