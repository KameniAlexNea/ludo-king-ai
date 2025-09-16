"""
Player representation for Ludo game.
Each player has a color and controls 4 tokens.
"""

from enum import Enum
from typing import Dict, List

from ludo_engine.constants import BoardConstants, GameConstants, StrategyConstants
from ludo_engine.strategies.base import Strategy
from ludo_engine.token import Token, TokenState


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

    def __init__(self, color: PlayerColor, player_id: int, strategy: Strategy = None):
        """
        Initialize a player with their color and 4 tokens.

        Args:
            color: Player's color (RED, BLUE, GREEN, YELLOW)
            player_id: Unique identifier for the player (0-3)
            strategy: Strategy instance for AI decision making (optional)
        """
        self.color = color
        self.player_id = player_id
        self.tokens: List[Token] = []
        self.strategy: Strategy = strategy

        # Create 4 tokens for this player
        for i in range(4):
            token = Token(token_id=i, player_color=color.value, state=TokenState.HOME)
            self.tokens.append(token)

        # Starting positions for each color on the board
        self.start_positions = BoardConstants.START_POSITIONS
        self.start_position = self.start_positions[color.value]

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
            if token.can_move(dice_value):  # Simplified check for now
                movable_tokens.append(token)

        return movable_tokens

    def has_tokens_in_home(self) -> bool:
        """Check if player has any tokens still in home."""
        return any(token.is_in_home() for token in self.tokens)

    def has_active_tokens(self) -> bool:
        """Check if player has any tokens on the board."""
        return any(
            token.is_active() or token.is_in_home_column() for token in self.tokens
        )

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
            "player_id": self.player_id,
            "color": self.color.value,
            "start_position": self.start_position,
            "tokens": [token.to_dict() for token in self.tokens],
            "tokens_in_home": sum(1 for token in self.tokens if token.is_in_home()),
            "active_tokens": sum(1 for token in self.tokens if token.is_active()),
            "tokens_in_home_column": sum(
                1 for token in self.tokens if token.is_in_home_column()
            ),
            "finished_tokens": self.get_finished_tokens_count(),
            "has_won": self.has_won(),
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
            if token.can_move(dice_value):
                target_position = token.get_target_position(
                    dice_value, self.start_position
                )

                strategic_value, _components = self._calculate_strategic_value(
                    token, dice_value, target_position
                )

                move_info = {
                    "token_id": token.token_id,
                    "current_position": token.position,
                    "current_state": token.state.value,
                    "target_position": target_position,
                    "move_type": self._get_move_type(token, dice_value),
                    "is_safe_move": self._is_safe_move(token, target_position),
                    "captures_opponent": False,  # Will be calculated by board
                    "strategic_value": strategic_value,
                    "strategic_components": _components,
                }

                possible_moves.append(move_info)

        return possible_moves

    def _get_move_type(self, token: Token, dice_value: int) -> str:
        """Determine the type of move being made."""
        if token.is_in_home() and dice_value == GameConstants.EXIT_HOME_ROLL:
            return "exit_home"
        if token.is_in_home_column():
            target = token.get_target_position(dice_value, self.start_position)
            if target == GameConstants.FINISH_POSITION:
                return "finish"
            return "advance_home_column"
        return "advance_main_board"

    def _is_safe_move(self, token: Token, target_position: int) -> bool:
        """Check if the target position is a safe square."""
        return BoardConstants.is_safe_position(target_position, self.color.value)

    def _calculate_strategic_value(
        self, token: Token, dice_value: int, target_position: int | None = None
    ) -> tuple[float, Dict[str, float]]:
        """Enhanced heuristic with component breakdown.

        Components implemented per requested improvements:
          1. Exit home bonus
          2. Home column depth scaling & finish value
          3. Forward progress weight
          4. Distance-to-finish acceleration
          5. Safety bonus (landing on safe square)
          6. Vulnerability penalty (simple heuristic: landing square unsafe & within
             typical opponent reach window => penalty)

        Returns summed value and component dict for later analysis.
        """
        if target_position is None:
            target_position = token.get_target_position(dice_value, self.start_position)

        components: Dict[str, float] = {
            "exit_home": 0.0,
            "finish": 0.0,
            "home_column_depth": 0.0,
            "forward_progress": 0.0,
            "acceleration": 0.0,
            "safety": 0.0,
            "vulnerability_penalty": 0.0,
        }

        # 1 & 2: Home column / finish logic
        if token.is_in_home_column():
            if target_position == GameConstants.FINISH_POSITION:
                components["finish"] = StrategyConstants.FINISH_TOKEN_VALUE
            else:
                depth = target_position - GameConstants.HOME_COLUMN_START  # 0..5
                max_depth = GameConstants.HOME_COLUMN_SIZE - 1
                depth_ratio = depth / max_depth if max_depth > 0 else 0
                base = StrategyConstants.HOME_COLUMN_ADVANCE_VALUE
                components["home_column_depth"] = base * (
                    1 + depth_ratio * StrategyConstants.HOME_COLUMN_DEPTH_MULTIPLIER
                )
        elif token.is_in_home() and dice_value == GameConstants.EXIT_HOME_ROLL:
            # 1: Exit home
            components["exit_home"] = StrategyConstants.EXIT_HOME_VALUE
        elif token.is_active():
            # 3: Forward progress
            components["forward_progress"] = (
                dice_value * StrategyConstants.FORWARD_PROGRESS_WEIGHT
            )
            # 4: Acceleration (closer to finish yields more)
            steps_remaining = self._estimate_steps_to_finish(token.position)
            # Heuristic: fewer remaining steps => larger bonus
            # Convert to pseudo remaining advantage (higher when closer)
            advantage = max(0, 60 - steps_remaining)  # 60 is rough total path+home
            components["acceleration"] = (
                advantage * StrategyConstants.ACCELERATION_WEIGHT
            )

        # 5: Safety bonus for landing square
        if BoardConstants.is_safe_position(target_position, self.color.value):
            components["safety"] = StrategyConstants.SAFETY_BONUS

        # 6: Vulnerability penalty (simple placeholder): if not safe and token is active
        # and not entering home column and not finishing, apply penalty.
        if (
            not BoardConstants.is_safe_position(target_position, self.color.value)
            and not BoardConstants.is_home_column_position(target_position)
            and token.is_active()
        ):
            components[
                "vulnerability_penalty"
            ] = -StrategyConstants.VULNERABILITY_PENALTY_WEIGHT

        total = sum(components.values())
        return total, components

    def _estimate_steps_to_finish(self, position: int) -> int:
        """Rough heuristic of remaining steps to finish from a main-board position.

        Not exact path math; adequate for acceleration heuristic.
        """
        # If already in home column or finished we don't normally call here, guard anyway
        if BoardConstants.is_home_column_position(position):
            # Remaining within column
            return GameConstants.FINISH_POSITION - position

        # Path: distance to home entry + home column size
        # Find this player's home entry square
        entry = BoardConstants.HOME_COLUMN_ENTRIES[self.color.value]
        if position <= entry:
            to_entry = entry - position
        else:
            to_entry = (GameConstants.MAIN_BOARD_SIZE - position) + entry
        return to_entry + GameConstants.HOME_COLUMN_SIZE

    def set_strategy(self, strategy: Strategy):
        """
        Set the strategy for this player.

        Args:
            strategy: Strategy instance for decision making
        """
        self.strategy = strategy

    def make_strategic_decision(self, game_context: Dict) -> int:
        """
        Make a strategic decision using the assigned strategy.

        Args:
            game_context: Complete game context from the game engine

        Returns:
            int: token_id to move (0-3)
        """
        if self.strategy is None:
            # Fallback to simple decision if no strategy set
            return self._make_simple_decision(game_context)

        return self.strategy.decide(game_context)

    def _make_simple_decision(self, game_context: Dict) -> int:
        """
        Simple fallback decision making without strategy.
        Uses basic priority system.
        """
        valid_moves = game_context.get("valid_moves", [])

        if not valid_moves:
            return 0

        # Simple priority: finish > capture > exit > highest value
        for move in valid_moves:
            if move["move_type"] == "finish":
                return move["token_id"]

        for move in valid_moves:
            if move["captures_opponent"]:
                return move["token_id"]

        for move in valid_moves:
            if move["move_type"] == "exit_home":
                return move["token_id"]

        # Choose highest strategic value
        best_move = max(valid_moves, key=lambda m: m["strategic_value"])
        return best_move["token_id"]

    def get_strategy_name(self) -> str:
        """Get the name of the current strategy."""
        if self.strategy is None:
            return "Simple"
        return self.strategy.name

    def get_strategy_description(self) -> str:
        """Get the description of the current strategy."""
        if self.strategy is None:
            return "Basic priority-based decision making"
        return self.strategy.description

    def __str__(self) -> str:
        """String representation of the player."""
        strategy_name = self.get_strategy_name()
        return f"Player({self.color.value}, strategy: {strategy_name}, tokens: {len([t for t in self.tokens if not t.is_in_home()])} active)"
