"""
Main Ludo game implementation.
Manages game flow, rules, and provides interface for AI players.
"""

import random
from typing import Dict, List, Optional

from .board import Board
from .player import Player, PlayerColor


class LudoGame:
    """
    Main Ludo game class that manages the entire game state and flow.

    This class provides comprehensive game management including:
    - Turn-based gameplay with dice rolling
    - Move validation and execution
    - AI player support with strategic decision making
    - Game state tracking and history
    - Flexible reset options that preserve player configurations
    """

    def __init__(self, player_colors: List[PlayerColor]):
        """
        Initialize a new Ludo game.

        Args:
            player_colors: List of colors for players (2-4 players)
        """
        if len(player_colors) < 2 or len(player_colors) > 4:
            raise ValueError("Ludo requires 2-4 players")

        self.board = Board()
        self.players: List[Player] = []

        # Create players
        for i, color in enumerate(player_colors):
            player = Player(color, i)
            self.players.append(player)

        self.current_player_index = 0
        self.game_over = False
        self.winner: Optional[Player] = None
        self.turn_count = 0

        # Game state tracking
        self.consecutive_sixes = 0
        self.last_dice_value = 0
        self.move_history: List[Dict] = []

        # Initialize board with all tokens in home
        self._initialize_board()

    def _initialize_board(self):
        """Place all tokens in their starting home positions."""
        for player in self.players:
            for token in player.tokens:
                # Tokens start in home (position -1), so no board placement needed
                pass

    def get_current_player(self) -> Player:
        """Get the player whose turn it is."""
        return self.players[self.current_player_index]

    def roll_dice(self) -> int:
        """Roll a six-sided die and return the result."""
        dice_value = random.randint(1, 6)
        self.last_dice_value = dice_value

        # Track consecutive sixes
        if dice_value == 6:
            self.consecutive_sixes += 1
        else:
            self.consecutive_sixes = 0

        return dice_value

    def can_player_move(self, player: Player, dice_value: int) -> bool:
        """Check if a player can make any move with the given dice value."""
        return player.can_move_any_token(dice_value)

    def get_valid_moves(self, player: Player, dice_value: int) -> List[Dict]:
        """
        Get all valid moves for a player with the given dice value.

        Returns:
            List[Dict]: List of valid moves with detailed information
        """
        if self.consecutive_sixes >= 3:
            # Player loses turn after 3 consecutive sixes
            return []

        possible_moves = player.get_possible_moves(dice_value)
        valid_moves = []

        for move in possible_moves:
            token = player.tokens[move["token_id"]]
            target_position = move["target_position"]

            # Check if the move is actually valid on the board
            can_move, tokens_to_capture = self.board.can_move_to_position(
                token, target_position
            )

            if can_move:
                move["captures_opponent"] = len(tokens_to_capture) > 0
                move["captured_tokens"] = [
                    {"player_color": t.player_color, "token_id": t.token_id}
                    for t in tokens_to_capture
                ]
                valid_moves.append(move)

        return valid_moves

    def execute_move(self, player: Player, token_id: int, dice_value: int) -> Dict:
        """
        Execute a move for a player.

        Args:
            player: The player making the move
            token_id: ID of the token to move (0-3)
            dice_value: The dice value used for the move

        Returns:
            Dict: Result of the move including any captures, extra turns, etc.
        """
        if token_id < 0 or token_id >= 4:
            return {"success": False, "error": "Invalid token ID"}

        token = player.tokens[token_id]

        # Check if move is valid
        if not token.can_move(dice_value, None):
            return {"success": False, "error": "Token cannot move with this dice value"}

        old_position = token.position
        target_position = token.get_target_position(dice_value, player.start_position)

        if target_position == -1:  # HOME_POSITION sentinel for invalid
            return {"success": False, "error": "Invalid target position"}

        # Validate board occupancy / capture
        can_move, tokens_to_capture = self.board.can_move_to_position(
            token, target_position
        )
        if not can_move:
            return {"success": False, "error": "Invalid move - position blocked"}

        # Apply board side-effects (captures & relocation) first
        captured_tokens = self.board.execute_move(token, old_position, target_position)

        # Now commit token internal state without recomputation
        token.commit_move(target_position, player.start_position)

        # Create move result
        move_result = {
            "success": True,
            "player_color": player.color.value,
            "token_id": token_id,
            "dice_value": dice_value,
            "old_position": old_position,
            "new_position": token.position,
            "captured_tokens": [
                {"player_color": t.player_color, "token_id": t.token_id}
                for t in captured_tokens
            ],
            "token_finished": token.is_finished(),
            "extra_turn": False,
        }

        # Check for extra turn conditions
        extra_turn = False

        # Rolling a 6 gives extra turn
        if dice_value == 6:
            extra_turn = True

        # Capturing opponent gives extra turn
        if captured_tokens:
            extra_turn = True

        # Finishing a token gives extra turn
        if token.is_finished():
            extra_turn = True

        # But not if player rolled 3 sixes in a row
        if self.consecutive_sixes >= 3:
            extra_turn = False

        move_result["extra_turn"] = extra_turn

        # Record move in history
        self.move_history.append(move_result.copy())

        # Check for game over
        if player.has_won():
            self.game_over = True
            self.winner = player
            move_result["game_won"] = True

        return move_result

    def next_turn(self):
        """Move to the next player's turn."""
        self.consecutive_sixes = 0  # Reset when changing players
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        self.turn_count += 1

    def play_turn(self, token_id: Optional[int] = None) -> Dict:
        """
        Play a complete turn for the current player.

        Args:
            token_id: If provided, move this specific token. Otherwise, use AI/random selection.

        Returns:
            Dict: Complete turn result
        """
        if self.game_over:
            return {"error": "Game is already over"}

        current_player = self.get_current_player()
        dice_value = self.roll_dice()

        turn_result = {
            "player_color": current_player.color.value,
            "dice_value": dice_value,
            "consecutive_sixes": self.consecutive_sixes,
            "moves": [],
            "extra_turn": False,
            "turn_ended": False,
        }

        # Check for 3 consecutive sixes
        if self.consecutive_sixes >= 3:
            turn_result["turn_ended"] = True
            turn_result["reason"] = "Three consecutive sixes - turn forfeited"
            self.next_turn()
            return turn_result

        # Get valid moves
        valid_moves = self.get_valid_moves(current_player, dice_value)

        if not valid_moves:
            turn_result["turn_ended"] = True
            turn_result["reason"] = "No valid moves available"
            self.next_turn()
            return turn_result

        # Select move (use provided token_id or first available)
        if token_id is not None:
            # Find the move for the specified token
            selected_move = None
            for move in valid_moves:
                if move["token_id"] == token_id:
                    selected_move = move
                    break

            if not selected_move:
                turn_result["error"] = f"Token {token_id} cannot move"
                self.next_turn()
                return turn_result
        else:
            # Select first available move (can be enhanced with AI logic)
            selected_move = valid_moves[0]

        # Execute the move
        move_result = self.execute_move(
            current_player, selected_move["token_id"], dice_value
        )
        turn_result["moves"].append(move_result)

        if not move_result["success"]:
            turn_result["error"] = move_result["error"]
            self.next_turn()
            return turn_result

        # Check if player gets extra turn
        if move_result["extra_turn"]:
            turn_result["extra_turn"] = True
        else:
            turn_result["turn_ended"] = True
            self.next_turn()

        return turn_result

    def get_ai_decision_context(self, dice_value: int) -> Dict:
        """
        Get context specifically designed for AI decision making.

        Args:
            dice_value: The dice value rolled

        Returns:
            Dict: AI decision context
        """
        current_player = self.get_current_player()
        valid_moves = self.get_valid_moves(current_player, dice_value)

        context = {
            "current_situation": {
                "player_color": current_player.color.value,
                "dice_value": dice_value,
                "consecutive_sixes": self.consecutive_sixes,
                "turn_count": self.turn_count,
            },
            "player_state": current_player.get_game_state(),
            "opponents": [
                {
                    "color": p.color.value,
                    "tokens_finished": p.get_finished_tokens_count(),
                    "tokens_active": sum(1 for t in p.tokens if t.is_active()),
                    "threat_level": self._calculate_threat_level(p),
                }
                for p in self.players
                if p != current_player
            ],
            "valid_moves": valid_moves,
            "strategic_analysis": self._analyze_strategic_situation(
                current_player, valid_moves
            ),
        }

        return context

    def _calculate_threat_level(self, player: Player) -> float:
        """Calculate how much of a threat a player is (0.0 to 1.0)."""
        finished_tokens = player.get_finished_tokens_count()
        active_tokens = sum(1 for token in player.tokens if not token.is_in_home())

        # Higher threat if more tokens are finished or close to finishing
        threat = (finished_tokens * 0.4) + (active_tokens * 0.1)
        return min(threat, 1.0)

    def _analyze_strategic_situation(
        self, player: Player, valid_moves: List[Dict]
    ) -> Dict:
        """Analyze the strategic situation for AI decision making."""
        analysis = {
            "can_capture": any(move["captures_opponent"] for move in valid_moves),
            "can_finish_token": any(
                move["move_type"] == "finish" for move in valid_moves
            ),
            "can_exit_home": any(
                move["move_type"] == "exit_home" for move in valid_moves
            ),
            "safe_moves": [move for move in valid_moves if move["is_safe_move"]],
            "risky_moves": [move for move in valid_moves if not move["is_safe_move"]],
            "best_strategic_move": (
                max(valid_moves, key=lambda m: m["strategic_value"])
                if valid_moves
                else None
            ),
        }

        return analysis

    def get_player_configurations(self) -> List[Dict]:
        """
        Get current player configurations including strategies.
        Useful for saving and restoring game setups.

        Returns:
            List[Dict]: Player configurations
        """
        configs = []
        for player in self.players:
            config = {
                "color": player.color.value,
                "player_id": player.player_id,
                "strategy_name": player.get_strategy_name(),
                "strategy_description": player.get_strategy_description(),
                "has_strategy": player.strategy is not None,
                "tokens_finished": player.get_finished_tokens_count(),
                "tokens_active": sum(1 for t in player.tokens if t.is_active()),
                "tokens_in_home": sum(1 for t in player.tokens if t.is_in_home()),
            }
            configs.append(config)
        return configs

    def __str__(self) -> str:
        """String representation of the current game state."""
        current_player = self.get_current_player()
        result = f"Ludo Game - Turn {self.turn_count}\n"
        result += f"Current Player: {current_player.color.value}\n"
        result += f"Game Over: {self.game_over}\n"
        if self.winner:
            result += f"Winner: {self.winner.color.value}\n"
        result += "\nPlayer States:\n"
        for player in self.players:
            result += f"  {player.color.value}: {player.get_finished_tokens_count()}/4 finished\n"
        return result
