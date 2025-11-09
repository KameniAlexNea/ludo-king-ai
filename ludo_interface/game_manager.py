import random
from typing import Dict, List, Optional

from ludo_rl.ludo.game import LudoGame
from ludo_rl.ludo.piece import Piece as Token
from ludo_rl.strategy.registry import create as create_strategy

from .models import PlayerColor, PTOPlayerColor


class GameState:
    """Tracks additional game state needed for the UI."""

    def __init__(self):
        self.current_player_index: int = 0
        self.winner_index: Optional[int] = None
        self.game_over: bool = False
        self.finish_order: List[int] = []


class GameManager:
    """Handles core game logic and state management."""

    def __init__(self, default_players: List[PlayerColor], show_token_ids: bool):
        self.default_players = default_players
        self.show_token_ids = show_token_ids

    def init_game(self, strategies: List[str]) -> tuple[LudoGame, GameState]:
        """Initializes a new Ludo game with the given strategies."""
        game = LudoGame()
        state = GameState()

        # Reset players and attach strategies
        for idx, (player, strategy_name) in enumerate(zip(game.players, strategies)):
            player.reset()
            player.strategy_name = strategy_name
            try:
                player._strategy = create_strategy(strategy_name)
            except KeyError:
                player.strategy_name = "random"
                player._strategy = None

        return game, state

    def game_state_tokens(self, game: LudoGame) -> Dict[PlayerColor, List[Token]]:
        """Extracts token information from the game state."""
        token_map: Dict[PlayerColor, List[Token]] = {c: [] for c in PlayerColor}
        for p in game.players:
            token_map[PTOPlayerColor[p.color]] = [t for t in p.pieces]
        return token_map

    def is_human_turn(self, game: LudoGame, state: GameState) -> bool:
        """Check if it's currently a human player's turn."""
        if state.game_over:
            return False
        current_player = game.players[state.current_player_index]
        return current_player.strategy_name.lower() == "human"

    def get_human_move_options(
        self, game: LudoGame, state: GameState, dice: int
    ) -> List[dict]:
        """Get move options for a human player."""
        valid_moves = game.get_valid_moves(state.current_player_index, dice)

        options = []
        for move in valid_moves:
            piece = move["piece"]
            options.append(
                {
                    "piece_id": piece.piece_id,
                    "description": f"Piece {piece.piece_id}: pos {piece.position} -> {move['new_pos']}",
                    "move": move,
                }
            )
        return options

    def serialize_move(self, player_index: int, move: dict, result: dict) -> str:
        """Serializes a move result into a human-readable string."""
        if not move or not result:
            return "No move"

        piece = move["piece"]
        new_pos = move["new_pos"]
        events = result.get("events", {})

        parts = [f"Player {player_index} piece {piece.piece_id} -> {new_pos}"]

        if events.get("knockouts"):
            parts.append(f"knocked out {len(events['knockouts'])}")
        if events.get("finished"):
            parts.append("finished")
        if result.get("extra_turn"):
            parts.append("extra turn")

        return ", ".join(parts)

    def play_step(
        self,
        game: LudoGame,
        state: GameState,
        human_move_choice: Optional[int] = None,
        dice: Optional[int] = None,
    ):
        """Plays a single step of the game.

        If `dice` is provided, use it; otherwise roll a new dice value.
        """
        if state.game_over:
            return game, state, "Game over", self.game_state_tokens(game), [], False

        current_player = game.players[state.current_player_index]

        # Check if player has already won
        if current_player.has_won():
            if state.current_player_index not in state.finish_order:
                state.finish_order.append(state.current_player_index)

            # Move to next player
            state.current_player_index = (state.current_player_index + 1) % 4

            # Check if game is over
            if len(state.finish_order) >= 4:
                state.game_over = True
                state.winner_index = state.finish_order[0]
                desc = f"Game over! Winner: Player {state.winner_index}"
                return game, state, desc, self.game_state_tokens(game), [], False

            desc = f"Player {state.current_player_index} has already won, moving to next player"
            return game, state, desc, self.game_state_tokens(game), [], False

        if dice is None:
            dice = game.roll_dice()

        valid_moves = game.get_valid_moves(state.current_player_index, dice)

        if not valid_moves:
            extra_turn = dice == 6
            token_positions = ", ".join(
                [
                    f"piece {i}: {p.position}"
                    for i, p in enumerate(current_player.pieces)
                ]
            )
            desc = f"Player {state.current_player_index} rolled {dice} - no moves{' (extra turn)' if extra_turn else ''} | Positions: {token_positions}"

            if not extra_turn:
                state.current_player_index = (state.current_player_index + 1) % 4

            return game, state, desc, self.game_state_tokens(game), [], False

        # Check if it's a human player's turn and we need input
        is_human = current_player.strategy_name.lower() == "human"
        if is_human and human_move_choice is None:
            move_options = self.get_human_move_options(game, state, dice)
            desc = (
                f"Player {state.current_player_index} rolled {dice} - Choose your move:"
            )
            return game, state, desc, self.game_state_tokens(game), move_options, True

        # Select move
        chosen_move = None
        if is_human and human_move_choice is not None:
            # Find the move for the chosen piece
            chosen_move = next(
                (m for m in valid_moves if m["piece"].piece_id == human_move_choice),
                None,
            )
        else:
            # AI decision
            board_stack = game.build_board_tensor(state.current_player_index)
            chosen_move = current_player.decide(board_stack, dice, valid_moves)

        if chosen_move is None:
            chosen_move = random.choice(valid_moves)

        # Execute move
        result = game.make_move(
            state.current_player_index,
            chosen_move["piece"],
            chosen_move["new_pos"],
            dice,
        )

        desc = f"Player {state.current_player_index} rolled {dice}: {self.serialize_move(state.current_player_index, chosen_move, result)}"

        # Check if player won
        if (
            current_player.has_won()
            and state.current_player_index not in state.finish_order
        ):
            state.finish_order.append(state.current_player_index)
            desc += " | Player has WON!"

        # Check if game is over
        if len(state.finish_order) >= 4:
            state.game_over = True
            state.winner_index = state.finish_order[0]
            desc += f" | GAME OVER! Winner: Player {state.winner_index}"

        # Move to next player if no extra turn
        if not result.get("extra_turn", False) and not state.game_over:
            state.current_player_index = (state.current_player_index + 1) % 4

        return game, state, desc, self.game_state_tokens(game), [], False
