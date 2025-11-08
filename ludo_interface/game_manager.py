from typing import Dict, List, Optional

from ludo_engine.models import MoveResult
from ludo_engine.strategies import HumanStrategy
from ludo_engine.strategies.strategy import StrategyFactory
from .models import PlayerColor
from ludo_rl.ludo.game import LudoGame
from ludo_rl.ludo.piece import Piece as Token


class GameManager:
    """Handles core game logic and state management."""

    def __init__(self, default_players: List[PlayerColor], show_token_ids: bool):
        self.default_players = default_players
        self.show_token_ids = show_token_ids

    def init_game(self, strategies: List[str]) -> LudoGame:
        """Initializes a new Ludo game with the given strategies."""
        strategy_objs = [StrategyFactory.create_strategy(name) for name in strategies]
        game = LudoGame(self.default_players)
        for player, strat in zip(game.players, strategy_objs):
            player.set_strategy(strat)
        return game

    def game_state_tokens(self, game: LudoGame) -> Dict[PlayerColor, List[Token]]:
        """Extracts token information from the game state."""
        token_map: Dict[PlayerColor, List[Token]] = {c: [] for c in PlayerColor}
        for p in game.players:
            for t in p.pieces:
                token_map[p.color].append(t)
        return token_map

    def get_human_strategy(self, game: LudoGame) -> Optional[HumanStrategy]:
        """Get the human strategy from the current player if it exists."""
        current_player = game.get_current_player()
        return (
            current_player.strategy
            if isinstance(current_player.strategy, HumanStrategy)
            else None
        )

    def is_human_turn(self, game: LudoGame) -> bool:
        """Check if it's currently a human player's turn."""
        return self.get_human_strategy(game) is not None

    def get_human_move_options(self, game: LudoGame, dice: int) -> List[dict]:
        """Get move options for a human player."""
        current_player = game.get_current_player()
        valid_moves = game.get_valid_moves(current_player, dice)

        options = []
        for move in valid_moves:
            token = current_player.pieces[move.token_id]
            options.append(
                {
                    "token_id": move.token_id,
                    "description": f"Token {move.token_id}: {token.state.value} at {token.position} -> {move.target_position}",
                    "move_type": move.move_type,
                }
            )
        return options

    def serialize_move(self, move_result: MoveResult) -> str:
        """Serializes a move result into a human-readable string."""
        if not move_result or not move_result.success:
            return "No move"
        parts = [
            f"{move_result.player_color} token {move_result.token_id} -> {move_result.new_position}"
        ]
        if move_result.captured_tokens:
            parts.append(f"captured {len(move_result.captured_tokens)}")
        if move_result.finished_token:
            parts.append("finished")
        if move_result.extra_turn:
            parts.append("extra turn")
        return ", ".join(parts)

    def play_step(
        self,
        game: LudoGame,
        human_move_choice: Optional[int] = None,
        dice: Optional[int] = None,
    ):
        """Plays a single step of the game.

        If `dice` is provided, use it; otherwise roll a new dice value.
        """
        if game.game_over:
            return game, "Game over", self.game_state_tokens(game), [], False

        current_player = game.get_current_player()
        if dice is None:
            dice = game.roll_dice()
        valid_moves = game.get_valid_moves(current_player, dice)

        if not valid_moves:
            extra_turn = dice == 6
            if not extra_turn:
                game.next_turn()

            token_positions = ", ".join(
                [
                    f"token {i}: {t.position} ({t.state.value})"
                    for i, t in enumerate(current_player.tokens)
                ]
            )
            desc = f"{current_player.color.value} rolled {dice} - no moves{' (extra turn)' if extra_turn else ''} | Positions: {token_positions}"
            return game, desc, self.game_state_tokens(game), [], False

        human_strategy = self.get_human_strategy(game)
        if human_strategy and human_move_choice is None:
            move_options = self.get_human_move_options(game, dice)
            desc = f"{current_player.color.value} rolled {dice} - Choose your move:"
            return game, desc, self.game_state_tokens(game), move_options, True

        chosen_move = None
        if human_strategy and human_move_choice is not None:
            chosen_move = next(
                (m for m in valid_moves if m.token_id == human_move_choice), None
            )
        else:
            ctx = game.get_ai_decision_context(dice)
            token_choice = current_player.make_strategic_decision(ctx)
            chosen_move = next(
                (m for m in valid_moves if m.token_id == token_choice), None
            )

        if chosen_move is None:
            chosen_move = valid_moves[0]

        move_res = game.execute_move(current_player, chosen_move.token_id, dice)
        desc = f"{current_player.color.value} rolled {dice}: {self.serialize_move(move_res)}"

        if not move_res.extra_turn and not game.game_over:
            game.next_turn()

        if game.game_over:
            desc += f" | WINNER: {game.winner.color.value}"

        return game, desc, self.game_state_tokens(game), [], False