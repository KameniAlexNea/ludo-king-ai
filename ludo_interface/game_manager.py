import random
from typing import Dict, List, Optional, Union


from ludo_rl.ludo_king import Color, Game, Player
from ludo_rl.ludo_king.piece import Piece as Token
from ludo_rl.ludo_king.types import Move, MoveResult
from ludo_rl.strategy.llm_agent import LLMStrategy
from ludo_rl.strategy.registry import create as create_strategy
from ludo_rl.strategy.rl_agent import RLStrategy

from .llm_config_ui import LLMProviderConfig, RLModelConfig
from .models import PlayerColor, PTOPlayerColor


def create_strategy_instance(
    strategy_name: str, configs: dict[str, Union[RLModelConfig, LLMProviderConfig]]
):
    """Factory to create strategy instances based on strategy name."""
    if strategy_name.startswith("llm:"):
        config: LLMProviderConfig = configs[strategy_name]
        return LLMStrategy.configure_with_model_name(
            model_name=f"{config.provider}:{config.model_name}", api_key=config.api_key
        )
    elif strategy_name.startswith("rl:"):
        config: RLModelConfig = configs[strategy_name]
        # Using default device='cpu' here; extend RLModelConfig to pass device/determinism if needed
        return RLStrategy.configure_from_path(config.path)
    else:
        return create_strategy(strategy_name)


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

    def init_game(
        self,
        strategies: List[str],
        configs: dict[str, Union[RLModelConfig, LLMProviderConfig]],
    ) -> tuple[Game, GameState]:
        """Initializes a new Ludo game with the given strategies.

        Supports 4-player and 2-player games. Any dropdown set to 'empty' is
        excluded; if exactly two non-empty strategies are provided, a 2-player
        game is created using only those seats (in the provided color order).
        Otherwise, a standard 4-player game is created, treating 'empty' as
        'random'.
        """
        color_order = [Color.RED, Color.GREEN, Color.YELLOW, Color.BLUE]

        # Pair each seat with its chosen strategy
        seat_specs = list(zip(color_order, strategies))
        active_specs = [
            (color, strat)
            for color, strat in seat_specs
            if str(strat).strip().lower() != "empty"
        ]

        if len(active_specs) == 2:
            # Two-player game with selected seats only
            players = [Player(color=int(color)) for color, _ in active_specs]
            game = Game(players=players)
            state = GameState()
            # Attach strategies to the two players
            for player, (_, strategy_name) in zip(game.players, active_specs):
                player.strategy_name = strategy_name
                try:
                    player.strategy = create_strategy_instance(strategy_name, configs)
                except KeyError:
                    player.strategy_name = "human"
            return game, state

        # Default: 4-player game (replace 'empty' with 'random')
        norm_strategies = [
            (str(s).strip().lower() if str(s).strip().lower() != "empty" else "random")
            for s in strategies
        ]
        players = [Player(color=int(c)) for c in color_order]
        game = Game(players=players)
        state = GameState()
        for player, strategy_name in zip(game.players, norm_strategies):
            player.strategy_name = strategy_name
            try:
                player.strategy = create_strategy_instance(strategy_name, configs)
            except KeyError:
                player.strategy_name = "human"
        return game, state

    def game_state_tokens(self, game: Game) -> Dict[PlayerColor, List[Token]]:
        """Extracts token information from the game state."""
        token_map: Dict[PlayerColor, List[Token]] = {c: [] for c in PlayerColor}
        for p in game.players:
            token_map[PTOPlayerColor[p.color]] = [t for t in p.pieces]
        return token_map

    def is_human_turn(self, game: Game, state: GameState) -> bool:
        """Check if it's currently a human player's turn."""
        if state.game_over:
            return False
        current_player = game.players[state.current_player_index]
        return current_player.strategy_name.lower() == "human"

    def get_human_move_options(
        self, game: Game, state: GameState, dice: int
    ) -> List[dict]:
        """Get move options for a human player."""
        valid_moves = game.legal_moves(state.current_player_index, dice)
        player = game.players[state.current_player_index]
        options: List[dict] = []
        for mv in valid_moves:
            piece = player.pieces[mv.piece_id]
            options.append(
                {
                    "piece_id": mv.piece_id,
                    "description": f"Piece {mv.piece_id}: pos {piece.position} -> {mv.new_pos}",
                }
            )
        return options

    def serialize_move(self, player_index: int, move: Move, result: MoveResult) -> str:
        """Serializes a move result into a human-readable string."""
        if move is None or result is None:
            return "No move"

        old_pos = result.old_position
        new_pos = move.new_pos
        parts = [f"Player {player_index} piece {move.piece_id}: {old_pos} -> {new_pos}"]

        if result.events.knockouts:
            knockout_details = []
            for ko in result.events.knockouts:
                knocked_player = ko["player"]
                knocked_piece = ko["piece_id"]
                knockout_details.append(
                    f"Player {knocked_player} piece {knocked_piece}"
                )
            parts.append(f"knocked out {', '.join(knockout_details)}")
        if result.events.finished:
            parts.append("finished")
        if result.extra_turn:
            parts.append("extra turn")

        return ", ".join(parts)

    def play_step(
        self,
        game: Game,
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

        # Check if player has already won - skip their turn
        if current_player.check_won():
            total = len(game.players)
            prev = state.current_player_index
            state.current_player_index = (state.current_player_index + 1) % total
            desc = f"Player {prev} has already won, skipping turn"
            return game, state, desc, self.game_state_tokens(game), [], False

        if dice is None:
            dice = game.roll_dice()

        valid_moves = game.legal_moves(state.current_player_index, dice)

        if not valid_moves:
            token_positions = ", ".join(
                [
                    f"piece {i}: {p.position}"
                    for i, p in enumerate(current_player.pieces)
                ]
            )
            desc = f"Player {state.current_player_index} rolled {dice} - no moves | Positions: {token_positions}"
            total = len(game.players)
            state.current_player_index = (state.current_player_index + 1) % total
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
        if is_human and human_move_choice is not None:
            chosen_move = next(
                (m for m in valid_moves if m.piece_id == human_move_choice), None
            )
        else:
            board_stack = game.board.build_tensor(int(current_player.color))
            chosen_move = current_player.choose(board_stack, dice, valid_moves)

        if chosen_move is None:
            chosen_move = random.choice(valid_moves)

        # Execute move
        result = game.apply_move(chosen_move)

        desc = f"Player {state.current_player_index} rolled {dice}: {self.serialize_move(state.current_player_index, chosen_move, result)}"

        # Check if player won - game ends immediately
        if current_player.check_won():
            state.game_over = True
            state.winner_index = state.current_player_index
            desc += f" | Player {state.current_player_index} has WON! GAME OVER!"
            return game, state, desc, self.game_state_tokens(game), [], False

        # Move to next player if no extra turn
        if not result.extra_turn and not state.game_over:
            total = len(game.players)
            state.current_player_index = (state.current_player_index + 1) % total

        return game, state, desc, self.game_state_tokens(game), [], False
