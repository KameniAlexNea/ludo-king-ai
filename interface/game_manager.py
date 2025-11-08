from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np

from ludo_rl.ludo.config import config
from ludo_rl.ludo.game import LudoGame, TurnOutcome
from ludo_rl.strategy.features import build_move_options
from ludo_rl.strategy.types import MoveOption

from .types import (
    GameConstants,
    INDEX_TO_COLOR,
    PlayerColor,
    SessionState,
    StepResult,
    TokenState,
    TokenView,
    absolute_to_path_index,
    token_state_from_position,
)


class GameManager:
    """Coordinates interactive games built on the refactored Ludo engine."""

    def __init__(
        self,
        default_players: List[PlayerColor],
        available_strategies: List[str],
        show_token_ids: bool,
    ) -> None:
        self.default_players = default_players
        self.available_strategies = list(dict.fromkeys(available_strategies))
        self.available_strategy_set = {name.lower() for name in self.available_strategies}
        self.show_token_ids = show_token_ids
        self._rng = random.Random()

    # ------------------------------------------------------------------
    # Session lifecycle helpers
    # ------------------------------------------------------------------
    def init_game(
        self, strategies: List[str], *, seed: Optional[int] = None
    ) -> SessionState:
        """Initialise a new Ludo session with the requested strategies."""

        game = LudoGame()
        if seed is not None:
            game.rng.seed(seed)

        for idx, player in enumerate(game.players):
            player.reset()
            requested = strategies[idx] if idx < len(strategies) else "random"
            strategy_name = self._normalize_strategy(requested)
            player.strategy_name = strategy_name
            player._strategy = None  # ensure fresh instance next decision

        return SessionState(game=game, current_player_index=0, turn_counter=0)

    def game_state_tokens(self, session: SessionState) -> Dict[PlayerColor, List[TokenView]]:
        """Capture the current board state as TokenView objects."""

        token_map: Dict[PlayerColor, List[TokenView]] = {
            color: [] for color in INDEX_TO_COLOR.values()
        }

        for idx, player in enumerate(session.game.players):
            color = INDEX_TO_COLOR[idx]
            tokens = [self._token_view_for_piece(session, idx, piece) for piece in player.pieces]
            token_map[color].extend(sorted(tokens, key=lambda t: t.token_id))

        return token_map

    def is_human_turn(self, session: SessionState) -> bool:
        return session.current_player().strategy_name.lower() == "human"

    # ------------------------------------------------------------------
    # Core turn progression
    # ------------------------------------------------------------------
    def play_step(
        self,
        session: SessionState,
        human_move_choice: Optional[int] = None,
        dice: Optional[int] = None,
    ) -> StepResult:
        """Advance the game by a single decision opportunity."""

        if session.game_over:
            tokens = self.game_state_tokens(session)
            return StepResult(
                session=session,
                description="Game over",
                tokens=tokens,
                move_options=[],
                waiting_for_human=False,
                dice_roll=dice or 0,
            )

        player_index = session.current_player_index
        player = session.game.players[player_index]
        color = INDEX_TO_COLOR[player_index]

        used_dice = int(dice) if dice is not None else session.game.roll_dice()
        valid_moves = session.game.get_valid_moves(player_index, used_dice)

        if not valid_moves:
            description = self._describe_no_move(color, used_dice, player)
            advance = used_dice != 6
            if advance:
                self._advance_player(session)
            session.turn_counter += 1
            tokens = self.game_state_tokens(session)
            self._finalise_if_needed(session)
            return StepResult(
                session=session,
                description=description,
                tokens=tokens,
                move_options=[],
                waiting_for_human=False,
                dice_roll=used_dice,
            )

        context_map = self._build_move_context(session, player_index, used_dice, valid_moves)
        is_human = player.strategy_name.lower() == "human"

        if is_human and human_move_choice is None:
            move_opts = self._format_move_options(valid_moves, context_map)
            tokens = self.game_state_tokens(session)
            description = (
                f"{color.display_name} rolled {used_dice} - choose your move"
            )
            return StepResult(
                session=session,
                description=description,
                tokens=tokens,
                move_options=move_opts,
                waiting_for_human=True,
                dice_roll=used_dice,
            )

        if is_human:
            chosen_move = next(
                (move for move in valid_moves if move["piece"].piece_id == human_move_choice),
                None,
            )
            if chosen_move is None:
                chosen_move = valid_moves[0]
        else:
            decision = player.decide(
                session.game.build_board_tensor(player_index),
                used_dice,
                valid_moves,
            )
            chosen_move = decision if decision is not None else valid_moves[0]

        if chosen_move.get("dice_roll") != used_dice:
            chosen_move = dict(chosen_move)
            chosen_move["dice_roll"] = used_dice

        outcome = session.game.take_turn(
            player_index, dice_roll=used_dice, move=chosen_move, rng=self._rng
        )

        if not outcome.skipped and player.has_won():
            session.winner_index = player_index

        if not outcome.extra_turn:
            self._advance_player(session)

        session.turn_counter += 1
        self._finalise_if_needed(session)

        tokens = self.game_state_tokens(session)
        description = self._serialize_outcome(color, used_dice, outcome)
        if session.game_over and session.winner_color is not None:
            description += f" | WINNER: {session.winner_color.display_name}"

        return StepResult(
            session=session,
            description=description,
            tokens=tokens,
            move_options=[],
            waiting_for_human=False,
            dice_roll=used_dice,
            outcome=outcome,
        )

    # ------------------------------------------------------------------
    # Tournament helpers (used by simulation tab)
    # ------------------------------------------------------------------
    def determine_rankings(
        self, session: SessionState, finish_order: List[int]
    ) -> List[int]:
        ordered = finish_order.copy()
        remaining = [idx for idx in range(config.NUM_PLAYERS) if idx not in ordered]
        remaining.sort(
            key=lambda idx: (
                self._finished_pieces(session, idx),
                self._player_progress(session, idx),
            ),
            reverse=True,
        )
        ordered.extend(remaining)
        return ordered[: config.NUM_PLAYERS]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_strategy(self, strategy_name: str) -> str:
        if not strategy_name:
            return "random"
        lowered = strategy_name.lower()
        if lowered in self.available_strategy_set:
            return lowered
        return "random"

    def _advance_player(self, session: SessionState) -> None:
        session.current_player_index = (session.current_player_index + 1) % config.NUM_PLAYERS

    def _finalise_if_needed(self, session: SessionState) -> None:
        if session.winner_index is not None:
            return
        if session.turn_counter < config.MAX_TURNS:
            return
        # Treat exhaustion as a draw; winner remains None

    def _token_view_for_piece(
        self, session: SessionState, player_index: int, piece
    ) -> TokenView:
        rel_pos = piece.position
        state = token_state_from_position(rel_pos)
        abs_pos = None
        marker = rel_pos

        if state is TokenState.MAIN:
            abs_pos = session.game.board.absolute_position(player_index, rel_pos)
            if abs_pos != -1:
                marker = absolute_to_path_index(abs_pos)
        elif state is TokenState.HOME_COLUMN:
            marker = max(rel_pos, GameConstants.HOME_COLUMN_START)

        return TokenView(
            token_id=piece.piece_id,
            position=marker,
            state=state,
            relative_position=rel_pos,
            absolute_position=abs_pos,
        )

    def _build_move_context(
        self,
        session: SessionState,
        player_index: int,
        dice: int,
        valid_moves: List[dict],
    ) -> Dict[int, MoveOption]:
        action_mask = np.zeros(config.PIECES_PER_PLAYER, dtype=bool)
        move_choices: List[Optional[dict]] = [None] * config.PIECES_PER_PLAYER
        for move in valid_moves:
            piece_id = move["piece"].piece_id
            action_mask[piece_id] = True
            move_choices[piece_id] = move

        context = build_move_options(
            session.game.build_board_tensor(player_index),
            dice,
            action_mask,
            move_choices,
        )
        return {option.piece_id: option for option in context.moves}

    def _format_move_options(
        self, valid_moves: List[dict], option_map: Dict[int, MoveOption]
    ) -> List[dict]:
        options: List[dict] = []
        for move in valid_moves:
            piece = move["piece"]
            option = option_map.get(piece.piece_id)
            description = self._format_move_description(piece.position, move["new_pos"], option)
            options.append(
                {
                    "token_id": piece.piece_id,
                    "description": description,
                    "move_type": self._classify_move(option),
                }
            )
        return options

    def _format_move_description(
        self, current: int, new: int, option: Optional[MoveOption]
    ) -> str:
        desc = f"{current} → {new}"
        if option is None:
            return desc
        extras: List[str] = []
        if option.can_capture:
            extras.append(f"capture x{option.capture_count}")
        if option.enters_home:
            extras.append("finish")
        elif option.enters_safe_zone:
            extras.append("safe")
        if option.extra_turn and not option.enters_home and not option.can_capture:
            extras.append("extra turn")
        if option.forms_blockade:
            extras.append("blockade")
        if extras:
            desc += " (" + ", ".join(extras) + ")"
        return desc

    def _classify_move(self, option: Optional[MoveOption]) -> str:
        if option is None:
            return "advance"
        if option.can_capture:
            return "capture"
        if option.enters_home:
            return "finish"
        if option.forms_blockade:
            return "blockade"
        if option.enters_safe_zone:
            return "safe"
        if option.extra_turn:
            return "extra"
        return "advance"

    def _serialize_outcome(
        self, color: PlayerColor, dice: int, outcome: TurnOutcome
    ) -> str:
        if outcome.skipped or outcome.move is None:
            return f"{color.display_name} rolled {dice}: no move"

        move = outcome.move
        piece = move["piece"]
        new_pos = move["new_pos"]
        description = f"{color.display_name} rolled {dice}: token {piece.piece_id} → {new_pos}"
        events = outcome.result.get("events", {}) if outcome.result else {}
        extras: List[str] = []
        if events.get("knockouts"):
            extras.append(f"captured {len(events['knockouts'])}")
        if events.get("finished"):
            extras.append("finished")
        if outcome.extra_turn:
            extras.append("extra turn")
        if extras:
            description += " (" + ", ".join(extras) + ")"
        return description

    def _describe_no_move(
        self, color: PlayerColor, dice: int, player
    ) -> str:
        token_positions = ", ".join(
            [f"token {piece.piece_id}: {piece.position}" for piece in player.pieces]
        )
        suffix = " (extra turn)" if dice == 6 else ""
        return (
            f"{color.display_name} rolled {dice} - no moves{suffix} | Positions: "
            f"{token_positions}"
        )

    def _finished_pieces(self, session: SessionState, idx: int) -> int:
        return sum(piece.position == 57 for piece in session.game.players[idx].pieces)

    def _player_progress(self, session: SessionState, idx: int) -> int:
        return sum(piece.position for piece in session.game.players[idx].pieces)