from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from ludo_rl.ludo.config import config
from ludo_rl.ludo.game import LudoGame, TurnOutcome


class PlayerColor(Enum):
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    BLUE = "blue"

    @property
    def display_name(self) -> str:
        return self.value.capitalize()


ALL_COLORS: List[PlayerColor] = [
    PlayerColor.RED,
    PlayerColor.GREEN,
    PlayerColor.YELLOW,
    PlayerColor.BLUE,
]
INDEX_TO_COLOR: Dict[int, PlayerColor] = {idx: color for idx, color in enumerate(ALL_COLORS)}
COLOR_TO_INDEX: Dict[PlayerColor, int] = {color: idx for idx, color in INDEX_TO_COLOR.items()}


class TokenState(Enum):
    HOME = "home"
    MAIN = "main"
    HOME_COLUMN = "home_column"
    FINISHED = "finished"


@dataclass(slots=True)
class TokenView:
    token_id: int
    position: int
    state: TokenState
    relative_position: int
    absolute_position: int | None = None


@dataclass(slots=True)
class SessionState:
    game: LudoGame
    current_player_index: int = 0
    turn_counter: int = 0
    winner_index: int | None = None

    @property
    def game_over(self) -> bool:
        return self.winner_index is not None or self.turn_counter >= config.MAX_TURNS

    @property
    def winner_color(self) -> Optional[PlayerColor]:
        if self.winner_index is None or self.winner_index not in INDEX_TO_COLOR:
            return None
        return INDEX_TO_COLOR[self.winner_index]

    def current_player(self):
        return self.game.players[self.current_player_index]


@dataclass(slots=True)
class StepResult:
    session: SessionState
    description: str
    tokens: Dict[PlayerColor, List[TokenView]]
    move_options: List[dict]
    waiting_for_human: bool
    dice_roll: int
    outcome: Optional[TurnOutcome] = None


PATH_LENGTH = 52
ABSOLUTE_SEQUENCE = list(range(1, PATH_LENGTH + 1))
ABSOLUTE_TO_PATH_INDEX: Dict[int, int] = {
    abs_pos: idx for idx, abs_pos in enumerate(ABSOLUTE_SEQUENCE)
}
PATH_INDEX_TO_ABSOLUTE: Dict[int, int] = {
    idx: abs_pos for abs_pos, idx in ABSOLUTE_TO_PATH_INDEX.items()
}


def absolute_to_path_index(abs_pos: int) -> int:
    try:
        return ABSOLUTE_TO_PATH_INDEX[abs_pos]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown absolute position: {abs_pos}") from exc


def path_index_to_absolute(idx: int) -> int:
    try:
        return PATH_INDEX_TO_ABSOLUTE[idx]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown path index: {idx}") from exc


class GameConstants:
    HOME_COLUMN_START = 52
    HOME_COLUMN_SIZE = 6  # 52..57 inclusive (57 treated as final finish)
    FINISH_POSITION = HOME_COLUMN_START + HOME_COLUMN_SIZE - 1


class BoardConstants:
    START_POSITIONS: Dict[PlayerColor, int]
    HOME_COLUMN_ENTRIES: Dict[PlayerColor, int]
    STAR_SQUARES: set[int]


def _compute_board_constants() -> tuple[Dict[PlayerColor, int], Dict[PlayerColor, int], set[int]]:
    game = LudoGame()
    start_positions: Dict[PlayerColor, int] = {}
    home_entries: Dict[PlayerColor, int] = {}

    for idx, color in INDEX_TO_COLOR.items():
        abs_start = config.PLAYER_START_SQUARES[idx]
        # Map absolute board position to visual path index (0-51)
        start_positions[color] = absolute_to_path_index(abs_start)

        abs_entry = game.board.absolute_position(idx, 51)
        # Add +1 because HOME_COLUMN_ENTRIES should point to where position 52 starts (the entry to home column)
        home_entries[color] = absolute_to_path_index(abs_entry) + 1

    star_squares = {absolute_to_path_index(pos) for pos in config.SAFE_SQUARES_ABS}
    return start_positions, home_entries, star_squares


(
    BoardConstants.START_POSITIONS,
    BoardConstants.HOME_COLUMN_ENTRIES,
    BoardConstants.STAR_SQUARES,
) = _compute_board_constants()


def token_state_from_position(relative_position: int) -> TokenState:
    if relative_position == 0:
        return TokenState.HOME
    if 1 <= relative_position <= 51:
        return TokenState.MAIN
    if 52 <= relative_position <= 56:
        return TokenState.HOME_COLUMN
    if relative_position >= 57:
        return TokenState.FINISHED
    return TokenState.MAIN


__all__ = [
    "ALL_COLORS",
    "BoardConstants",
    "COLOR_TO_INDEX",
    "GameConstants",
    "INDEX_TO_COLOR",
    "PlayerColor",
    "TokenState",
    "TokenView",
    "SessionState",
    "StepResult",
    "absolute_to_path_index",
    "path_index_to_absolute",
    "token_state_from_position",
]
