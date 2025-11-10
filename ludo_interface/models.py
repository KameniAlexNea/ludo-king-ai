from enum import Enum

from ludo_rl.ludo_king.config import config


class PlayerColor(Enum):
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    BLUE = "blue"


ALL_COLORS = [PlayerColor.RED, PlayerColor.GREEN, PlayerColor.YELLOW, PlayerColor.BLUE]

# Viz Variables
PLAYER_START_SQUARES = {
    color: start for color, start in zip(ALL_COLORS, config.PLAYER_START_SQUARES)
}
HOME_ENTRY = {color: entry for color, entry in zip(ALL_COLORS, config.HOME_ENTRY)}
PTOPlayerColor = {i: color for i, color in enumerate(ALL_COLORS)}
