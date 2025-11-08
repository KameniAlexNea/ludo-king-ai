from __future__ import annotations

import os

from typing import List, Optional, Sequence

os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(os.getcwd(), "gradio_runtime"))
os.environ.setdefault(
    "GRADIO_CACHE_DIR",
    os.path.join(os.getcwd(), "gradio_runtime", "cache"),
)

from ludo_rl.strategy import available

from .event_handler import EventHandler
from .game_manager import GameManager
from .ui_builder import UIBuilder
from .utils import Utils
from .board_viz import preload_board_template
from .types import ALL_COLORS, PlayerColor


def _default_strategy_pool() -> List[str]:
    names = ["human", "random"]
    for strategy_name in sorted(available().keys()):
        if strategy_name not in names:
            names.append(strategy_name)
    return names

DEFAULT_PLAYERS = list(ALL_COLORS)
DEFAULT_STRATEGIES = _default_strategy_pool()


class LudoApp:
    """Encapsulates the Ludo game application logic and Gradio UI."""

    def __init__(
        self,
        players: Optional[Sequence[PlayerColor]] = None,
        strategies: Optional[Sequence[str]] = None,
        show_token_ids: bool = True,
    ):
        """
        Initializes the Ludo application.

        Args:
            players (Optional[Sequence[PlayerColor]]): Custom player order. Defaults to engine order.
            strategies (Optional[Sequence[str]]): Strategy names to expose in the UI.
            show_token_ids (bool): Whether to display token IDs on the board.
        """
        self.default_players = (
            list(players) if players is not None else list(DEFAULT_PLAYERS)
        )
        self.show_token_ids = show_token_ids

        strategy_source = list(strategies) if strategies is not None else list(DEFAULT_STRATEGIES)
        combined = ["human", "random"] + [name.lower() for name in strategy_source]
        seen = set()
        self.strategy_options: List[str] = []
        for name in combined:
            lowered = name.lower()
            if lowered not in seen:
                self.strategy_options.append(lowered)
                seen.add(lowered)

        # Initialize components with the refactored engine
        self.game_manager = GameManager(
            self.default_players, self.strategy_options, self.show_token_ids
        )
        self.utils = Utils()
        self.event_handler = EventHandler(
            self.game_manager,
            self.utils,
            self.strategy_options,
            self.default_players,
            self.show_token_ids,
        )
        self.ui_builder = UIBuilder(
            self.strategy_options,
            self.default_players,
            self.show_token_ids,
            self.event_handler,
        )

        # Preload assets
        preload_board_template()
        print("🚀 Initializing Enhanced Ludo Game...")

    def create_ui(self):
        """Creates and returns the Gradio UI for the Ludo game."""
        return self.ui_builder.create_ui()

    def launch(self, server_name="0.0.0.0", server_port=7860, **kwargs):
        """Launches the Gradio application."""
        demo = self.create_ui()
        demo.launch(server_name=server_name, server_port=server_port, **kwargs)


def launch_app():
    """Main entry point for the application."""
    return LudoApp()


if __name__ == "__main__":
    launch_app().launch(share=False, inbrowser=True, show_error=True)