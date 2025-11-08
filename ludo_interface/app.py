import os
from typing import List, Optional

os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(os.getcwd(), "gradio_runtime"))
os.environ.setdefault(
    "GRADIO_CACHE_DIR",
    os.path.join(os.getcwd(), "gradio_runtime", "cache"),
)


from ludo_rl.strategy.registry import available as get_available_strategies
from .board_viz import preload_board_template

from .event_handler import EventHandler
from .game_manager import GameManager
from .ui_builder import UIBuilder
from .utils import Utils
from .models import PlayerColor, ALL_COLORS

AI_STRATEGIES = list(get_available_strategies().keys())
DEFAULT_PLAYERS = ALL_COLORS


class LudoApp:
    """Encapsulates the Ludo game application logic and Gradio UI."""

    def __init__(
        self, players: Optional[List[PlayerColor]] = None, show_token_ids: bool = True
    ):
        """
        Initializes the Ludo application.

        Args:
            players (Optional[List[PlayerColor]]): A list of player colors. Defaults to standard four players.
            show_token_ids (bool): Whether to display token IDs on the board.
        """
        self.default_players = players if players is not None else DEFAULT_PLAYERS
        self.show_token_ids = show_token_ids
        self.ai_strategies = list(get_available_strategies().keys())

        # Initialize components
        self.game_manager = GameManager(self.default_players, self.show_token_ids)
        self.utils = Utils()
        self.event_handler = EventHandler(
            self.game_manager,
            self.utils,
            self.ai_strategies,
            self.default_players,
            self.show_token_ids,
        )
        self.ui_builder = UIBuilder(
            self.ai_strategies,
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