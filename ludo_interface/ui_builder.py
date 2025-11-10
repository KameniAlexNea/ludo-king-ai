from typing import List

import gradio as gr

from .event_handler import EventHandler
from .models import PlayerColor
from .views.llm_config_ui import (
    StrategyConfigManager,
    create_llm_config_ui,
    create_rl_config_ui,
)
from .views.play_tab import PlayTabView
from .views.simulation_tab import SimulationTabView


class UIBuilder:
    """Handles Gradio UI construction and layout."""

    def __init__(
        self,
        ai_strategies: List[str],
        default_players: List[PlayerColor],
        show_token_ids: bool,
        handler: EventHandler,
        strategy_config_manager: StrategyConfigManager,
    ):
        self.ai_strategies = ai_strategies
        self.default_players = default_players
        self.show_token_ids = show_token_ids
        self.handler = handler  # The object with event handler methods (e.g., LudoApp)
        self.strategy_config_manager = strategy_config_manager
        self.strategy_dropdowns = []  # To be populated with strategy dropdown components

    def create_ui(self):
        """Creates and returns the Gradio UI for the Ludo game."""
        with gr.Blocks(
            title="🎲 Enhanced Ludo AI Visualizer",
            theme=gr.themes.Soft(),
            css="""
            .board-container {
                max-height: 80vh !important;
                overflow: hidden !important;
            }
            .board-container img {
                max-width: 100% !important;
                max-height: 80vh !important;
                object-fit: contain !important;
            }
            .gradio-accordion {
                margin: 0.25rem 0 !important;
            }
            .gradio-box {
                padding: 0.5rem !important;
                margin: 0.25rem 0 !important;
            }
            """,
        ) as demo:
            game_state = gr.State()
            state_state = gr.State()  # GameState object
            move_history = gr.State([])
            stats_state = gr.State(
                {"games": 0, "wins": {c.value: 0 for c in self.default_players}}
            )
            waiting_for_human = gr.State(False)
            human_move_options = gr.State([])
            # Persist the dice rolled when auto-play pauses for a human turn
            pending_dice = gr.State(None)
            # Holds the token id chosen by the human via a button
            selected_token_id = gr.State(None)
            # Track remaining auto steps and delay to allow resume after human move
            auto_steps_remaining = gr.State(0)
            auto_delay_state = gr.State(0.5)

            with gr.Tabs():
                with gr.TabItem("🎮 Play Game"):
                    self._build_play_game_tab(
                        game_state,
                        state_state,
                        move_history,
                        stats_state,
                        waiting_for_human,
                        human_move_options,
                        pending_dice,
                        selected_token_id,
                        auto_steps_remaining,
                        auto_delay_state,
                    )
                with gr.TabItem("🏆 Simulate Multiple Games"):
                    self._build_simulation_tab()
                with gr.TabItem("⚙️ Configure Strategies"):
                    create_llm_config_ui(
                        self.strategy_config_manager,
                        self.ai_strategies,
                        self.strategy_dropdowns,
                    )
                    create_rl_config_ui(
                        self.strategy_config_manager,
                        self.ai_strategies,
                        self.strategy_dropdowns,
                    )

            gr.Markdown(
                """
            ## 🎲 Enhanced Ludo AI Visualizer
            **Features:** 🤖 Multiple AI Strategies • 👤 Human Players • 🎨 Enhanced Graphics • 📊 Statistics
            """
            )
        return demo

    def _build_play_game_tab(
        self,
        game_state,
        state_state,
        move_history,
        stats_state,
        waiting_for_human,
        human_move_options,
        pending_dice,
        selected_token_id,
        auto_steps_remaining,
        auto_delay_state,
    ):
        view = PlayTabView(
            self.ai_strategies,
            self.default_players,
            self.show_token_ids,
            self.handler,
            self.strategy_config_manager,
            self.strategy_dropdowns,
        )
        view.build(
            game_state,
            state_state,
            move_history,
            stats_state,
            waiting_for_human,
            human_move_options,
            pending_dice,
            selected_token_id,
            auto_steps_remaining,
            auto_delay_state,
        )

    def _build_simulation_tab(self):
        view = SimulationTabView(
            self.ai_strategies,
            self.default_players,
            self.handler,
            self.strategy_config_manager,
            self.strategy_dropdowns,
        )
        view.build()
