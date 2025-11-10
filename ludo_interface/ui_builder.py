from typing import List

import gradio as gr

from .event_handler import HISTORY_LIMIT, EventHandler
from .llm_config_ui import (
    StrategyConfigManager,
    create_llm_config_ui,
    create_rl_config_ui,
)
from .models import PlayerColor


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
        """Builds the 'Play Game' tab of the UI."""
        # Main layout: Board dominates the center, controls in compact sidebars
        with gr.Row():
            # Left sidebar: Player config and game controls (compact)
            with gr.Column(scale=1, min_width=280):
                with gr.Accordion("👥 Players", open=True):
                    strategy_inputs = [
                        gr.Dropdown(
                            choices=self.ai_strategies + ["empty"],
                            value=(
                                "human"
                                if i == 0
                                else (
                                    self.ai_strategies[1]
                                    if len(self.ai_strategies) > 1
                                    else self.ai_strategies[0]
                                )
                            ),
                            label="🔴🟢🟡🔵"[i] + f" {color.name.title()} Strategy",
                            container=True,
                            scale=1,
                        )
                        for i, color in enumerate(self.default_players)
                    ]
                    self.strategy_dropdowns.extend(strategy_inputs)

                with gr.Accordion("🎮 Controls", open=True):
                    init_btn = gr.Button("🎮 New Game", variant="primary", size="sm")
                    random_btn = gr.Button("🎲 Random", size="sm")
                    with gr.Row():
                        step_btn = gr.Button("▶️ Step", size="sm", scale=1)
                        auto_steps_n = gr.Number(
                            value=100, minimum=1, maximum=1000, container=False, scale=1
                        )
                    with gr.Row():
                        run_auto_btn = gr.Button("🔄 Auto", size="sm", scale=1)
                        auto_delay = gr.Number(
                            value=0.5,
                            minimum=0,
                            maximum=5,
                            step=0.1,
                            container=False,
                            scale=1,
                        )

                with gr.Accordion("⚙️ Options", open=False):
                    show_ids = gr.Checkbox(
                        label="Show Token IDs",
                        value=self.show_token_ids,
                        container=False,
                    )
                    export_btn = gr.Button("📤 Export", size="sm")
                    move_history_btn = gr.Button("📜 History", size="sm")

            # Center: Main game board (large, no scrolling)
            with gr.Column(scale=3):
                board_plot = gr.HTML(
                    label="🎯 Game Board", elem_classes=["board-container"]
                )

                # Human move controls (overlay when needed)
                with gr.Row(visible=False) as human_controls:
                    with gr.Column():
                        human_moves_display = gr.HTML()
                        with gr.Row():
                            move_buttons = [
                                gr.Button(
                                    f"Token {i}",
                                    visible=False,
                                    variant="secondary",
                                    size="sm",
                                )
                                for i in range(4)
                            ]

            # Right sidebar: Game info and stats (compact)
            with gr.Column(scale=1, min_width=280):
                # Current player when not in human turn
                with gr.Row():
                    current_player_display = gr.HTML(
                        value="<h3>🎯 Current Player: Game not started</h3>"
                    )

                with gr.Accordion("📝 Last Action", open=True):
                    log = gr.Textbox(
                        show_label=False,
                        interactive=False,
                        lines=3,
                        max_lines=4,
                        container=False,
                    )

                with gr.Accordion("📊 Statistics", open=True):
                    stats_display = gr.JSON(
                        show_label=False,
                        container=False,
                        value={
                            "games": 0,
                            "wins": {c.name: 0 for c in self.default_players},
                        },
                    )

                with gr.Accordion("📚 History", open=False):
                    history_box = gr.Textbox(
                        show_label=False,
                        lines=6,
                        max_lines=10,
                        container=False,
                        show_copy_button=True,
                    )

        # Hidden elements
        export_box = gr.Textbox(label="Game State JSON", lines=6, visible=False)

        # Event Handlers
        init_btn.click(
            self.handler._ui_init,
            strategy_inputs,
            [
                game_state,
                state_state,
                board_plot,
                log,
                move_history,
                stats_state,
                current_player_display,
                human_controls,
                human_moves_display,
            ]
            + move_buttons
            + [
                human_move_options,
                pending_dice,
                selected_token_id,
                auto_steps_remaining,
                auto_delay_state,
            ],
        )
        random_btn.click(
            self.handler._ui_random_strategies, outputs=strategy_inputs
        ).then(
            self.handler._ui_init,
            strategy_inputs,
            [
                game_state,
                state_state,
                board_plot,
                log,
                move_history,
                stats_state,
                current_player_display,
                human_controls,
                human_moves_display,
            ]
            + move_buttons
            + [
                human_move_options,
                pending_dice,
                selected_token_id,
                auto_steps_remaining,
                auto_delay_state,
            ],
        )
        step_btn.click(
            self.handler._ui_steps,
            [game_state, state_state, move_history, show_ids, pending_dice],
            [
                game_state,
                state_state,
                board_plot,
                log,
                move_history,
                waiting_for_human,
                current_player_display,
                human_moves_display,
                human_controls,
            ]
            + move_buttons
            + [
                human_move_options,
                pending_dice,
                selected_token_id,
                auto_steps_remaining,
                auto_delay_state,
            ],
        ).then(
            self.handler._ui_update_stats,
            [stats_state, game_state, state_state],
            [stats_state],
        ).then(lambda s: s, [stats_state], [stats_display])

        for i, btn in enumerate(move_buttons):
            btn.click(
                lambda opts, idx=i: opts[idx]["piece_id"] if idx < len(opts) else None,
                [human_move_options],
                [selected_token_id],
            ).then(
                self.handler._ui_make_human_move,
                [
                    selected_token_id,
                    game_state,
                    state_state,
                    move_history,
                    show_ids,
                    human_move_options,
                    pending_dice,
                    auto_steps_remaining,
                    auto_delay_state,
                ],
                [
                    game_state,
                    state_state,
                    board_plot,
                    log,
                    move_history,
                    waiting_for_human,
                    current_player_display,
                    human_moves_display,
                    human_controls,
                ]
                + move_buttons
                + [
                    human_move_options,
                    pending_dice,
                    selected_token_id,
                    auto_steps_remaining,
                    auto_delay_state,
                ],
            ).then(
                self.handler._ui_update_stats,
                [stats_state, game_state, state_state],
                [stats_state],
            ).then(lambda s: s, [stats_state], [stats_display]).then(
                self.handler._ui_resume_auto,
                [
                    auto_steps_remaining,
                    auto_delay_state,
                    game_state,
                    state_state,
                    move_history,
                    show_ids,
                ],
                [
                    game_state,
                    state_state,
                    board_plot,
                    log,
                    move_history,
                    waiting_for_human,
                    current_player_display,
                    human_moves_display,
                    human_controls,
                ]
                + move_buttons
                + [
                    human_move_options,
                    pending_dice,
                    selected_token_id,
                    auto_steps_remaining,
                    auto_delay_state,
                ],
            ).then(
                self.handler._ui_update_stats,
                [stats_state, game_state, state_state],
                [stats_state],
            ).then(lambda s: s, [stats_state], [stats_display])

        run_auto_btn.click(
            self.handler._ui_run_auto,
            [auto_steps_n, auto_delay, game_state, state_state, move_history, show_ids],
            [
                game_state,
                state_state,
                board_plot,
                log,
                move_history,
                waiting_for_human,
                current_player_display,
                human_moves_display,
                human_controls,
            ]
            + move_buttons
            + [
                human_move_options,
                pending_dice,
                selected_token_id,
                auto_steps_remaining,
                auto_delay_state,
            ],
        ).then(
            self.handler._ui_update_stats,
            [stats_state, game_state, state_state],
            [stats_state],
        ).then(lambda s: s, [stats_state], [stats_display])

        move_history_btn.click(
            lambda h: "\n".join(h[-HISTORY_LIMIT:]), [move_history], [history_box]
        )
        export_btn.click(
            self.handler._ui_export, [game_state, state_state], [export_box]
        )

    def _build_simulation_tab(self):
        """Builds the 'Simulate Multiple Games' tab of the UI."""
        with gr.Row():
            # Left sidebar: Tournament configuration
            with gr.Column(scale=1, min_width=320):
                with gr.Accordion("🏆 Tournament Setup", open=True):
                    gr.Markdown("**Configure AI strategies for bulk simulation**")
                    sim_strat_inputs = [
                        gr.Dropdown(
                            choices=[s for s in self.ai_strategies if s != "human"]
                            + ["empty"],
                            value=[s for s in self.ai_strategies if s != "human"][
                                min(
                                    i,
                                    len([s for s in self.ai_strategies if s != "human"])
                                    - 1,
                                )
                            ],
                            label="🔴🟢🟡🔵"[i] + f" {color.name.title()} Strategy",
                            container=True,
                        )
                        for i, color in enumerate(self.default_players)
                    ]
                    self.strategy_dropdowns.extend(sim_strat_inputs)

                with gr.Accordion("⚙️ Simulation Parameters", open=True):
                    bulk_games = gr.Slider(
                        10,
                        5000,
                        value=500,
                        step=50,
                        label="Number of Games",
                        info="More games = more accurate statistics",
                    )
                    with gr.Row():
                        preset_100 = gr.Button(
                            "100 Games", size="sm", variant="secondary"
                        )
                        preset_500 = gr.Button(
                            "500 Games", size="sm", variant="secondary"
                        )
                        preset_1000 = gr.Button(
                            "1K Games", size="sm", variant="secondary"
                        )
                        preset_2000 = gr.Button(
                            "2K Games", size="sm", variant="secondary"
                        )

                with gr.Accordion("🎮 Quick Actions", open=True):
                    with gr.Row():
                        randomize_btn = gr.Button("🎲 Randomize All", size="sm")
                        reset_btn = gr.Button("🔄 Reset to Default", size="sm")
                    bulk_run_btn = gr.Button(
                        "🚀 Run Tournament", variant="primary", size="lg"
                    )
                    gr.Button(
                        "⏹️ Stop Simulation", variant="stop", size="sm", visible=False
                    )

            # Right side: Results and statistics
            with gr.Column(scale=2):
                with gr.Accordion("📊 Tournament Results", open=True):
                    # Progress indicator
                    gr.Progress()
                    simulation_status = gr.HTML(
                        value="<p>🎯 Ready to run tournament simulation</p>"
                    )

                    # Results display
                    with gr.Tabs():
                        with gr.TabItem("📈 Summary"):
                            bulk_results = gr.JSON(
                                label="Win Statistics",
                                show_label=False,
                                value={"status": "No simulation run yet"},
                            )

                        with gr.TabItem("📋 Detailed Results"):
                            detailed_results = gr.Textbox(
                                label="Detailed Statistics",
                                lines=12,
                                max_lines=20,
                                show_label=False,
                                placeholder="Detailed results will appear here after simulation...",
                            )

                        with gr.TabItem("📊 Win Rate Chart"):
                            chart_placeholder = gr.HTML(
                                value="<div style='text-align:center;padding:40px;color:#666;'>📊 Win rate visualization will appear here after simulation</div>"
                            )

                with gr.Accordion("💡 Simulation Tips", open=False):
                    gr.Markdown(
                        """
                    **Optimization Tips:**
                    - **100-500 games**: Quick testing of strategies
                    - **500-1000 games**: Reliable performance comparison  
                    - **1000+ games**: Statistical significance for research
                    
                    **Strategy Notes:**
                    - Different AI strategies have varying computational costs
                    - Larger simulations provide more accurate win rate estimates
                    - Compare multiple runs for consistency validation
                    """
                    )

        # Event handlers for presets
        preset_100.click(lambda: 100, outputs=[bulk_games])
        preset_500.click(lambda: 500, outputs=[bulk_games])
        preset_1000.click(lambda: 1000, outputs=[bulk_games])
        preset_2000.click(lambda: 2000, outputs=[bulk_games])

        # Randomize strategies
        randomize_btn.click(
            self.handler._ui_random_strategies, outputs=sim_strat_inputs
        )

        # Reset to default (first non-human strategy for all)
        reset_btn.click(
            lambda: [[s for s in self.ai_strategies if s != "human"][0]]
            * len(self.default_players),
            outputs=sim_strat_inputs,
        )

        # Main simulation handler
        bulk_run_btn.click(
            self.handler._ui_run_bulk,
            [bulk_games] + sim_strat_inputs,
            [bulk_results, detailed_results, simulation_status, chart_placeholder],
        )
