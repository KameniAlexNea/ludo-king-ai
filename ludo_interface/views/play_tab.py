from typing import List

import gradio as gr

from ..event_handler import HISTORY_LIMIT, EventHandler
from ..models import PlayerColor
from .llm_config_ui import StrategyConfigManager


class PlayTabView:
    """Builds the 'Play Game' tab UI."""

    def __init__(
        self,
        ai_strategies: List[str],
        default_players: List[PlayerColor],
        show_token_ids: bool,
        handler: EventHandler,
        strategy_config_manager: StrategyConfigManager,
        strategy_dropdowns: List,
    ) -> None:
        self.ai_strategies = ai_strategies
        self.default_players = default_players
        self.show_token_ids = show_token_ids
        self.handler = handler
        self.strategy_config_manager = strategy_config_manager
        self.strategy_dropdowns = strategy_dropdowns

    def build(
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
    ) -> None:
        """Compose the Play Game tab within the current Gradio context."""
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
