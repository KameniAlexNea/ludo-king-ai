from typing import List

import gradio as gr

from ..event_handler import EventHandler
from ..models import PlayerColor
from .llm_config_ui import StrategyConfigManager


class SimulationTabView:
    """Builds the 'Simulate Multiple Games' tab UI."""

    def __init__(
        self,
        ai_strategies: List[str],
        default_players: List[PlayerColor],
        handler: EventHandler,
        strategy_config_manager: StrategyConfigManager,
        strategy_dropdowns: List,
    ) -> None:
        self.ai_strategies = ai_strategies
        self.default_players = default_players
        self.handler = handler
        self.strategy_config_manager = strategy_config_manager
        self.strategy_dropdowns = strategy_dropdowns

    def build(self) -> None:
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
