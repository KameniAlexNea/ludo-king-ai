"""UI components for configuring LLM and RL strategies dynamically."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import gradio as gr


@dataclass
class LLMProviderConfig:
    """Configuration for an LLM provider."""

    provider: str  # ollama, openai, gemini, grok, claude, etc.
    model_name: str
    api_key: Optional[str] = None
    label: str = ""  # Display name for the strategy


@dataclass
class RLModelConfig:
    """Configuration for an RL model."""

    label: str
    path: str


class StrategyConfigManager:
    """Manages LLM and RL strategy configurations in memory."""

    def __init__(self):
        self.llm_configs: Dict[str, LLMProviderConfig] = {}
        self.rl_configs: Dict[str, RLModelConfig] = {}

    def __getitem__(self, key: str):
        if key.startswith("llm:"):
            label = key[len("llm:") :]
            return self.llm_configs[label]
        elif key.startswith("rl:"):
            label = key[len("rl:") :]
            return self.rl_configs[label]
        return None

    def add_llm_config(
        self, provider: str, model_name: str, label: str, api_key: Optional[str] = None
    ):
        """Add or update an LLM configuration."""
        if not label:
            label = f"{provider}_{model_name}"

        config = LLMProviderConfig(
            provider=provider,
            model_name=model_name,
            api_key=api_key if api_key else None,
            label=label,
        )
        self.llm_configs[label] = config
        return f"✅ Added LLM strategy: {label}"

    def remove_llm_config(self, label: str):
        """Remove an LLM configuration."""
        if label in self.llm_configs:
            del self.llm_configs[label]
            return f"✅ Removed LLM strategy: {label}"
        return f"❌ Strategy not found: {label}"

    def add_rl_config(self, label: str, path: str):
        """Add or update an RL configuration."""
        if not Path(path).exists():
            return f"❌ Model file not found: {path}"

        config = RLModelConfig(label=label, path=path)
        self.rl_configs[label] = config
        return f"✅ Added RL strategy: {label}"

    def remove_rl_config(self, label: str):
        """Remove an RL configuration."""
        if label in self.rl_configs:
            del self.rl_configs[label]
            return f"✅ Removed RL strategy: {label}"
        return f"❌ Strategy not found: {label}"

    def get_all_strategy_names(self, base_strategies: List[str]) -> List[str]:
        """Get list of all strategies including configured ones."""
        strategies = base_strategies.copy()
        strategies.extend([f"llm:{label}" for label in self.llm_configs.keys()])
        strategies.extend([f"rl:{label}" for label in self.rl_configs.keys()])
        return strategies

    def get_llm_list(self) -> List[str]:
        """Get list of configured LLM strategies."""
        return list(self.llm_configs.keys())

    def get_rl_list(self) -> List[str]:
        """Get list of configured RL strategies."""
        return list(self.rl_configs.keys())


def create_llm_config_ui(
    config_manager: StrategyConfigManager,
    base_strategies: list[str],
    strategy_dropdowns: List,
):
    """Create the LLM configuration UI."""

    with gr.Accordion("⚙️ Configure LLM Strategies", open=False):
        gr.Markdown("### Add LLM Strategy")

        with gr.Row():
            provider_dropdown = gr.Dropdown(
                choices=[
                    "ollama",
                    "openai",
                    "gemini",
                    "grok",
                    "claude",
                    "anthropic",
                    "custom",
                ],
                label="Provider",
                value="ollama",
            )
            model_name_input = gr.Textbox(
                label="Model Name",
                placeholder="e.g., qwen3, gpt-4, gemini-pro",
            )

        with gr.Row():
            label_input = gr.Textbox(
                label="Strategy Label",
                placeholder="e.g., My_GPT4_Agent (leave empty to auto-generate)",
            )
            api_key_input = gr.Textbox(
                label="API Key (optional)",
                placeholder="Enter API key if required",
                type="password",
            )

        add_llm_btn = gr.Button("➕ Add LLM Strategy", variant="primary")
        llm_status = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("### Configured LLM Strategies")
        llm_list = gr.Dropdown(
            choices=[],
            label="Select to Remove",
        )
        remove_llm_btn = gr.Button("🗑️ Remove Selected LLM", variant="stop")

        # Event handlers
        def add_llm(provider, model, label, api_key):
            result = config_manager.add_llm_config(provider, model, label, api_key)

            # Update all strategy dropdowns
            all_strategies = config_manager.get_all_strategy_names(base_strategies)
            # Preserve the special 'empty' seat option in all dropdowns
            if "empty" not in all_strategies:
                all_strategies = all_strategies + ["empty"]
            dropdown_updates = [
                gr.update(choices=all_strategies) for _ in strategy_dropdowns
            ]

            return [
                result,
                gr.update(choices=config_manager.get_llm_list()),
            ] + dropdown_updates

        def remove_llm(label):
            from ludo_rl.strategy.registry import available as get_available_strategies

            if not label:
                return ["❌ No strategy selected", gr.update()] + [
                    gr.update() for _ in strategy_dropdowns
                ]
            result = config_manager.remove_llm_config(label)

            # Update all strategy dropdowns
            base_strategies = list(get_available_strategies(False).keys())
            all_strategies = config_manager.get_all_strategy_names(base_strategies)
            if "empty" not in all_strategies:
                all_strategies = all_strategies + ["empty"]
            dropdown_updates = [
                gr.update(choices=all_strategies) for _ in strategy_dropdowns
            ]

            return [
                result,
                gr.update(choices=config_manager.get_llm_list()),
            ] + dropdown_updates

        add_llm_btn.click(
            fn=add_llm,
            inputs=[provider_dropdown, model_name_input, label_input, api_key_input],
            outputs=[llm_status, llm_list] + strategy_dropdowns,
        )

        remove_llm_btn.click(
            fn=remove_llm,
            inputs=[llm_list],
            outputs=[llm_status, llm_list] + strategy_dropdowns,
        )

    return llm_status, llm_list


def create_rl_config_ui(
    config_manager: StrategyConfigManager,
    base_strategies: list[str],
    strategy_dropdowns: List,
):
    """Create the RL model configuration UI."""

    with gr.Accordion("🤖 Configure RL Strategies", open=False):
        gr.Markdown("### Add RL Model Strategy")

        with gr.Row():
            rl_label_input = gr.Textbox(
                label="Strategy Label",
                placeholder="e.g., PPO_1M_Steps",
            )
            rl_path_input = gr.Textbox(
                label="Model Path",
                placeholder="e.g., training/save_strategies/ludo_model.zip",
            )

        add_rl_btn = gr.Button("➕ Add RL Strategy", variant="primary")
        rl_status = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("### Configured RL Strategies")
        rl_list = gr.Dropdown(
            choices=[],
            label="Select to Remove",
        )
        remove_rl_btn = gr.Button("🗑️ Remove Selected RL", variant="stop")

        # Event handlers
        def add_rl(label, path):
            if not label or not path:
                return ["❌ Label and path are required", gr.update()] + [
                    gr.update() for _ in strategy_dropdowns
                ]
            result = config_manager.add_rl_config(label, path)

            # Update all strategy dropdowns
            all_strategies = config_manager.get_all_strategy_names(base_strategies)
            if "empty" not in all_strategies:
                all_strategies = all_strategies + ["empty"]
            dropdown_updates = [
                gr.update(choices=all_strategies) for _ in strategy_dropdowns
            ]

            return [
                result,
                gr.update(choices=config_manager.get_rl_list()),
            ] + dropdown_updates

        def remove_rl(label):
            from ludo_rl.strategy.registry import available as get_available_strategies

            if not label:
                return ["❌ No strategy selected", gr.update()] + [
                    gr.update() for _ in strategy_dropdowns
                ]
            result = config_manager.remove_rl_config(label)

            # Update all strategy dropdowns
            base_strategies = list(get_available_strategies(False).keys())
            all_strategies = config_manager.get_all_strategy_names(base_strategies)
            if "empty" not in all_strategies:
                all_strategies = all_strategies + ["empty"]
            dropdown_updates = [
                gr.update(choices=all_strategies) for _ in strategy_dropdowns
            ]

            return [
                result,
                gr.update(choices=config_manager.get_rl_list()),
            ] + dropdown_updates

        add_rl_btn.click(
            fn=add_rl,
            inputs=[rl_label_input, rl_path_input],
            outputs=[rl_status, rl_list] + strategy_dropdowns,
        )

        remove_rl_btn.click(
            fn=remove_rl,
            inputs=[rl_list],
            outputs=[rl_status, rl_list] + strategy_dropdowns,
        )

    return rl_status, rl_list
