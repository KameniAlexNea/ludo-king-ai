import os
from typing import Dict, List

os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(os.getcwd(), "gradio_runtime"))
os.environ.setdefault(
    "GRADIO_CACHE_DIR",
    os.path.join(os.getcwd(), "gradio_runtime", "cache"),
)

import base64
import io
import json

import gradio as gr

from ludo.game import LudoGame
from ludo.player import PlayerColor
from ludo.strategy import StrategyFactory
from ludo_gr.board_viz import draw_board

AI_STRATEGIES = StrategyFactory.get_available_strategies()
DEFAULT_PLAYERS = [
    PlayerColor.RED,
    PlayerColor.GREEN,
    PlayerColor.YELLOW,
    PlayerColor.BLUE,
]


def _img_to_data_uri(pil_img):
    """Return an inline data URI for the PIL image to avoid Gradio temp file folders."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return (
        "<div style='display:flex;justify-content:center;'>"
        f"<img src='data:image/png;base64,{b64}' "
        "style='image-rendering:pixelated;max-width:640px;width:100%;height:auto;' />"
        "</div>"
    )


def _init_game(strategies: List[str]):
    # Instantiate strategies via factory
    strategy_objs = []
    for color, strat_name in zip(DEFAULT_PLAYERS, strategies):
        strategy_objs.append(StrategyFactory.create_strategy(strat_name))
    # Build game with chosen strategies
    game = LudoGame([c for c in DEFAULT_PLAYERS])
    # Attach strategies
    for player, strat in zip(game.players, strategy_objs):
        player.strategy = strat
    return game


def _game_state_tokens(game: LudoGame) -> Dict[str, List[Dict]]:
    token_map: Dict[str, List[Dict]] = {c.value: [] for c in PlayerColor}
    for p in game.players:
        for t in p.tokens:
            token_map[p.color.value].append(t.to_dict())
    return token_map


def _serialize_move(move_result: Dict) -> str:
    if not move_result or not move_result.get("success"):
        return "No move"
    parts = [
        f"{move_result['player_color']} token {move_result['token_id']} -> {move_result['new_position']}"
    ]
    if move_result.get("captured_tokens"):
        cap = move_result["captured_tokens"]
        parts.append(f"captured {len(cap)}")
    if move_result.get("token_finished"):
        parts.append("finished")
    if move_result.get("extra_turn"):
        parts.append("extra turn")
    return ", ".join(parts)


def _play_step(game: LudoGame):
    if game.game_over:
        return game, "Game over", _game_state_tokens(game)
    current_player = game.get_current_player()
    dice = game.roll_dice()
    valid = game.get_valid_moves(current_player, dice)
    if not valid:
        game.next_turn()
        return (
            game,
            f"{current_player.color.value} rolled {dice} - no moves",
            _game_state_tokens(game),
        )
    # If player has strategy use it; else pick first
    chosen = None
    ctx = game.get_game_state_for_ai()
    ctx["game_info"]["dice_value"] = dice
    token_choice = current_player.make_strategic_decision(ctx)
    # find move with that token_id
    for mv in valid:
        if mv["token_id"] == token_choice:
            chosen = mv
            break
    if chosen is None:
        chosen = valid[0]
    move_res = game.execute_move(current_player, chosen["token_id"], dice)
    desc = f"{current_player.color.value} rolled {dice}: {_serialize_move(move_res)}"
    if move_res.get("extra_turn") and not game.game_over:
        # do not advance turn
        pass
    else:
        if not game.game_over:
            game.next_turn()
    if game.game_over:
        desc += f" | WINNER: {game.winner.color.value}"
    return game, desc, _game_state_tokens(game)


def launch_app():
    with gr.Blocks(title="Ludo AI Visualizer") as demo:
        gr.Markdown("# Ludo AI Visualizer")
        with gr.Tabs():
            with gr.TabItem("Play Game"):
                with gr.Row():
                    strategy_inputs = []
                    for color in DEFAULT_PLAYERS:
                        strategy_inputs.append(
                            gr.Dropdown(
                                choices=AI_STRATEGIES,
                                value=AI_STRATEGIES[0],
                                label=f"{color.value} strategy",
                            )
                        )
                with gr.Row():
                    init_btn = gr.Button("Start New Game")
                    step_btn = gr.Button("Play Step")
                    auto_steps_n = gr.Number(value=1, label="Steps")
                    auto_delay = gr.Number(value=0.2, label="Delay (s)")
                    run_auto_btn = gr.Button("Run Auto Steps")
                with gr.Row():
                    show_ids = gr.Checkbox(label="Show Token IDs", value=True)
                    export_btn = gr.Button("Export Game State")
                    move_history_btn = gr.Button("Show Move History (last 50)")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        board_plot = gr.HTML(label="Board")
                    with gr.Column(scale=1):
                        log = gr.Textbox(label="Last Action", interactive=False)
                        history_box = gr.Textbox(label="Move History", lines=10)
                stats_display = gr.JSON(
                    label="Performance",
                    value={"games": 0, "wins": {c.value: 0 for c in DEFAULT_PLAYERS}},
                )
            with gr.TabItem("Simulate Multiple Games"):
                sim_strat_inputs = []
                for color in DEFAULT_PLAYERS:
                    sim_strat_inputs.append(
                        gr.Dropdown(
                            choices=AI_STRATEGIES,
                            value=AI_STRATEGIES[0],
                            label=f"{color.value} strategy",
                        )
                    )
                with gr.Row():
                    bulk_games = gr.Slider(
                        10, 2000, value=200, step=10, label="Number of Games"
                    )
                    bulk_run_btn = gr.Button("Run Simulation")
                bulk_results = gr.Textbox(label="Simulation Results")
        export_box = gr.Textbox(label="Game State JSON", lines=6, visible=False)
        game_state = gr.State()
        move_history = gr.State([])
        stats_state = gr.State(
            {"games": 0, "wins": {c.value: 0 for c in DEFAULT_PLAYERS}}
        )

        def _init(*strats):
            game = _init_game(list(strats))
            pil_img = draw_board(_game_state_tokens(game), show_ids=True)
            html = _img_to_data_uri(pil_img)
            return (
                game,
                html,
                "Game initialized",
                [],
                {"games": 0, "wins": {c.value: 0 for c in DEFAULT_PLAYERS}},
            )

        def _steps(game, history: list[str], show):
            game, desc, tokens = _play_step(game)
            history.append(desc)
            if len(history) > 50:
                history = history[-50:]
            pil_img = draw_board(tokens, show_ids=show)
            html = _img_to_data_uri(pil_img)
            return game, html, desc, history

        import time

        def _run_auto(n, delay, game: LudoGame, history: list[str], show: bool):
            if game is None:
                return None, None, "No game", history
            tokens = _game_state_tokens(game)
            desc = ""
            for _ in range(int(n)):
                game, step_desc, tokens = _play_step(game)
                desc = step_desc
                history.append(step_desc)
                if len(history) > 50:
                    history = history[-50:]
                pil_img = draw_board(tokens, show_ids=show)
                html = _img_to_data_uri(pil_img)
                yield game, html, desc, history
                if game.game_over:
                    break
                if delay and delay > 0:
                    time.sleep(float(delay))

        def _export(game: LudoGame):
            if not game:
                return "No game"
            state_dict = {
                "current_turn": game.current_player_index,
                "tokens": _game_state_tokens(game),
                "game_over": game.game_over,
                "winner": game.winner.color.value if game.winner else None,
            }
            return json.dumps(state_dict, indent=2)

        def _run_bulk(n_games, *strats):
            win_counts = {c.value: 0 for c in DEFAULT_PLAYERS}
            for _ in range(int(n_games)):
                g = _init_game(list(strats))
                while not g.game_over:
                    g, _, _ = _play_step(g)
                win_counts[g.winner.color.value] += 1
            total = sum(win_counts.values()) or 1
            summary = {
                k: {"wins": v, "win_rate": round(v / total, 3)}
                for k, v in win_counts.items()
            }
            return json.dumps(summary, indent=2)

        def _update_stats(stats, game: LudoGame):
            if game and game.game_over and game.winner:
                stats = dict(stats)
                stats["games"] += 1
                stats["wins"][game.winner.color.value] += 1
            return stats

        init_btn.click(
            _init,
            strategy_inputs,
            [game_state, board_plot, log, move_history, stats_state],
        )
        step_btn.click(
            _steps,
            [game_state, move_history, show_ids],
            [game_state, board_plot, log, move_history],
        ).then(_update_stats, [stats_state, game_state], [stats_state]).then(
            lambda s: s, [stats_state], [stats_display]
        )
        run_auto_btn.click(
            _run_auto,
            [auto_steps_n, auto_delay, game_state, move_history, show_ids],
            [game_state, board_plot, log, move_history],
        ).then(_update_stats, [stats_state, game_state], [stats_state]).then(
            lambda s: s, [stats_state], [stats_display]
        )
        move_history_btn.click(
            lambda h: "\n".join(h[-50:]), [move_history], [history_box]
        )
        export_btn.click(_export, [game_state], [export_box])
        bulk_run_btn.click(_run_bulk, [bulk_games] + sim_strat_inputs, [bulk_results])

    return demo


if __name__ == "__main__":
    launch_app().launch()
