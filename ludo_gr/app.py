import os
from typing import Dict, List

_GRADIO_BASE = os.path.join(os.getcwd(), "gradio_runtime")
os.environ.setdefault("GRADIO_TEMP_DIR", _GRADIO_BASE)
os.environ.setdefault(
    "GRADIO_CACHE_DIR",
    os.path.join(_GRADIO_BASE, "cache"),
)
for _p in (
    os.environ["GRADIO_TEMP_DIR"],
    os.environ["GRADIO_CACHE_DIR"],
    os.path.join(_GRADIO_BASE, "vibe_edit_history"),
):
    try:
        os.makedirs(_p, exist_ok=True)
    except Exception:
        pass

import json
from copy import deepcopy
import io, base64

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
    return f"<img src='data:image/png;base64,{b64}' style='image-rendering:pixelated;width:100%;max-width:640px;' />"


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


def _play_step(game):
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
    if current_player.strategy:
        try:
            ctx = game.get_game_state_for_ai()
            ctx["game_info"]["dice_value"] = dice
            token_choice = current_player.strategy.decide(ctx)
            # find move with that token_id
            for mv in valid:
                if mv["token_id"] == token_choice:
                    chosen = mv
                    break
            if chosen is None:
                chosen = valid[0]
        except Exception:
            chosen = valid[0]
    else:
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
        gr.Markdown("# Ludo AI Visualizer\nAI vs AI token movement visualization")
        with gr.Row():
            strategy_inputs = []
            for color in DEFAULT_PLAYERS:
                dd = gr.Dropdown(
                    choices=AI_STRATEGIES,
                    value=AI_STRATEGIES[0],
                    label=f"{color.value} strategy",
                )
                strategy_inputs.append(dd)

        with gr.Row():
            init_btn = gr.Button("Start New Game")
            step_btn = gr.Button("Play Step")
            auto_steps = gr.Slider(1, 200, value=1, step=1, label="Auto Steps")
            bulk_games = gr.Slider(10, 1000, value=100, step=10, label="Bulk Games")
            bulk_run_btn = gr.Button("Run Bulk Games")

        with gr.Row():
            show_ids = gr.Checkbox(label="Show Token IDs", value=True)
            export_btn = gr.Button("Export Game State")
            move_history_btn = gr.Button("Show Move History (last 50)")

        # Replace Image with HTML to avoid filesystem writes per render
        board_plot = gr.HTML(label="Board")
        log = gr.Textbox(label="Last Action", interactive=False)
        history_box = gr.Textbox(label="Move History", lines=10)
        bulk_results = gr.Textbox(label="Bulk Results")
        export_box = gr.Textbox(label="Game State JSON", lines=6)
        stats_display = gr.JSON(label="Strategy Performance")

        # States
        game_state = gr.State()
        move_history = gr.State([])
        stats_state = gr.State(
            {"games": 0, "wins": {c.value: 0 for c in DEFAULT_PLAYERS}}
        )

        def _init(*strats):
            game = _init_game(list(strats))
            pil_img = draw_board(_game_state_tokens(game), show_ids=True)
            html = _img_to_data_uri(pil_img)
            return game, html, "Game initialized", []

        def _steps(n, game, history, show):
            if game is None:
                return None, None, "No game", history
            desc = ""
            tokens = _game_state_tokens(game)
            for _ in range(n):
                game, step_desc, tokens = _play_step(game)
                desc = step_desc
                history.append(step_desc)
                if len(history) > 50:
                    history = history[-50:]
                if game.game_over:
                    break
            pil_img = draw_board(tokens, show_ids=show)
            html = _img_to_data_uri(pil_img)
            return game, html, desc, history

        def _export(game):
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

        def _update_stats(stats, game):
            if game and game.game_over and game.winner:
                stats = deepcopy(stats)
                stats["games"] += 1
                stats["wins"][game.winner.color.value] += 1
            return stats

        # Wiring
        init_btn.click(
            _init,
            strategy_inputs,
            [game_state, board_plot, log, move_history],
        )
        step_btn.click(
            lambda g, h, s: _steps(1, g, h, s),
            [game_state, move_history, show_ids],
            [game_state, board_plot, log, move_history],
        ).then(
            _update_stats,
            [stats_state, game_state],
            [stats_state],
        ).then(lambda st: st, [stats_state], [stats_display])

        auto_steps.release(
            lambda n, g, h, s: _steps(n, g, h, s),
            [auto_steps, game_state, move_history, show_ids],
            [game_state, board_plot, log, move_history],
        ).then(
            _update_stats,
            [stats_state, game_state],
            [stats_state],
        ).then(lambda st: st, [stats_state], [stats_display])

        bulk_run_btn.click(
            _run_bulk,
            [bulk_games] + strategy_inputs,
            [bulk_results],
        )

        export_btn.click(_export, [game_state], [export_box])
        move_history_btn.click(
            lambda h: "\n".join(h[-50:]), [move_history], [history_box]
        )

    return demo


if __name__ == "__main__":
    launch_app().launch()
