import json
import random
import time
from dataclasses import asdict
from typing import List

import gradio as gr

from ludo_rl.ludo.game import LudoGame
from .models import PlayerColor
from ludo_interface.board_viz import draw_board

from .game_manager import GameManager
from .utils import Utils


class EventHandler:
    """Handles UI event callbacks and interactions."""

    def __init__(
        self,
        game_manager: GameManager,
        utils: Utils,
        ai_strategies: List[str],
        default_players: List[PlayerColor],
        show_token_ids: bool,
    ):
        self.game_manager = game_manager
        self.utils = utils
        self.ai_strategies = ai_strategies
        self.default_players = default_players
        self.show_token_ids = show_token_ids

    def _ui_init(self, *strats):
        game = self.game_manager.init_game(list(strats))
        pil_img = draw_board(
            self.game_manager.game_state_tokens(game), show_ids=self.show_token_ids
        )
        html = self.utils.img_to_data_uri(pil_img)

        current_player = game.get_current_player()
        player_html = f"<h3 style='color: {current_player.color.value};'>🎯 Current Player: {current_player.color.value.title()}</h3>"

        has_human = any(s == "human" for s in strats)
        controls_visible = has_human and self.game_manager.is_human_turn(game)

        return (
            game,
            html,
            "🎮 Game initialized! Roll the dice to start.",
            [],
            {"games": 0, "wins": {c.value: 0 for c in self.default_players}},
            player_html,
            gr.update(visible=controls_visible),
            "",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            [],
            None,  # reset pending_dice
            None,  # reset selected_token_id
            0,  # auto_steps_remaining
            0.5,  # auto_delay_state default
        )

    def _ui_random_strategies(self):
        strategies = [s for s in self.ai_strategies if s != "human"]
        return [random.choice(strategies) for _ in range(len(self.default_players))]

    def _ui_steps(
        self, game, history: List[str], show, pending_dice, human_choice=None
    ):
        if game is None:
            return (
                None,
                None,
                "No game initialized",
                history,
                False,
                "",
                "",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                [],
                None,  # pending_dice
                None,  # selected_token_id
                0,  # auto_steps_remaining
                None,  # auto_delay_state
            )

        game, desc, tokens, move_opts, waiting = self.game_manager.play_step(
            game, human_choice, pending_dice
        )
        history.append(desc)
        if len(history) > 50:
            history = history[-50:]

        pil_img = draw_board(tokens, show_ids=show)
        html = self.utils.img_to_data_uri(pil_img)

        if not game.game_over:
            current_player = game.get_current_player()
            player_html = f"<h3 style='color: {current_player.color.value};'>🎯 Current Player: {current_player.color.value.title()}</h3>"
        else:
            player_html = f"<h3>🏆 Winner: {game.winner.color.value.title()}!</h3>"

        if waiting and move_opts:
            moves_html = (
                "<h4>Choose your move:</h4><ul>"
                + "".join(
                    [
                        f"<li><strong>Token {opt['token_id']}</strong>: {opt['description']} ({opt['move_type']})</li>"
                        for opt in move_opts
                    ]
                )
                + "</ul>"
            )
            btn_updates = [
                gr.update(
                    visible=i < len(move_opts),
                    value=(
                        f"Move Token {move_opts[i]['token_id']}"
                        if i < len(move_opts)
                        else ""
                    ),
                )
                for i in range(4)
            ]
            # keep pending_dice if we're still waiting (it may be provided from auto-play)
            next_pending_dice = pending_dice
            return (
                game,
                html,
                desc,
                history,
                waiting,
                player_html,
                moves_html,
                gr.update(visible=True),
                *btn_updates,
                move_opts,
                next_pending_dice,  # pending_dice
                None,  # selected_token_id
                0,  # auto_steps_remaining (manual step path)
                None,  # auto_delay_state
            )
        else:
            # clear pending_dice and selected_token_id after the turn resolves
            return (
                game,
                html,
                desc,
                history,
                False,
                player_html,
                "",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                [],
                None,  # pending_dice
                None,  # selected_token_id
                0,  # auto_steps_remaining
                None,  # auto_delay_state
            )

    def _ui_run_auto(self, n, delay, game: LudoGame, history: List[str], show: bool):
        if game is None:
            yield (
                None,
                None,
                "No game",
                history,
                False,
                "",
                "",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                [],
                None,  # pending_dice
                None,  # selected_token_id
                0,  # auto_steps_remaining
                delay,  # auto_delay_state
            )
            return

        desc = ""
        remaining = int(n)
        for _ in range(int(n)):
            if self.game_manager.is_human_turn(game):
                current_player = game.get_current_player()
                dice = game.roll_dice()
                valid_moves = game.get_valid_moves(current_player, dice)

                if valid_moves:
                    pil_img = draw_board(
                        self.game_manager.game_state_tokens(game), show_ids=show
                    )
                    html = self.utils.img_to_data_uri(pil_img)
                    player_html = f"<h3 style='color: {current_player.color.value};'>🎯 Current Player: {current_player.color.value.title()}</h3>"
                    desc = f"Auto-play paused: {current_player.color.value} rolled {dice} - Choose your move:"
                    history.append(desc)
                    if len(history) > 50:
                        history = history[-50:]

                    move_options = self.game_manager.get_human_move_options(game, dice)
                    moves_html = (
                        "<h4>Choose your move:</h4><ul>"
                        + "".join(
                            [
                                f"<li><strong>Token {opt['token_id']}</strong>: {opt['description']} ({opt['move_type']})</li>"
                                for opt in move_options
                            ]
                        )
                        + "</ul>"
                    )
                    btn_updates = [
                        gr.update(
                            visible=i < len(move_options),
                            value=(
                                f"Move Token {move_options[i]['token_id']}"
                                if i < len(move_options)
                                else ""
                            ),
                        )
                        for i in range(4)
                    ]

                    # set pending_dice to the rolled value and pause for human
                    remaining_after_pause = max(remaining - 1, 0)
                    yield (
                        game,
                        html,
                        desc,
                        history,
                        True,
                        player_html,
                        moves_html,
                        gr.update(visible=True),
                        *btn_updates,
                        move_options,
                        dice,  # pending_dice
                        None,  # selected_token_id
                        remaining_after_pause,  # auto_steps_remaining
                        delay,  # auto_delay_state
                    )
                    return
                else:
                    extra_turn = dice == 6
                    if not extra_turn:
                        game.next_turn()
                    desc = f"{current_player.color.value} rolled {dice} - no moves{' (extra turn)' if extra_turn else ''}"
                    history.append(desc)
                    if len(history) > 50:
                        history = history[-50:]
                    remaining = max(remaining - 1, 0)
            else:
                game, step_desc, _, _, _ = self.game_manager.play_step(game)
                desc = step_desc
                history.append(step_desc)
                if len(history) > 50:
                    history = history[-50:]
                remaining = max(remaining - 1, 0)

            pil_img = draw_board(
                self.game_manager.game_state_tokens(game), show_ids=show
            )
            html = self.utils.img_to_data_uri(pil_img)

            if not game.game_over:
                current_player = game.get_current_player()
                player_html = f"<h3 style='color: {current_player.color.value};'>🎯 Current Player: {current_player.color.value.title()}</h3>"
            else:
                player_html = f"<h3>🏆 Winner: {game.winner.color.value.title()}!</h3>"

            waiting = self.game_manager.is_human_turn(game) and not game.game_over
            # clear pending_dice while continuing auto-play (no human pause)
            yield (
                game,
                html,
                desc,
                history,
                waiting,
                player_html,
                "",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                [],
                None,  # pending_dice
                None,  # selected_token_id
                remaining,  # auto_steps_remaining
                delay,  # auto_delay_state
            )

            if game.game_over:
                break
            if delay and delay > 0 and not waiting:
                time.sleep(float(delay))

    def _ui_make_human_move(
        self,
        token_id,
        game,
        history,
        show,
        move_opts,
        pending_dice,
        auto_steps_remaining,
        auto_delay_state,
    ):
        if not move_opts:
            return (
                game,
                None,
                "No moves available",
                history,
                False,
                "",
                "",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                [],
                pending_dice,  # pending_dice
                None,  # selected_token_id
                auto_steps_remaining,  # auto_steps_remaining
                auto_delay_state,  # auto_delay_state
            )
        # Use the pending_dice from auto-play to execute the human's chosen move
        out = list(self._ui_steps(game, history, show, pending_dice, token_id))
        # out shape: [..., move_options, pending_dice, selected_token_id, auto_steps_remaining, auto_delay_state]
        if len(out) >= 17:
            out[-2] = auto_steps_remaining
            out[-1] = auto_delay_state
        return tuple(out)

    def _ui_resume_auto(
        self, remaining, delay, game: LudoGame, history: List[str], show: bool
    ):
        try:
            rem = int(remaining) if remaining is not None else 0
        except Exception:
            rem = 0
        if rem <= 0 or game is None:
            # No resume needed; return a snapshot without changing states
            if game is not None:
                pil_img = draw_board(
                    self.game_manager.game_state_tokens(game), show_ids=show
                )
                html = self.utils.img_to_data_uri(pil_img)
                if not game.game_over:
                    current_player = game.get_current_player()
                    player_html = f"<h3 style='color: {current_player.color.value};'>🎯 Current Player: {current_player.color.value.title()}</h3>"
                else:
                    player_html = (
                        f"<h3>🏆 Winner: {game.winner.color.value.title()}!</h3>"
                    )
            else:
                html = None
                player_html = ""
            return (
                game,
                html,
                "",
                history,
                bool(
                    game
                    and self.game_manager.is_human_turn(game)
                    and not game.game_over
                ),
                player_html,
                "",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                [],
                None,  # pending_dice
                None,  # selected_token_id
                0,  # auto_steps_remaining
                delay,  # auto_delay_state
            )
        # Resume by delegating to _ui_run_auto
        for out in self._ui_run_auto(rem, delay, game, history, show):
            yield out

    def _ui_export(self, game: LudoGame):
        if not game:
            return "No game"
        state_dict = {
            "current_turn": game.current_player_index,
            "tokens": {
                k: [asdict(v.to_dict()) for v in vs]
                for k, vs in self.game_manager.game_state_tokens(game).items()
            },
            "game_over": game.game_over,
            "winner": game.winner.color.value if game.winner else None,
        }
        return json.dumps(state_dict, indent=2)

    def _ui_run_bulk(self, n_games, *strats):
        ai_strats = [s if s != "human" else "random" for s in strats]
        win_counts = {c.value: 0 for c in self.default_players}

        # Run the simulation
        total_games = int(n_games)
        for _ in range(total_games):
            g = self.game_manager.init_game(list(ai_strats))
            turns_taken = 0
            while not g.game_over and turns_taken < 1000:  # Safety limit
                g, _, _, _, _ = self.game_manager.play_step(g)
                turns_taken += 1
            if g.winner:
                win_counts[g.winner.color.value] += 1

        # Calculate statistics
        total = sum(win_counts.values()) or 1

        # Summary JSON for the main results
        summary = {
            "tournament_results": {
                "total_games": total_games,
                "completed_games": total,
                "strategies": {strats[i]: ai_strats[i] for i in range(len(strats))},
            },
            "win_statistics": {
                k: {
                    "wins": v,
                    "win_rate": round(v / total, 3),
                    "win_percentage": f"{round(v / total * 100, 1)}%",
                }
                for k, v in win_counts.items()
            },
        }

        # Detailed text results
        detailed_text = f"""🏆 TOURNAMENT RESULTS
        
📊 SIMULATION SUMMARY
├─ Total Games: {total_games:,}
├─ Completed: {total:,}
├─ Success Rate: {round(total / total_games * 100, 1)}%
└─ Strategies: {len(ai_strats)} players

🎯 PLAYER PERFORMANCE
"""

        # Sort by win count for better display
        sorted_results = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)

        for i, (player, wins) in enumerate(sorted_results):
            rank_emoji = ["🥇", "🥈", "🥉", "🏅"][min(i, 3)]
            win_rate = wins / total if total > 0 else 0
            strategy = ai_strats[list(win_counts.keys()).index(player)]

            detailed_text += f"""
{rank_emoji} {player.upper()} ({strategy})
   ├─ Wins: {wins:,} / {total:,}
   ├─ Win Rate: {win_rate:.3f} ({win_rate * 100:.1f}%)
   └─ Strategy: {strategy}
"""

        # Performance analysis
        if total > 100:
            best_player = max(win_counts.items(), key=lambda x: x[1])
            worst_player = min(win_counts.items(), key=lambda x: x[1])
            best_rate = best_player[1] / total
            worst_rate = worst_player[1] / total

            detailed_text += f"""
📈 ANALYSIS
├─ Best Performer: {best_player[0]} ({best_rate * 100:.1f}%)
├─ Worst Performer: {worst_player[0]} ({worst_rate * 100:.1f}%)
├─ Performance Gap: {(best_rate - worst_rate) * 100:.1f}%
└─ Statistical Confidence: {"High" if total >= 1000 else "Medium" if total >= 500 else "Low"}
"""

        # Status message
        status_html = f"""
        <div style='text-align:center;padding:20px;'>
            <h3 style='color:#28a745;margin:0;'>✅ Tournament Complete!</h3>
            <p style='margin:5px 0;color:#666;'>
                Simulated {total_games:,} games • Best: <strong>{max(win_counts.items(), key=lambda x: x[1])[0].title()}</strong> 
                ({max(win_counts.values()) / total * 100:.1f}% win rate)
            </p>
        </div>
        """

        # Simple chart visualization (HTML/CSS bar chart)
        chart_html = self._create_win_rate_chart(win_counts, total, ai_strats)

        return summary, detailed_text, status_html, chart_html

    def _create_win_rate_chart(self, win_counts: dict, total, strategies):
        """Create a simple HTML/CSS bar chart for win rates."""
        if total == 0:
            return "<div style='text-align:center;padding:40px;color:#666;'>No data to display</div>"

        # Color scheme for players
        colors = {
            PlayerColor.RED: "#dc3545",
            PlayerColor.GREEN: "#28a745",
            PlayerColor.YELLOW: "#ffc107",
            PlayerColor.BLUE: "#007bff",
        }

        chart_html = """
        <div style='padding:20px;'>
            <h4 style='text-align:center;margin-bottom:20px;color:#333;'>🏆 Win Rate Comparison</h4>
            <div style='max-width:600px;margin:0 auto;'>
        """

        max_wins = max(win_counts.values()) if win_counts.values() else 1

        for player, wins in win_counts.items():
            win_rate = wins / total if total > 0 else 0
            bar_width = (wins / max_wins * 100) if max_wins > 0 else 0
            color = colors.get(player, "#6c757d")
            strategy = strategies[list(win_counts.keys()).index(player)]

            chart_html += f"""
            <div style='margin:15px 0;'>
                <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;'>
                    <span style='font-weight:bold;color:{color};'>{player.title()}</span>
                    <span style='font-size:0.9em;color:#666;'>{strategy}</span>
                    <span style='font-weight:bold;'>{win_rate * 100:.1f}%</span>
                </div>
                <div style='background:#f8f9fa;border-radius:10px;height:25px;position:relative;overflow:hidden;'>
                    <div style='background:{color};height:100%;width:{bar_width:.1f}%;border-radius:10px;transition:width 0.5s ease;display:flex;align-items:center;justify-content:center;'>
                        <span style='color:white;font-size:0.8em;font-weight:bold;'>{wins}</span>
                    </div>
                </div>
            </div>
            """

        chart_html += (
            """
            </div>
            <div style='text-align:center;margin-top:20px;font-size:0.9em;color:#666;'>
                📊 Based on """
            + f"{total:,} completed games"
            + """
            </div>
        </div>
        """
        )

        return chart_html

    def _ui_update_stats(self, stats, game: LudoGame):
        if game and game.game_over and game.winner:
            stats = dict(stats)
            stats["games"] += 1
            stats["wins"][game.winner.color.value] += 1
        return stats