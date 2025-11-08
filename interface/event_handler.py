from __future__ import annotations

import json
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import gradio as gr

from ludo_rl.ludo.config import config

from .board_viz import draw_board
from .game_manager import GameManager
from .types import (
    INDEX_TO_COLOR,
    PlayerColor,
    SessionState,
    TokenView,
)
from .utils import Utils


class EventHandler:
    """Handles UI event callbacks and interactions for the refactored engine."""

    MAX_HISTORY = 50
    BUTTON_COUNT = 4

    def __init__(
        self,
        game_manager: GameManager,
        utils: Utils,
        ai_strategies: List[str],
        default_players: List[PlayerColor],
        show_token_ids: bool,
    ) -> None:
        self.game_manager = game_manager
        self.utils = utils
        self.ai_strategies = list(dict.fromkeys(ai_strategies))
        self.default_players = default_players
        self.show_token_ids = show_token_ids
        self.non_human_strategies = [
            name for name in self.ai_strategies if name.lower() != "human"
        ]
        if not self.non_human_strategies:
            self.non_human_strategies = ["random"]

    # ------------------------------------------------------------------
    # Public UI callbacks
    # ------------------------------------------------------------------
    def _ui_init(self, *strategies: str):
        session = self.game_manager.init_game(list(strategies))
        tokens = self.game_manager.game_state_tokens(session)
        board_html = self._render_board(tokens, self.show_token_ids)
        current_html = self._current_player_html(session)

        stats_template = self._initial_stats()
        controls_visible = self.game_manager.is_human_turn(session)
        button_updates = self._build_move_buttons([])

        return (
            session,
            board_html,
            "🎮 Game initialized! Roll the dice to start.",
            [],
            stats_template,
            current_html,
            gr.update(visible=controls_visible),
            "",
            button_updates[0],
            button_updates[1],
            button_updates[2],
            button_updates[3],
            [],
            None,
            None,
            0,
            0.5,
        )

    def _ui_random_strategies(self) -> List[str]:
        choices = [
            random.choice(self.non_human_strategies)
            for _ in range(len(self.default_players))
        ]
        return choices

    def _ui_steps(
        self,
        session: Optional[SessionState],
        history: Optional[List[str]],
        show_ids: Optional[bool],
        pending_dice: Optional[int],
        human_choice: Optional[int] = None,
    ):
        if session is None:
            return self._no_game_response(history, show_ids, None)

        result = self.game_manager.play_step(session, human_choice, pending_dice)
        updated_history = self._append_history(history, result.description)
        pending_value = result.dice_roll if result.waiting_for_human else None

        return self._step_response(
            result.session,
            tokens=result.tokens,
            description=result.description,
            history=updated_history,
            waiting=result.waiting_for_human,
            move_options=result.move_options,
            show_ids=show_ids,
            pending_dice=pending_value,
            selected_token_id=None,
            auto_steps_remaining=0,
            auto_delay_state=None,
        )

    def _ui_run_auto(
        self,
        n_steps,
        delay,
        session: Optional[SessionState],
        history: Optional[List[str]],
        show_ids: Optional[bool],
    ):
        delay_value = self._coerce_delay(delay)

        if session is None:
            yield self._no_game_response(history, show_ids, delay_value)
            return

        try:
            total_steps = max(int(n_steps), 0)
        except (TypeError, ValueError):
            total_steps = 0

        if session.game_over:
            tokens = self.game_manager.game_state_tokens(session)
            history_list = list(history or [])
            yield self._step_response(
                session,
                tokens=tokens,
                description="Game over",
                history=history_list,
                waiting=False,
                move_options=[],
                show_ids=show_ids,
                pending_dice=None,
                selected_token_id=None,
                auto_steps_remaining=0,
                auto_delay_state=delay_value,
                controls_visible=False,
            )
            return

        if total_steps <= 0:
            tokens = self.game_manager.game_state_tokens(session)
            history_list = list(history or [])
            waiting = bool(
                self.game_manager.is_human_turn(session) and not session.game_over
            )
            yield self._step_response(
                session,
                tokens=tokens,
                description="",
                history=history_list,
                waiting=waiting,
                move_options=[],
                show_ids=show_ids,
                pending_dice=None,
                selected_token_id=None,
                auto_steps_remaining=0,
                auto_delay_state=delay_value,
                controls_visible=waiting,
            )
            return

        remaining = total_steps
        history_list = list(history or [])

        while remaining > 0:
            result = self.game_manager.play_step(session)
            history_list = self._append_history(history_list, result.description)
            remaining -= 1
            pending_value = result.dice_roll if result.waiting_for_human else None

            yield self._step_response(
                result.session,
                tokens=result.tokens,
                description=result.description,
                history=history_list,
                waiting=result.waiting_for_human,
                move_options=result.move_options,
                show_ids=show_ids,
                pending_dice=pending_value,
                selected_token_id=None,
                auto_steps_remaining=remaining,
                auto_delay_state=delay_value,
            )

            if result.waiting_for_human:
                return
            if result.session.game_over:
                break
            if delay_value and delay_value > 0 and remaining > 0:
                time.sleep(delay_value)

    def _ui_make_human_move(
        self,
        token_id: Optional[int],
        session: Optional[SessionState],
        history: Optional[List[str]],
        show_ids: Optional[bool],
        move_opts: Optional[List[dict]],
        pending_dice: Optional[int],
        auto_steps_remaining: int,
        auto_delay_state,
    ):
        if session is None:
            return self._no_game_response(history, show_ids, auto_delay_state)

        if not move_opts:
            return self._no_moves_response(
                session,
                history,
                show_ids,
                move_opts or [],
                pending_dice,
                auto_steps_remaining,
                auto_delay_state,
            )

        outputs = list(
            self._ui_steps(session, history, show_ids, pending_dice, token_id)
        )
        if len(outputs) >= 2:
            outputs[-2] = auto_steps_remaining
            outputs[-1] = auto_delay_state
        return tuple(outputs)

    def _ui_resume_auto(
        self,
        remaining,
        delay,
        session: Optional[SessionState],
        history: Optional[List[str]],
        show_ids: Optional[bool],
    ):
        delay_value = self._coerce_delay(delay)

        try:
            rem = int(remaining) if remaining is not None else 0
        except (TypeError, ValueError):
            rem = 0

        if session is None:
            return self._no_game_response(history, show_ids, delay_value)

        if rem <= 0:
            tokens = self.game_manager.game_state_tokens(session)
            history_list = list(history or [])
            waiting = bool(
                session
                and not session.game_over
                and self.game_manager.is_human_turn(session)
            )
            return self._step_response(
                session,
                tokens=tokens,
                description="",
                history=history_list,
                waiting=waiting,
                move_options=[],
                show_ids=show_ids,
                pending_dice=None,
                selected_token_id=None,
                auto_steps_remaining=0,
                auto_delay_state=delay_value,
                controls_visible=False,
            )

        for payload in self._ui_run_auto(rem, delay_value, session, history, show_ids):
            yield payload

    def _ui_export(self, session: Optional[SessionState]) -> str:
        if session is None:
            return "No game"

        tokens = self.game_manager.game_state_tokens(session)
        token_payload = {
            color.value: [
                {
                    "token_id": token.token_id,
                    "position": token.position,
                    "state": token.state.value,
                    "relative_position": token.relative_position,
                    "absolute_position": token.absolute_position,
                }
                for token in pieces
            ]
            for color, pieces in tokens.items()
        }

        state_dict = {
            "current_turn": session.current_player_index,
            "turn_counter": session.turn_counter,
            "game_over": session.game_over,
            "winner": session.winner_color.value if session.winner_color else None,
            "tokens": token_payload,
        }
        return json.dumps(state_dict, indent=2)

    def _ui_run_bulk(self, n_games, *strategies: str):
        try:
            total_games = max(int(n_games), 0)
        except (TypeError, ValueError):
            total_games = 0

        strategy_list = list(strategies)
        if not strategy_list:
            strategy_list = ["random"] * len(self.default_players)
        if len(strategy_list) < len(self.default_players):
            strategy_list.extend(
                ["random"] * (len(self.default_players) - len(strategy_list))
            )
        strategy_list = strategy_list[: len(self.default_players)]
        normalized = [s if s.lower() != "human" else "random" for s in strategy_list]

        win_counts: Dict[PlayerColor, int] = {
            color: 0 for color in self.default_players
        }

        for _ in range(total_games):
            session = self.game_manager.init_game(normalized)
            turns_taken = 0
            while not session.game_over and turns_taken < config.MAX_TURNS:
                _ = self.game_manager.play_step(session)
                turns_taken += 1
            if session.winner_color is not None:
                win_counts[session.winner_color] += 1

        completed_games = sum(win_counts.values())
        total_completed = completed_games or 1

        strategies_summary = {
            self.default_players[i].value: normalized[i]
            for i in range(len(self.default_players))
        }

        summary = {
            "tournament_results": {
                "total_games": total_games,
                "completed_games": completed_games,
                "strategies": strategies_summary,
            },
            "win_statistics": {
                color.value: {
                    "wins": wins,
                    "win_rate": round(wins / total_completed, 3),
                    "win_percentage": f"{round(wins / total_completed * 100, 1)}%",
                }
                for color, wins in win_counts.items()
            },
        }

        detailed_lines = [
            "🏆 TOURNAMENT RESULTS",
            "",
            "📊 SIMULATION SUMMARY",
            f"├─ Total Games: {total_games:,}",
            f"├─ Completed: {completed_games:,}",
            f"├─ Success Rate: {round(completed_games / total_games * 100, 1) if total_games else 0:.1f}%",
            f"└─ Strategies: {len(normalized)} players",
            "",
            "🎯 PLAYER PERFORMANCE",
        ]

        sorted_results = sorted(
            win_counts.items(), key=lambda item: item[1], reverse=True
        )

        rank_emojis = ["🥇", "🥈", "🥉", "🏅"]
        color_to_strategy = {
            self.default_players[i]: normalized[i]
            for i in range(len(self.default_players))
        }

        for idx, (color, wins) in enumerate(sorted_results):
            emoji = rank_emojis[min(idx, len(rank_emojis) - 1)]
            win_rate = wins / total_completed
            strategy = color_to_strategy.get(color, "random")
            detailed_lines.extend(
                [
                    "",
                    f"{emoji} {color.value.upper()} ({strategy})",
                    f"   ├─ Wins: {wins:,} / {total_completed:,}",
                    f"   ├─ Win Rate: {win_rate:.3f} ({win_rate * 100:.1f}%)",
                    f"   └─ Strategy: {strategy}",
                ]
            )

        if completed_games > 0:
            best_player = max(win_counts.items(), key=lambda item: item[1])
            worst_player = min(win_counts.items(), key=lambda item: item[1])
            best_rate = best_player[1] / total_completed
            worst_rate = worst_player[1] / total_completed
            confidence = (
                "High"
                if completed_games >= 1000
                else "Medium"
                if completed_games >= 500
                else "Low"
            )

            detailed_lines.extend(
                [
                    "",
                    "📈 ANALYSIS",
                    f"├─ Best Performer: {best_player[0].value} ({best_rate * 100:.1f}%)",
                    f"├─ Worst Performer: {worst_player[0].value} ({worst_rate * 100:.1f}%)",
                    f"├─ Performance Gap: {(best_rate - worst_rate) * 100:.1f}%",
                    f"└─ Statistical Confidence: {confidence}",
                ]
            )

        detailed_text = "\n".join(detailed_lines)

        if completed_games > 0:
            best_color = max(win_counts.items(), key=lambda item: item[1])[0]
            best_rate = win_counts[best_color] / total_completed * 100
            status_html = (
                "<div style='text-align:center;padding:20px;'>"
                "<h3 style='color:#28a745;margin:0;'>✅ Tournament Complete!</h3>"
                f"<p style='margin:5px 0;color:#666;'>Simulated {total_games:,} games • Best: "
                f"<strong>{best_color.display_name}</strong> ({best_rate:.1f}% win rate)</p>"
                "</div>"
            )
        else:
            status_html = (
                "<div style='text-align:center;padding:20px;'>"
                "<h3 style='color:#dc3545;margin:0;'>⚠️ No completed games</h3>"
                "<p style='margin:5px 0;color:#666;'>Consider increasing the number of simulations.</p>"
                "</div>"
            )

        chart_html = self._create_win_rate_chart(win_counts, completed_games, normalized)

        return summary, detailed_text, status_html, chart_html

    def _ui_update_stats(
        self, stats: Optional[Dict[str, object]], session: Optional[SessionState]
    ):
        if session and session.game_over and session.winner_color:
            updated = dict(stats or self._initial_stats())
            wins = dict(updated.get("wins", {}))
            updated["games"] = int(updated.get("games", 0)) + 1
            winner_key = session.winner_color.value
            wins[winner_key] = wins.get(winner_key, 0) + 1
            updated["wins"] = wins
            return updated
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _initial_stats(self) -> Dict[str, object]:
        return {"games": 0, "wins": {color.value: 0 for color in self.default_players}}

    def _append_history(
        self, history: Optional[List[str]], entry: Optional[str]
    ) -> List[str]:
        items = list(history or [])
        if entry:
            items.append(entry)
            if len(items) > self.MAX_HISTORY:
                items = items[-self.MAX_HISTORY :]
        return items

    def _render_board(
        self,
        tokens: Optional[Dict[PlayerColor, List[TokenView]]],
        show_flag: Optional[bool],
    ) -> Optional[str]:
        if tokens is None:
            return None
        show_ids = bool(show_flag) if show_flag is not None else self.show_token_ids
        pil_img = draw_board(tokens, show_ids=show_ids)
        return self.utils.img_to_data_uri(pil_img)

    def _current_player_html(self, session: Optional[SessionState]) -> str:
        if session is None:
            return "<h3>🎯 Current Player: Game not started</h3>"
        if session.game_over:
            if session.winner_color is not None:
                return (
                    f"<h3>🏆 Winner: {session.winner_color.display_name}!</h3>"
                )
            return "<h3>🏁 Game over</h3>"

        color = INDEX_TO_COLOR.get(session.current_player_index, PlayerColor.BLUE)
        return (
            f"<h3 style='color: {color.value};'>🎯 Current Player: {color.display_name}</h3>"
        )

    def _format_move_choices(self, options: Sequence[dict]) -> str:
        if not options:
            return ""
        items = "".join(
            f"<li><strong>Token {opt['token_id']}</strong>: {opt['description']} ({opt['move_type']})</li>"
            for opt in options
        )
        return f"<h4>Choose your move:</h4><ul>{items}</ul>"

    def _build_move_buttons(self, options: Sequence[dict]) -> Tuple[gr.Update, ...]:
        updates: List[gr.Update] = []
        for idx in range(self.BUTTON_COUNT):
            if idx < len(options):
                token_id = options[idx]["token_id"]
                updates.append(
                    gr.update(visible=True, value=f"Move Token {token_id}")
                )
            else:
                updates.append(gr.update(visible=False))
        return tuple(updates)

    def _step_response(
        self,
        session: Optional[SessionState],
        *,
        tokens: Optional[Dict[PlayerColor, List[TokenView]]],
        description: str,
        history: List[str],
        waiting: bool,
        move_options: List[dict],
        show_ids: Optional[bool],
        pending_dice: Optional[int],
        selected_token_id: Optional[int],
        auto_steps_remaining: int,
        auto_delay_state,
        controls_visible: Optional[bool] = None,
        moves_html: Optional[str] = None,
    ):
        token_map = tokens or (
            self.game_manager.game_state_tokens(session) if session else None
        )
        board_html = self._render_board(token_map, show_ids)
        current_html = self._current_player_html(session)
        moves_html_str = (
            moves_html if moves_html is not None else self._format_move_choices(move_options)
        )
        controls_update = gr.update(
            visible=controls_visible if controls_visible is not None else waiting
        )
        button_updates = self._build_move_buttons(move_options)

        return (
            session,
            board_html,
            description,
            history,
            waiting,
            current_html,
            moves_html_str,
            controls_update,
            button_updates[0],
            button_updates[1],
            button_updates[2],
            button_updates[3],
            move_options,
            pending_dice,
            selected_token_id,
            auto_steps_remaining,
            auto_delay_state,
        )

    def _no_game_response(
        self, history: Optional[List[str]], show_ids: Optional[bool], auto_delay_state
    ):
        history_list = list(history or [])
        return self._step_response(
            None,
            tokens=None,
            description="No game initialized",
            history=history_list,
            waiting=False,
            move_options=[],
            show_ids=show_ids,
            pending_dice=None,
            selected_token_id=None,
            auto_steps_remaining=0,
            auto_delay_state=auto_delay_state,
            controls_visible=False,
        )

    def _no_moves_response(
        self,
        session: SessionState,
        history: Optional[List[str]],
        show_ids: Optional[bool],
        move_opts: List[dict],
        pending_dice: Optional[int],
        auto_steps_remaining: int,
        auto_delay_state,
    ):
        history_list = list(history or [])
        return self._step_response(
            session,
            tokens=self.game_manager.game_state_tokens(session),
            description="No moves available",
            history=history_list,
            waiting=False,
            move_options=move_opts,
            show_ids=show_ids,
            pending_dice=pending_dice,
            selected_token_id=None,
            auto_steps_remaining=auto_steps_remaining,
            auto_delay_state=auto_delay_state,
            controls_visible=False,
            moves_html="",
        )

    def _create_win_rate_chart(
        self,
        win_counts: Dict[PlayerColor, int],
        completed: int,
        strategies: Sequence[str],
    ) -> str:
        if completed == 0:
            return "<div style='text-align:center;padding:40px;color:#666;'>No data to display</div>"

        colors = {
            PlayerColor.RED: "#dc3545",
            PlayerColor.GREEN: "#28a745",
            PlayerColor.YELLOW: "#ffc107",
            PlayerColor.BLUE: "#0d6efd",
        }

        chart_html = [
            "<div style='padding:20px;'>",
            "<h4 style='text-align:center;margin-bottom:20px;color:#333;'>🏆 Win Rate Comparison</h4>",
            "<div style='max-width:600px;margin:0 auto;'>",
        ]

        max_wins = max(win_counts.values()) if win_counts else 0
        max_wins = max_wins or 1

        for idx, color in enumerate(self.default_players):
            wins = win_counts.get(color, 0)
            win_rate = wins / completed if completed else 0.0
            bar_width = wins / max_wins * 100 if max_wins else 0.0
            css_color = colors.get(color, "#6c757d")
            strategy = strategies[idx] if idx < len(strategies) else "random"

            chart_html.extend(
                [
                    "<div style='margin:15px 0;'>",
                    "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;'>",
                    f"<span style='font-weight:bold;color:{css_color};'>{color.display_name}</span>",
                    f"<span style='font-size:0.9em;color:#666;'>{strategy}</span>",
                    f"<span style='font-weight:bold;'>{win_rate * 100:.1f}%</span>",
                    "</div>",
                    "<div style='background:#f8f9fa;border-radius:10px;height:25px;position:relative;overflow:hidden;'>",
                    f"<div style='background:{css_color};height:100%;width:{bar_width:.1f}%;border-radius:10px;transition:width 0.5s ease;display:flex;align-items:center;justify-content:center;'>",
                    f"<span style='color:white;font-size:0.8em;font-weight:bold;'>{wins}</span>",
                    "</div>",
                    "</div>",
                    "</div>",
                ]
            )

        chart_html.extend(
            [
                "</div>",
                f"<div style='text-align:center;margin-top:20px;font-size:0.9em;color:#666;'>📊 Based on {completed:,} completed games</div>",
                "</div>",
            ]
        )

        return "".join(chart_html)

    def _coerce_delay(self, delay) -> float:
        try:
            return max(float(delay), 0.0)
        except (TypeError, ValueError):
            return 0.0