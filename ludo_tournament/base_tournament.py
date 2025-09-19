#!/usr/bin/env python3
"""
Base Tournament System for Ludo AI Tournaments
Shared functionality for different tournament types.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

from ludo_engine import LudoGame

from ludo_tournament.game_state_saver import GameStateSaver


class BaseTournament:
    """Base class for Ludo tournament systems with shared functionality."""

    def __init__(
        self,
        max_turns_per_game: int = 1000,
        state_saver: Optional[GameStateSaver] = None,
    ):
        """Initialize base tournament with common attributes."""
        self.max_turns_per_game = max_turns_per_game
        self.state_saver = state_saver

        # Initialize tracking structures
        self.results = defaultdict(lambda: defaultdict(int))
        self.detailed_stats = defaultdict(
            lambda: {
                "games_played": 0,
                "games_won": 0,
                "total_turns": 0,
                "tokens_captured": 0,
                "tokens_finished": 0,
                "average_finish_position": 0,
                "head_to_head": defaultdict(lambda: {"wins": 0, "games": 0}),
            }
        )

    def _play_four_player_game(
        self, game: LudoGame, game_number: str, verbose_output: bool = True
    ) -> Dict[str, Any]:
        """Play a complete 4-player game and return detailed results."""
        turn_count = 0
        game_results = {
            "winner": None,
            "final_positions": [],
            "turns_played": 0,
            "game_events": [],
            "player_stats": {},
        }

        # Initialize player stats
        for player in game.players:
            game_results["player_stats"][player.strategy_name] = {
                "tokens_captured": 0,
                "tokens_finished": 0,
                "moves_made": 0,
                "turns_taken": 0,
            }

        while not game.game_over and turn_count < self.max_turns_per_game:
            current_player = game.get_current_player()
            strategy_name = current_player.strategy_name
            dice_value = game.roll_dice()

            context = game.get_ai_decision_context(dice_value)

            if context.valid_moves:
                selected_token = current_player.make_strategic_decision(context)
                move_result = game.execute_move(
                    current_player, selected_token, dice_value
                )

                # Save decision if state saver is available
                if self.state_saver:
                    self.state_saver.save_decision(
                        strategy_name, context, selected_token, move_result
                    )

                game_results["player_stats"][strategy_name]["moves_made"] += 1

                if move_result.captured_tokens:
                    captures = len(move_result.captured_tokens)
                    game_results["player_stats"][strategy_name]["tokens_captured"] += (
                        captures
                    )
                    game_results["game_events"].append(
                        f"Turn {turn_count}: {strategy_name} captured {captures} token(s)"
                    )

                if move_result.finished_token:
                    game_results["player_stats"][strategy_name]["tokens_finished"] += 1
                    game_results["game_events"].append(
                        f"Turn {turn_count}: {strategy_name} finished a token"
                    )

                if move_result.game_won:
                    game_results["winner"] = current_player
                    game_results["turns_played"] = turn_count
                    if verbose_output:
                        print(
                            f"  Game {game_number}: {strategy_name.upper()} WINS! ({turn_count} turns)"
                        )
                    break

                if not move_result.extra_turn:
                    game.next_turn()
            else:
                game.next_turn()

            turn_count += 1
            game_results["player_stats"][strategy_name]["turns_taken"] += 1

        if not game_results["winner"]:
            if verbose_output:
                print(f"  Game {game_number}: DRAW (time limit reached)")
            game_results["turns_played"] = turn_count

        # Save game states if state saver is available
        if self.state_saver:
            self.state_saver.save_game(game_number)

        return game_results

    def _process_game_results(
        self, results: Dict[str, Any], game_players: List[str]
    ) -> None:
        """Process and store game results for analysis."""
        winner_name = results["winner"].strategy_name if results["winner"] else None

        for player_name in game_players:
            stats = self.detailed_stats[player_name]
            stats["games_played"] += 1
            stats["total_turns"] += results["player_stats"][player_name]["turns_taken"]
            stats["tokens_captured"] += results["player_stats"][player_name][
                "tokens_captured"
            ]
            stats["tokens_finished"] += results["player_stats"][player_name][
                "tokens_finished"
            ]

            if player_name == winner_name:
                stats["games_won"] += 1
                self.results[player_name]["wins"] += 1
            else:
                self.results[player_name]["losses"] += 1

            # Update head-to-head records
            for opponent in game_players:
                if opponent != player_name:
                    stats["head_to_head"][opponent]["games"] += 1
                    if player_name == winner_name:
                        stats["head_to_head"][opponent]["wins"] += 1

    def _display_final_results(
        self, participants: List[str], title: str = "FINAL TOURNAMENT STANDINGS"
    ) -> List[Dict[str, Any]]:
        """Display comprehensive tournament results."""
        print(f"\n🏆 {title} 🏆")
        print("=" * 70)

        # Calculate standings for all participants that played
        standings = []

        for participant in participants:
            if participant in self.detailed_stats:
                stats = self.detailed_stats[participant]
                wins = stats["games_won"]
                games = stats["games_played"]
                win_rate = (wins / games * 100) if games > 0 else 0

                standings.append(
                    {
                        "model": participant,
                        "wins": wins,
                        "games": games,
                        "win_rate": win_rate,
                        "avg_turns": stats["total_turns"] / games if games > 0 else 0,
                        "captures": stats["tokens_captured"],
                        "finished": stats["tokens_finished"],
                    }
                )

        # Sort by win rate, then by total wins
        standings.sort(key=lambda x: (x["win_rate"], x["wins"]), reverse=True)

        # Display standings table
        print(
            f"{'Rank':<4} {'Model':<20} {'Wins':<6} {'Games':<7} {'Win Rate':<10} {'Avg Turns':<10}"
        )
        print("-" * 75)

        for rank, entry in enumerate(standings, 1):
            medal = (
                "🥇"
                if rank == 1
                else "🥈"
                if rank == 2
                else "🥉"
                if rank == 3
                else "  "
            )
            print(
                f"{rank:<4} {entry['model'].upper():<20} {entry['wins']:<6} {entry['games']:<7} "
                f"{entry['win_rate']:<9.1f}% {entry['avg_turns']:<9.1f} {medal}"
            )

        return standings

    def _display_detailed_analysis(self, participants: List[str]) -> None:
        """Show detailed strategic analysis."""
        print("\n📊 DETAILED PERFORMANCE ANALYSIS 📊")
        print("=" * 70)

        # Performance metrics
        print(
            f"\n{'Model':<20} {'Captures':<10} {'Finished':<10} {'Efficiency':<12}"
        )
        print("-" * 60)

        for participant in participants:
            if participant in self.detailed_stats:
                stats = self.detailed_stats[participant]
                efficiency = (
                    (stats["tokens_finished"] / stats["games_played"])
                    if stats["games_played"] > 0
                    else 0
                )
                print(
                    f"{participant.upper():<20} {stats['tokens_captured']:<10} {stats['tokens_finished']:<10} {efficiency:<11.2f}"
                )

        # Head-to-head analysis (only show models with significant interactions)
        print("\n🥊 HEAD-TO-HEAD ANALYSIS 🥊")
        print("-" * 50)

        for participant in participants:
            if participant in self.detailed_stats:
                h2h = self.detailed_stats[participant]["head_to_head"]
                has_interactions = any(record["games"] > 0 for record in h2h.values())

                if has_interactions:
                    print(f"\n{participant.upper()} vs Others:")
                    for opponent, record in h2h.items():
                        if record["games"] > 0:
                            win_rate = (record["wins"] / record["games"]) * 100
                            print(
                                f"  vs {opponent.upper():<18}: {record['wins']}/{record['games']} ({win_rate:.1f}%)"
                            )

    def _get_tournament_summary(
        self, participants: List[str], tournament_type: str
    ) -> Dict[str, Any]:
        """Return structured tournament summary."""
        # Find champion among participants that actually played
        played_participants = [
            p
            for p in participants
            if p in self.detailed_stats and self.detailed_stats[p]["games_played"] > 0
        ]

        champion = (
            max(
                played_participants,
                key=lambda p: (
                    self.detailed_stats[p]["games_won"],
                    self.detailed_stats[p]["games_won"]
                    / max(1, self.detailed_stats[p]["games_played"]),
                ),
            )
            if played_participants
            else None
        )

        summary = {
            "tournament_type": tournament_type,
            "participants": participants,
            "total_games": sum(
                stats["games_played"] for stats in self.detailed_stats.values()
            )
            // 4,  # each game increments games_played for 4 players
            "results": dict(self.detailed_stats),
            "champion": champion,
        }
        return summary
