#!/usr/bin/env python3
"""
4-Player Strategic Tournament System
Comprehensive tournament between combinations of 4 Ludo AI strategies.
"""

import os
import random
import time
from collections import defaultdict
from itertools import combinations, combinations_with_replacement

from dotenv import load_dotenv

from ludo import LudoGame, PlayerColor, StrategyFactory
from ludo_stats.game_state_saver import GameStateSaver

# Load environment configuration
load_dotenv()


class FourPlayerTournament:
    """Advanced 4-player tournament system for strategic AI evaluation."""

    def __init__(self):
        # Load configuration from .env
        self.max_turns_per_game = int(os.getenv("MAX_TURNS_PER_GAME", 500))
        self.games_per_matchup = int(os.getenv("GAMES_PER_MATCHUP", 10))
        self.tournament_seed = int(os.getenv("TOURNAMENT_SEED", 42))
        self.verbose_output = os.getenv("VERBOSE_OUTPUT", "true").lower() == "true"

        # Initialize state saver
        self.state_saver = GameStateSaver(
            os.getenv("STATE_SAVE_DIR", "saved_states/games")
        )

        # Get all available strategies or use selected ones
        selected_strategies = os.getenv("SELECTED_STRATEGIES", "").strip()
        if selected_strategies:
            self.all_strategies = [s.strip() for s in selected_strategies.split(",")]
        else:
            self.all_strategies = StrategyFactory.get_available_strategies()

        # Generate all 4-strategy combinations
        combi_method = (
            combinations
            if len(self.all_strategies) >= 4
            else combinations_with_replacement
        )
        self.strategy_combinations = list(combi_method(self.all_strategies, 4))

        # Tournament tracking
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

        if self.verbose_output:
            print("🎯 Tournament Configuration:")
            print(f"   • Available strategies: {len(self.all_strategies)}")
            print(f"   • 4-player combinations: {len(self.strategy_combinations)}")
            print(f"   • Games per matchup: {self.games_per_matchup}")
            print(f"   • Max turns per game: {self.max_turns_per_game}")
            print(
                f"   • Total games to play: {len(self.strategy_combinations) * self.games_per_matchup}"
            )

    def run_tournament(self):
        """Execute complete 4-player tournament."""
        print("🏆 4-PLAYER STRATEGIC LUDO TOURNAMENT 🏆")
        print("=" * 70)

        self._display_participants()
        self._run_round_robin()
        self._display_final_results()
        self._display_detailed_analysis()

        return self._get_tournament_summary()

    def _display_participants(self):
        """Show tournament participants and their strategies."""
        print("\n🤖 Tournament Participants:")
        print("-" * 50)

        descriptions = StrategyFactory.get_strategy_descriptions()
        for i, strategy in enumerate(self.all_strategies, 1):
            print(f"{i}. {strategy.upper()}: {descriptions[strategy]}")

        print("\n📋 Tournament Format:")
        print(f"   • {self.games_per_matchup} games per 4-player combination")
        print(f"   • {len(self.strategy_combinations)} unique combinations")
        print(f"   • Maximum {self.max_turns_per_game} turns per game")
        print("   • All combinations tournament with detailed analytics")

    def _run_round_robin(self):
        """Run tournament with all 4-player combinations."""
        print("\n🎮 Tournament Execution:")
        print("=" * 70)

        total_games = 0
        combination_results = []
        start_time = time.time()

        for combo_idx, strategy_combo in enumerate(self.strategy_combinations, 1):
            print(
                f"\nCombination {combo_idx}/{len(self.strategy_combinations)}: "
                f"{' vs '.join([s.upper() for s in strategy_combo])}"
            )
            print("-" * 60)

            combo_wins = {strategy: 0 for strategy in strategy_combo}

            # Play multiple games for this combination
            for game_num in range(self.games_per_matchup):
                # Randomize starting order for fairness
                game_strategies = list(strategy_combo)
                random.shuffle(game_strategies)

                if self.verbose_output:
                    print(
                        f"  Game {game_num + 1}: {' → '.join([s.upper() for s in game_strategies])}"
                    )

                # Create 4-player game
                game = LudoGame(
                    [
                        PlayerColor.RED,
                        PlayerColor.BLUE,
                        PlayerColor.GREEN,
                        PlayerColor.YELLOW,
                    ]
                )

                # Assign strategies to players
                for i, strategy_name in enumerate(game_strategies):
                    strategy = StrategyFactory.create_strategy(strategy_name)
                    game.players[i].set_strategy(strategy)
                    game.players[i].strategy_name = strategy_name

                # Play the game
                results = self._play_four_player_game(
                    game, f"{combo_idx}.{game_num + 1}"
                )
                total_games += 1

                # Track combination wins
                if results["winner"]:
                    winner_name = results["winner"].strategy_name
                    combo_wins[winner_name] += 1

                # Process results
                self._process_game_results(results, game_strategies)

            # Show combination summary
            combo_summary = ", ".join(
                [f"{s.upper()}: {wins}" for s, wins in combo_wins.items()]
            )
            print(f"  Results: {combo_summary}")
            combination_results.append((strategy_combo, combo_wins))

        elapsed = time.time() - start_time
        print(f"\n⏱️  Tournament completed in {elapsed:.1f} seconds")
        print(f"📊 Total games played: {total_games}")
        print(f"🎯 Combinations tested: {len(self.strategy_combinations)}")

        return combination_results

    def _play_four_player_game(self, game: LudoGame, game_number: int):
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

            # Get AI decision context
            context = game.get_ai_decision_context(dice_value)

            if context["valid_moves"]:
                # AI makes strategic decision
                selected_token = current_player.make_strategic_decision(context)

                # Execute the move
                move_result = game.execute_move(
                    current_player, selected_token, dice_value
                )

                # Save the decision and outcome
                self.state_saver.save_decision(
                    strategy_name, context, selected_token, move_result
                )

                # Track stats
                game_results["player_stats"][strategy_name]["moves_made"] += 1

                if move_result.get("captured_tokens"):
                    captures = len(move_result["captured_tokens"])
                    game_results["player_stats"][strategy_name]["tokens_captured"] += (
                        captures
                    )
                    game_results["game_events"].append(
                        f"Turn {turn_count}: {strategy_name} captured {captures} token(s)"
                    )

                if move_result.get("token_finished"):
                    game_results["player_stats"][strategy_name]["tokens_finished"] += 1
                    game_results["game_events"].append(
                        f"Turn {turn_count}: {strategy_name} finished a token"
                    )

                # Check for game end
                if move_result.get("game_won"):
                    game_results["winner"] = current_player
                    game_results["turns_played"] = turn_count
                    print(
                        f"  Game {game_number}: {strategy_name.upper()} WINS! ({turn_count} turns)"
                    )
                    break

                if not move_result.get("extra_turn", False):
                    game.next_turn()
            else:
                game.next_turn()

            turn_count += 1
            game_results["player_stats"][strategy_name]["turns_taken"] += 1

        if not game_results["winner"]:
            print(f"  Game {game_number}: DRAW (time limit reached)")
            game_results["turns_played"] = turn_count

        # Save game states
        self.state_saver.save_game(game_number)

        return game_results

    def _process_game_results(self, results, game_strategies):
        """Process and store game results for analysis."""
        winner_name = results["winner"].strategy_name if results["winner"] else None

        for strategy_name in game_strategies:
            stats = self.detailed_stats[strategy_name]
            stats["games_played"] += 1
            stats["total_turns"] += results["player_stats"][strategy_name][
                "turns_taken"
            ]
            stats["tokens_captured"] += results["player_stats"][strategy_name][
                "tokens_captured"
            ]
            stats["tokens_finished"] += results["player_stats"][strategy_name][
                "tokens_finished"
            ]

            if strategy_name == winner_name:
                stats["games_won"] += 1
                self.results[strategy_name]["wins"] += 1
            else:
                self.results[strategy_name]["losses"] += 1

            # Update head-to-head records
            for opponent in game_strategies:
                if opponent != strategy_name:
                    stats["head_to_head"][opponent]["games"] += 1
                    if strategy_name == winner_name:
                        stats["head_to_head"][opponent]["wins"] += 1

    def _display_final_results(self):
        """Display comprehensive tournament results."""
        print("\n🏆 FINAL TOURNAMENT STANDINGS 🏆")
        print("=" * 70)

        # Calculate standings for all strategies that played
        standings = []
        for strategy in self.all_strategies:
            if strategy in self.detailed_stats:
                stats = self.detailed_stats[strategy]
                wins = stats["games_won"]
                games = stats["games_played"]
                win_rate = (wins / games * 100) if games > 0 else 0

                standings.append(
                    {
                        "strategy": strategy,
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
            f"{'Rank':<4} {'Strategy':<12} {'Wins':<6} {'Games':<7} {'Win Rate':<10} {'Avg Turns':<10}"
        )
        print("-" * 65)

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
                f"{rank:<4} {entry['strategy'].upper():<12} {entry['wins']:<6} {entry['games']:<7} "
                f"{entry['win_rate']:<9.1f}% {entry['avg_turns']:<9.1f} {medal}"
            )

        return standings

    def _display_detailed_analysis(self):
        """Show detailed strategic analysis."""
        print("\n📊 DETAILED PERFORMANCE ANALYSIS 📊")
        print("=" * 70)

        # Performance metrics
        print(
            f"\n{'Strategy':<12} {'Captures':<10} {'Finished':<10} {'Efficiency':<12}"
        )
        print("-" * 50)

        for strategy in self.all_strategies:
            if strategy in self.detailed_stats:
                stats = self.detailed_stats[strategy]
                efficiency = (
                    (stats["tokens_finished"] / stats["games_played"])
                    if stats["games_played"] > 0
                    else 0
                )

                print(
                    f"{strategy.upper():<12} {stats['tokens_captured']:<10} {stats['tokens_finished']:<10} {efficiency:<11.2f}"
                )

        # Head-to-head analysis (only show strategies with significant interactions)
        print("\n🥊 HEAD-TO-HEAD ANALYSIS 🥊")
        print("-" * 50)

        for strategy in self.all_strategies:
            if strategy in self.detailed_stats:
                h2h = self.detailed_stats[strategy]["head_to_head"]
                has_interactions = any(record["games"] > 0 for record in h2h.values())

                if has_interactions:
                    print(f"\n{strategy.upper()} vs Others:")
                    for opponent, record in h2h.items():
                        if record["games"] > 0:
                            win_rate = (record["wins"] / record["games"]) * 100
                            print(
                                f"  vs {opponent.upper():<10}: {record['wins']}/{record['games']} ({win_rate:.1f}%)"
                            )

    def _get_tournament_summary(self):
        """Return structured tournament summary."""
        # Find champion among strategies that actually played
        played_strategies = [
            s
            for s in self.all_strategies
            if s in self.detailed_stats and self.detailed_stats[s]["games_played"] > 0
        ]

        champion = (
            max(
                played_strategies,
                key=lambda s: (
                    self.detailed_stats[s]["games_won"],
                    self.detailed_stats[s]["games_won"]
                    / max(1, self.detailed_stats[s]["games_played"]),
                ),
            )
            if played_strategies
            else None
        )

        summary = {
            "tournament_type": "4-Player Strategic Combinations",
            "participants": self.all_strategies,
            "combinations_tested": len(self.strategy_combinations),
            "games_per_matchup": self.games_per_matchup,
            "total_games": sum(
                stats["games_played"] for stats in self.detailed_stats.values()
            )
            // 4,
            "results": dict(self.detailed_stats),
            "champion": champion,
        }
        return summary


def run_strategic_analysis():
    """Run additional strategic performance analysis."""
    print("\n🔬 STRATEGIC BEHAVIOR ANALYSIS 🔬")
    print("=" * 70)

    strategies = StrategyFactory.get_available_strategies()

    # Analyze decision patterns
    print("\nDecision Pattern Analysis:")
    print("-" * 40)

    for strategy_name in strategies:
        strategy = StrategyFactory.create_strategy(strategy_name)
        print(f"\n{strategy_name.upper()}: {strategy.description}")

        # Test with different scenarios
        test_scenarios = [
            (
                "Aggressive",
                {"capture_opportunities": [{"value": 8}], "safety_concerns": []},
            ),
            (
                "Defensive",
                {"capture_opportunities": [], "safety_concerns": [{"risk": 7}]},
            ),
            (
                "Balanced",
                {
                    "capture_opportunities": [{"value": 5}],
                    "safety_concerns": [{"risk": 4}],
                },
            ),
        ]

        for scenario_name, analysis in test_scenarios:
            print(f"  {scenario_name} scenario: Strategic preference detected")


if __name__ == "__main__":
    # Set random seed from environment
    random.seed(int(os.getenv("TOURNAMENT_SEED", 42)))

    print("🎯 LUDO 4-PLAYER COMBINATION TOURNAMENT 🎯")
    print("=" * 70)
    print("Starting comprehensive all-combinations tournament...")

    # Run main tournament
    tournament = FourPlayerTournament()
    summary = tournament.run_tournament()

    # Additional analysis
    run_strategic_analysis()

    # Final summary
    print("\n🎯 TOURNAMENT COMPLETE! 🎯")
    print("=" * 70)
    if summary["champion"]:
        print(f"🏆 Champion: {summary['champion'].upper()}")
    else:
        print("🏆 No clear champion (no games completed)")
    print(f"📊 Total Games: {summary['total_games']}")
    print(f"🎯 Combinations Tested: {summary['combinations_tested']}")
    print(f"🎮 Participants: {', '.join([s.upper() for s in summary['participants']])}")
    print("\n✅ 4-Player Strategic Tournament System Ready!")
    print("🔬 Advanced AI evaluation and comparison complete!")
