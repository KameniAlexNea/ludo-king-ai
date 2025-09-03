#!/usr/bin/env python3
"""
PPO vs Strategies Tournament System
Tournament pitting the best PPO model against all available Ludo strategies.
"""

import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import random
import time
from collections import defaultdict
from itertools import combinations

import numpy as np
from dotenv import load_dotenv

from ludo import LudoGame, PlayerColor, StrategyFactory
# Dynamic PPO strategy & EnvConfig import handled after CLI args (see FourPlayerPPOTournament._load_ppo_wrapper)
from ludo_stats.game_state_saver import GameStateSaver

load_dotenv()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PPO vs Strategies Tournament System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        default=int(os.getenv("MAX_TURNS_PER_GAME", 1000)),
        help="Maximum turns per game before declaring draw"
    )

    parser.add_argument(
        "--games-per-matchup",
        type=int,
        default=int(os.getenv("GAMES_PER_MATCHUP", 10)),
        help="Number of games to play per strategy combination"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Tournament random seed"
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directory containing PPO model files"
    )

    parser.add_argument(
        "--strategies",
        type=str,
        nargs="*",
        help="Specific strategies to include (default: all available)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbose output"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="saved_states/ppo_vs_strategies",
        help="Directory to save game states and results"
    )

    parser.add_argument(
        "--env",
        choices=["single-seat", "classic"],
        default="single-seat",
        help="Which PPO environment/wrapper to use: single-seat (ludo_rls) or classic multi-seat (ludo_rl)."
    )

    return parser.parse_args()


def run_game_with_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


class FourPlayerPPOTournament:
    """Tournament system pitting PPO model against Ludo strategies."""

    def __init__(self, args):
        # Configuration from arguments
        self.max_turns_per_game = args.max_turns
        self.games_per_matchup = args.games_per_matchup
        self.tournament_seed = args.seed
        self.models_dir = args.models_dir
        self.verbose_output = not args.quiet
        self.output_dir = args.output_dir
        self.selected_strategies = args.strategies
        self.env_kind = args.env  # 'single-seat' or 'classic'

        # Load appropriate PPO wrapper & EnvConfig dynamically
        self._load_ppo_wrapper()

        # Initialize state saver
        self.state_saver = GameStateSaver(self.output_dir)

        # Select the best PPO model (use FINAL or the one with most steps)
        self.ppo_model = self._select_best_ppo_model()

        # Get available strategies
        self.all_strategies = self._get_strategies()

        # Generate combinations: PPO + 3 different strategies
        self.strategy_combinations = list(combinations(self.all_strategies, 3))

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
            print("🎯 PPO vs Strategies Tournament Configuration:")
            print(f"   • PPO Model: {self.ppo_model}")
            print(f"   • Available strategies: {len(self.all_strategies)}")
            print(f"   • Strategy combinations: {len(self.strategy_combinations)}")
            print(f"   • Games per matchup: {self.games_per_matchup}")
            print(f"   • Max turns per game: {self.max_turns_per_game}")
            print(f"   • Models directory: {self.models_dir}")
            print(f"   • Output directory: {self.output_dir}")
            print(f"   • Environment mode: {self.env_kind}")
            print(
                f"   • Total games to play: {len(self.strategy_combinations) * self.games_per_matchup}"
            )

    def _load_ppo_wrapper(self):
        """Dynamically import the correct EnvConfig and PPOStrategy based on env_kind."""
        if self.env_kind == "classic":
            from ludo_rl.envs.model import EnvConfig as ClassicEnvConfig
            from ludo_rl.ppo_strategy import PPOStrategy as ClassicPPOStrategy

            self.EnvConfigClass = ClassicEnvConfig
            self.PPOStrategyClass = ClassicPPOStrategy
        else:  # single-seat
            from ludo_rls.envs.model import EnvConfig as SingleEnvConfig
            from ludo_rls.ppo_strategy import PPOStrategy as SinglePPOStrategy

            self.EnvConfigClass = SingleEnvConfig
            self.PPOStrategyClass = SinglePPOStrategy

    def _get_strategies(self):
        """Get strategies to use in tournament."""
        if self.selected_strategies:
            return self.selected_strategies
        if os.getenv("SELECTED_STRATEGIES"):
            return list(os.getenv("SELECTED_STRATEGIES").split(","))
        return StrategyFactory.get_available_strategies()

    def _select_best_ppo_model(self):
        """Select the best PPO model (prefer FINAL, then highest step count)."""
        if not os.path.exists(self.models_dir):
            raise FileNotFoundError(f"Models directory {self.models_dir} not found")

        model_files = [f for f in os.listdir(self.models_dir) if f.endswith(".zip")]
        if not model_files:
            raise FileNotFoundError(f"No PPO model files found in {self.models_dir}/")

        # Prefer FINAL model, then highest step count
        final_model = next((f for f in model_files if "final" in f.lower()), None)
        if final_model:
            return final_model.replace(".zip", "")

        # Extract step numbers and find highest
        step_models = []
        for f in model_files:
            try:
                # Extract number from filename like "ppo_ludo_1000000_steps"
                parts = f.replace(".zip", "").split("_")
                for part in parts:
                    if part.isdigit():
                        step_models.append((int(part), f.replace(".zip", "")))
                        break
            except Exception:
                continue

        if step_models:
            step_models.sort(reverse=True)
            return step_models[0][1]

        # Fallback to first model
        return model_files[0].replace(".zip", "")

    def run_tournament(self):
        """Execute PPO vs Strategies tournament."""
        print("🏆 PPO vs STRATEGIES TOURNAMENT 🏆")
        print("=" * 70)

        self._display_participants()
        self._run_round_robin()
        self._display_final_results()
        self._display_detailed_analysis()

        return self._get_tournament_summary()

    def _display_participants(self):
        """Show tournament participants."""
        print("\n🤖 Tournament Participants:")
        print("-" * 50)
        print(f"PPO: {self.ppo_model.upper()}")
        print("\nStrategies:")

        descriptions = StrategyFactory.get_strategy_descriptions()
        for i, strategy in enumerate(self.all_strategies, 1):
            desc = descriptions.get(strategy, "No description")
            print(f"{i}. {strategy.upper()}: {desc}")

        print("\n📋 Tournament Format:")
        print(f"   • {self.games_per_matchup} games per 4-player combination")
        print(f"   • {len(self.strategy_combinations)} unique combinations")
        print(f"   • Maximum {self.max_turns_per_game} turns per game")
        print("   • All combinations tournament with detailed analytics")

    def _run_round_robin(self):
        """Run tournament with PPO vs strategy combinations."""
        print("\n🎮 Tournament Execution:")
        print("=" * 70)

        total_games = 0
        combination_results = []
        start_time = time.time()

        for combo_idx, strategy_combo in enumerate(self.strategy_combinations, 1):
            # Create combination: PPO + 3 strategies
            game_players = [self.ppo_model] + list(strategy_combo)

            print(
                f"\nCombination {combo_idx}/{len(self.strategy_combinations)}: "
                f"{' vs '.join([p.upper() for p in game_players])}"
            )
            print("-" * 60)

            combo_wins = {player: 0 for player in game_players}

            # Play multiple games for this combination
            for game_num in range(self.games_per_matchup):
                # Randomize starting order for fairness
                random.shuffle(game_players)

                if self.verbose_output:
                    print(
                        f"  Game {game_num + 1}: {' → '.join([p.upper() for p in game_players])}"
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

                # Assign strategies ensuring PPO always sits at RED (index 0)
                # Find PPO index in randomized order
                if self.ppo_model in game_players:
                    ppo_idx = game_players.index(self.ppo_model)
                    # Swap to front if not already
                    if ppo_idx != 0:
                        game_players[0], game_players[ppo_idx] = (
                            game_players[ppo_idx],
                            game_players[0],
                        )
                for i, player_name in enumerate(game_players):
                    if player_name == self.ppo_model:
                        model_path = f"{self.models_dir}/{player_name}.zip"
                        # Force agent_color red to match training seat (common assumption)
                        strategy = self.PPOStrategyClass(
                            model_path, player_name, self.EnvConfigClass(agent_color="red")
                        )
                    else:
                        strategy = StrategyFactory.create_strategy(player_name)
                    game.players[i].set_strategy(strategy)
                    game.players[i].strategy_name = player_name

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
                self._process_game_results(results, game_players)

            # Show combination summary
            combo_summary = ", ".join(
                [f"{p.upper()}: {wins}" for p, wins in combo_wins.items()]
            )
            print(f"  Results: {combo_summary}")
            combination_results.append((game_players, combo_wins))

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
            context = game.get_game_state_for_ai()
            context["dice_value"] = dice_value  # Add dice value to context

            if context["valid_moves"]:
                # PPO makes strategic decision
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

    def _process_game_results(self, results, game_models):
        """Process and store game results for analysis."""
        winner_name = results["winner"].strategy_name if results["winner"] else None

        for model_name in game_models:
            stats = self.detailed_stats[model_name]
            stats["games_played"] += 1
            stats["total_turns"] += results["player_stats"][model_name]["turns_taken"]
            stats["tokens_captured"] += results["player_stats"][model_name][
                "tokens_captured"
            ]
            stats["tokens_finished"] += results["player_stats"][model_name][
                "tokens_finished"
            ]

            if model_name == winner_name:
                stats["games_won"] += 1
                self.results[model_name]["wins"] += 1
            else:
                self.results[model_name]["losses"] += 1

            # Update head-to-head records
            for opponent in game_models:
                if opponent != model_name:
                    stats["head_to_head"][opponent]["games"] += 1
                    if model_name == winner_name:
                        stats["head_to_head"][opponent]["wins"] += 1

    def _display_final_results(self):
        """Display comprehensive tournament results."""
        print("\n🏆 FINAL TOURNAMENT STANDINGS 🏆")
        print("=" * 70)

        # Calculate standings for all participants that played
        standings = []

        # Add PPO model if it played
        if self.ppo_model in self.detailed_stats:
            stats = self.detailed_stats[self.ppo_model]
            wins = stats["games_won"]
            games = stats["games_played"]
            win_rate = (wins / games * 100) if games > 0 else 0

            standings.append(
                {
                    "model": self.ppo_model,
                    "wins": wins,
                    "games": games,
                    "win_rate": win_rate,
                    "avg_turns": stats["total_turns"] / games if games > 0 else 0,
                    "captures": stats["tokens_captured"],
                    "finished": stats["tokens_finished"],
                }
            )

        # Add strategies that played
        for strategy in self.all_strategies:
            if strategy in self.detailed_stats:
                stats = self.detailed_stats[strategy]
                wins = stats["games_won"]
                games = stats["games_played"]
                win_rate = (wins / games * 100) if games > 0 else 0

                standings.append(
                    {
                        "model": strategy,
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

    def _display_detailed_analysis(self):
        """Show detailed strategic analysis."""
        print("\n📊 DETAILED PERFORMANCE ANALYSIS 📊")
        print("=" * 70)

        # Performance metrics
        print(f"\n{'Model':<20} {'Captures':<10} {'Finished':<10} {'Efficiency':<12}")
        print("-" * 60)

        # Add PPO model
        if self.ppo_model in self.detailed_stats:
            stats = self.detailed_stats[self.ppo_model]
            efficiency = (
                (stats["tokens_finished"] / stats["games_played"])
                if stats["games_played"] > 0
                else 0
            )
            print(
                f"{self.ppo_model.upper():<20} {stats['tokens_captured']:<10} {stats['tokens_finished']:<10} {efficiency:<11.2f}"
            )

        # Add strategies
        for strategy in self.all_strategies:
            if strategy in self.detailed_stats:
                stats = self.detailed_stats[strategy]
                efficiency = (
                    (stats["tokens_finished"] / stats["games_played"])
                    if stats["games_played"] > 0
                    else 0
                )
                print(
                    f"{strategy.upper():<20} {stats['tokens_captured']:<10} {stats['tokens_finished']:<10} {efficiency:<11.2f}"
                )

        # Head-to-head analysis (only show models with significant interactions)
        print("\n🥊 HEAD-TO-HEAD ANALYSIS 🥊")
        print("-" * 50)

        # Check PPO model
        if self.ppo_model in self.detailed_stats:
            h2h = self.detailed_stats[self.ppo_model]["head_to_head"]
            has_interactions = any(record["games"] > 0 for record in h2h.values())

            if has_interactions:
                print(f"\n{self.ppo_model.upper()} vs Others:")
                for opponent, record in h2h.items():
                    if record["games"] > 0:
                        win_rate = (record["wins"] / record["games"]) * 100
                        print(
                            f"  vs {opponent.upper():<18}: {record['wins']}/{record['games']} ({win_rate:.1f}%)"
                        )

        # Check strategies
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
                                f"  vs {opponent.upper():<18}: {record['wins']}/{record['games']} ({win_rate:.1f}%)"
                            )

    def _get_tournament_summary(self):
        """Return structured tournament summary."""
        # Find champion among participants that actually played
        played_participants = []

        # Add PPO model if it played
        if (
            self.ppo_model in self.detailed_stats
            and self.detailed_stats[self.ppo_model]["games_played"] > 0
        ):
            played_participants.append(self.ppo_model)

        # Add strategies that played
        for strategy in self.all_strategies:
            if (
                strategy in self.detailed_stats
                and self.detailed_stats[strategy]["games_played"] > 0
            ):
                played_participants.append(strategy)

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
            "tournament_type": "PPO vs Strategies Tournament",
            "ppo_model": self.ppo_model,
            "strategies": self.all_strategies,
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


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Set random seed
    run_game_with_seed(args.seed)

    print("🎯 LUDO PPO vs STRATEGIES TOURNAMENT 🎯")
    print("=" * 70)
    print("Starting comprehensive PPO vs Strategies tournament...")

    # Run main tournament
    tournament = FourPlayerPPOTournament(args)
    summary = tournament.run_tournament()

    # Final summary
    print("\n🎯 TOURNAMENT COMPLETE! 🎯")
    print("=" * 70)
    if summary["champion"]:
        print(f"🏆 Champion: {summary['champion'].upper()}")
    else:
        print("🏆 No clear champion (no games completed)")
    print(f"📊 Total Games: {summary['total_games']}")
    print(f"🎯 Combinations Tested: {summary['combinations_tested']}")
    print(f"🤖 PPO Model: {summary['ppo_model'].upper()}")
    print(f"🎮 Strategies: {', '.join([s.upper() for s in summary['strategies']])}")
    print("\n✅ PPO vs Strategies Tournament System Ready!")
    print("🔬 Advanced PPO vs Strategies evaluation complete!")
