#!/usr/bin/env python3
"""
PPO vs Strategies Tournament System
Tournament pitting the best PPO model against all available Ludo strategies.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import random
import time
from itertools import combinations

import numpy as np
from dotenv import load_dotenv
from ludo_engine import LudoGame, StrategyFactory
from ludo_engine.models import ALL_COLORS

from ludo_tournament import BaseTournament
from ludo_tournament.game_state_saver import GameStateSaver
from ludo_tournament.load_ppo_model_base import load_ppo_strategy, select_best_ppo_model
from ludo_tournament.tournament_args import parse_arguments

load_dotenv()


def seed_environ(seed):
    random.seed(seed)
    np.random.seed(seed)


class FourPlayerPPOTournament(BaseTournament):
    """Tournament system pitting PPO model against Ludo strategies."""

    def __init__(self, args):
        # Core configuration
        self.max_turns_per_game = args.max_turns
        self.games_per_matchup = args.games_per_matchup
        self.tournament_seed = args.seed
        self.models_dir: str = args.models_dir
        self.verbose_output = not args.quiet
        self.output_dir = args.output_dir
        self.selected_strategies = args.strategies
        self.env_kind = args.env  # 'single-seat' or 'classic'
        self.model_preference = args.model_preference  # 'best', 'final', or 'steps'

        # Initialize state saver only if output directory is specified
        state_saver = GameStateSaver(self.output_dir) if self.output_dir else None

        # Initialize parent class
        super().__init__(
            max_turns_per_game=self.max_turns_per_game, state_saver=state_saver
        )

        # Resolve model name & path
        if not os.path.isdir(self.models_dir):
            if os.path.isfile(self.models_dir):
                models_dir = os.path.dirname(self.models_dir)
                self.ppo_model = os.path.basename(self.models_dir).replace(".zip", "")
                self.models_dir = models_dir
            else:
                raise ValueError(f"Models directory '{self.models_dir}' does not exist")
        else:
            self.ppo_model = select_best_ppo_model(self.models_dir, self.model_preference)
        self.ppo_model_path = os.path.join(self.models_dir, f"{self.ppo_model}.zip")

        # PPO policy will be accessed via PPOStrategyClass wrapper uniformly.

        # Get strategies universe
        self.all_strategies = self._get_strategies()

        # Compute opponent combinations (3 strategies vs PPO)
        self.strategy_combinations = list(combinations(self.all_strategies, 3))

        if self.verbose_output:
            print("🎯 PPO vs Strategies Tournament Configuration:")
            print(f"   • PPO Model: {self.ppo_model}")
            print(f"   • Model Preference: {self.model_preference}")
            print(f"   • Available strategies: {len(self.all_strategies)}")
            print(f"   • Strategy combinations: {len(self.strategy_combinations)}")
            print(f"   • Games per matchup: {self.games_per_matchup}")
            print(f"   • Max turns per game: {self.max_turns_per_game}")
            print(f"   • Models directory: {self.models_dir}")
            print(
                f"   • Output directory: {self.output_dir if self.output_dir else 'None (no saving)'}"
            )
            print(f"   • Environment mode: {self.env_kind}")
            print(
                f"   • Total games to play: {len(self.strategy_combinations) * self.games_per_matchup}"
            )

    def _get_strategies(self):
        """Get strategies to use in tournament."""
        if self.selected_strategies:
            return self.selected_strategies
        if os.getenv("SELECTED_STRATEGIES"):
            return list(os.getenv("SELECTED_STRATEGIES").split(","))
        return [
            i for i in StrategyFactory.get_available_strategies() if "human" not in i
        ]

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
                colours = ALL_COLORS
                game = LudoGame(colours)

                # Assign strategies. Always wrap PPO model in its strategy class for consistency across env modes.
                for i, (player_name, colour) in enumerate(zip(game_players, colours)):
                    if player_name == self.ppo_model:
                        strategy = load_ppo_strategy(
                            env_kind=self.env_kind,
                            models_dir=self.models_dir,
                            player_name=player_name,
                            agent_color=colour,
                            model_preference=self.model_preference,
                            game=game,
                            max_turns=self.max_turns_per_game,
                        )
                    else:
                        strategy = StrategyFactory.create_strategy(player_name)
                    game.players[i].set_strategy(strategy)
                    game.players[i].strategy_name = player_name

                # Play the game
                results = super()._play_four_player_game(
                    game, f"{combo_idx}.{game_num + 1}", self.verbose_output
                )
                total_games += 1

                # Track combination wins
                if results["winner"]:
                    winner_name = results["winner"].strategy_name
                    combo_wins[winner_name] += 1

                # Process results
                super()._process_game_results(results, game_players)

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

    def _display_final_results(self):
        """Display comprehensive tournament results."""
        # Create list of all participants including PPO model
        participants = [self.ppo_model] + self.all_strategies
        return super()._display_final_results(
            participants, "PPO vs STRATEGIES TOURNAMENT STANDINGS"
        )

    def _display_detailed_analysis(self):
        """Show detailed strategic analysis."""
        # Create list of all participants including PPO model
        participants = [self.ppo_model] + self.all_strategies
        return super()._display_detailed_analysis(participants)

    def _get_tournament_summary(self):
        """Return structured tournament summary."""
        participants = [self.ppo_model] + self.all_strategies
        summary = super()._get_tournament_summary(
            participants, "PPO vs Strategies Tournament"
        )
        # Add PPO-specific information
        summary.update(
            {
                "ppo_model": self.ppo_model,
                "strategies": self.all_strategies,
                "combinations_tested": len(self.strategy_combinations),
                "games_per_matchup": self.games_per_matchup,
            }
        )
        return summary


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Set random seed
    seed_environ(args.seed)

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
