#!/usr/bin/env python3
"""
PPO vs Strategies Tournament System
Tournament pitting the best PPO model against all available Ludo strategies.
"""

import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import random
import time
from itertools import combinations

import numpy as np
from dotenv import load_dotenv

from ludo import LudoGame, PlayerColor, StrategyFactory

# Dynamic PPO strategy & EnvConfig import handled after CLI args (see FourPlayerPPOTournament._load_ppo_wrapper)
from ludo_stats.game_state_saver import GameStateSaver
from ludo_tournament import BaseTournament
from rl_base.load_ppo_model import load_ppo_strategy

load_dotenv()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PPO vs Strategies Tournament System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        default=int(os.getenv("MAX_TURNS_PER_GAME", 1000)),
        help="Maximum turns per game before declaring draw",
    )

    parser.add_argument(
        "--games-per-matchup",
        type=int,
        default=int(os.getenv("GAMES_PER_MATCHUP", 10)),
        help="Number of games to play per strategy combination",
    )

    parser.add_argument("--seed", type=int, default=42, help="Tournament random seed")

    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directory containing PPO model files",
    )

    parser.add_argument(
        "--strategies",
        type=str,
        nargs="*",
        help="Specific strategies to include (default: all available)",
    )

    parser.add_argument("--quiet", action="store_true", help="Reduce verbose output")

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save game states and results (optional)",
    )

    parser.add_argument(
        "--model-preference",
        type=str,
        choices=["best", "final", "steps"],
        default="best",
        help="Preference for selecting PPO model: 'best' (prefer BEST model), 'final' (prefer FINAL model), 'steps' (prefer highest step count)",
    )


def run_game_with_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


class FourPlayerPPOTournament(BaseTournament):
    """Tournament system pitting PPO model against Ludo strategies."""

    def __init__(self, args):
        # Core configuration
        self.max_turns_per_game = args.max_turns
        self.games_per_matchup = args.games_per_matchup
        self.tournament_seed = args.seed
        self.models_dir = args.models_dir
        self.verbose_output = not args.quiet
        self.output_dir = args.output_dir
        self.selected_strategies = args.strategies
        self.env_kind = args.env  # 'single-seat' or 'classic'
        self.model_preference = args.model_preference  # 'best', 'final', or 'steps'

        # Dynamic PPO wrapper classes (for classic mode) / policy loading
        self._load_ppo_wrapper()

        # Initialize state saver only if output directory is specified
        state_saver = GameStateSaver(self.output_dir) if self.output_dir else None

        # Initialize parent class
        super().__init__(
            max_turns_per_game=self.max_turns_per_game, state_saver=state_saver
        )

        # Resolve model name & path
        self.ppo_model = self._select_best_ppo_model()
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
        """Select the best PPO model based on configured preference."""
        if not os.path.exists(self.models_dir):
            raise FileNotFoundError(f"Models directory {self.models_dir} not found")

        model_files = [f for f in os.listdir(self.models_dir) if f.endswith(".zip")]
        if not model_files:
            raise FileNotFoundError(f"No PPO model files found in {self.models_dir}/")
        
        # Define preference order
        if self.model_preference == "best":
            prefs = ["best", "final", "steps"]
        elif self.model_preference == "final":
            prefs = ["final", "best", "steps"]
        elif self.model_preference == "steps":
            prefs = ["steps", "best", "final"]
        
        # Try each preference in order
        for pref in prefs:
            if pref == "best":
                best_model = next((f for f in model_files if "best" in f.lower()), None)
                if best_model:
                    return best_model.replace(".zip", "")
            elif pref == "final":
                final_model = next((f for f in model_files if "final" in f.lower()), None)
                if final_model:
                    return final_model.replace(".zip", "")
            elif pref == "steps":
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
                colours = [
                    PlayerColor.RED,
                    PlayerColor.BLUE,
                    PlayerColor.GREEN,
                    PlayerColor.YELLOW,
                ]
                game = LudoGame(colours)

                # Assign strategies. Always wrap PPO model in its strategy class for consistency across env modes.
                for i, (player_name, colour) in enumerate(zip(game_players, colours)):
                    if player_name == self.ppo_model:
                        model_path = f"{self.models_dir}/{player_name}.zip"
                        try:
                            strategy = self.PPOStrategyClass(
                                model_path,
                                player_name,
                                self.EnvConfigClass(agent_color=colour.value),
                            )
                        except Exception as e:
                            raise RuntimeError(
                                f"Failed to initialize PPO strategy for model '{player_name}': {e}"
                            ) from e
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
