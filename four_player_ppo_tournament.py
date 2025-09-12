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
from ludo_stats.game_state_saver import GameStateSaver
from ludo_tournament import BaseTournament
from rl_base.load_ppo_model import load_ppo_strategy, select_best_ppo_model
from loguru import logger
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
        default="final",
        help="Preference for selecting PPO model: 'best' (prefer BEST model), 'final' (prefer FINAL model), 'steps' (prefer highest step count)",
    )

    parser.add_argument(
        "--env",
        type=str,
        choices=["classic", "single-seat"],
        default="classic",
        help="Environment kind: 'classic' or 'single-seat'",
    )
    return parser.parse_args()


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

        # Initialize state saver only if output directory is specified
        state_saver = GameStateSaver(self.output_dir) if self.output_dir else None

        # Initialize parent class
        super().__init__(
            max_turns_per_game=self.max_turns_per_game, state_saver=state_saver
        )

        # Resolve model name & path
        self.ppo_model = select_best_ppo_model(self.models_dir, self.model_preference)
        self.ppo_model_path = os.path.join(self.models_dir, f"{self.ppo_model}.zip")

        # PPO policy will be accessed via PPOStrategyClass wrapper uniformly.

        # Get strategies universe
        self.all_strategies = self._get_strategies()

        # Compute opponent combinations (3 strategies vs PPO)
        self.strategy_combinations = list(combinations(self.all_strategies, 3))

        if self.verbose_output:
            logger.info("🎯 PPO vs Strategies Tournament Configuration:")
            logger.info(f"   • PPO Model: {self.ppo_model}")
            logger.info(f"   • Model Preference: {self.model_preference}")
            logger.info(f"   • Available strategies: {len(self.all_strategies)}")
            logger.info(f"   • Strategy combinations: {len(self.strategy_combinations)}")
            logger.info(f"   • Games per matchup: {self.games_per_matchup}")
            logger.info(f"   • Max turns per game: {self.max_turns_per_game}")
            logger.info(f"   • Models directory: {self.models_dir}")
            logger.info(
                f"   • Output directory: {self.output_dir if self.output_dir else 'None (no saving)'}"
            )
            logger.info(f"   • Environment mode: {self.env_kind}")
            logger.info(
                f"   • Total games to play: {len(self.strategy_combinations) * self.games_per_matchup}"
            )

    def _get_strategies(self):
        """Get strategies to use in tournament."""
        if self.selected_strategies:
            return self.selected_strategies
        if os.getenv("SELECTED_STRATEGIES"):
            return list(os.getenv("SELECTED_STRATEGIES").split(","))
        return StrategyFactory.get_available_strategies()

    def run_tournament(self):
        """Execute PPO vs Strategies tournament."""
        logger.info("🏆 PPO vs STRATEGIES TOURNAMENT 🏆")
        logger.info("=" * 70)

        self._display_participants()
        self._run_round_robin()
        self._display_final_results()
        self._display_detailed_analysis()

        return self._get_tournament_summary()

    def _display_participants(self):
        """Show tournament participants."""
        logger.info("\n🤖 Tournament Participants:")
        logger.info("-" * 50)
        logger.info(f"PPO: {self.ppo_model.upper()}")
        logger.info("\nStrategies:")

        descriptions = StrategyFactory.get_strategy_descriptions()
        for i, strategy in enumerate(self.all_strategies, 1):
            desc = descriptions.get(strategy, "No description")
            logger.info(f"{i}. {strategy.upper()}: {desc}")

        logger.info("\n📋 Tournament Format:")
        logger.info(f"   • {self.games_per_matchup} games per 4-player combination")
        logger.info(f"   • {len(self.strategy_combinations)} unique combinations")
        logger.info(f"   • Maximum {self.max_turns_per_game} turns per game")
        logger.info("   • All combinations tournament with detailed analytics")

    def _run_round_robin(self):
        """Run tournament with PPO vs strategy combinations."""
        logger.info("\n🎮 Tournament Execution:")
        logger.info("=" * 70)

        total_games = 0
        combination_results = []
        start_time = time.time()

        for combo_idx, strategy_combo in enumerate(self.strategy_combinations, 1):
            # Create combination: PPO + 3 strategies
            game_players = [self.ppo_model] + list(strategy_combo)

            logger.info(
                f"\nCombination {combo_idx}/{len(self.strategy_combinations)}: "
                f"{' vs '.join([p.upper() for p in game_players])}"
            )
            logger.info("-" * 60)

            combo_wins = {player: 0 for player in game_players}

            # Play multiple games for this combination
            for game_num in range(self.games_per_matchup):
                # Randomize starting order for fairness
                random.shuffle(game_players)

                if self.verbose_output:
                    logger.info(
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
                        strategy = load_ppo_strategy(
                            self.env_kind,
                            self.models_dir,
                            player_name,
                            colour,
                            self.model_preference,
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
            logger.info(f"  Results: {combo_summary}")
            combination_results.append((game_players, combo_wins))

        elapsed = time.time() - start_time
        logger.info(f"\n⏱️  Tournament completed in {elapsed:.1f} seconds")
        logger.info(f"📊 Total games played: {total_games}")
        logger.info(f"🎯 Combinations tested: {len(self.strategy_combinations)}")

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

    logger.info("🎯 LUDO PPO vs STRATEGIES TOURNAMENT 🎯")
    logger.info("=" * 70)
    logger.info("Starting comprehensive PPO vs Strategies tournament...")

    # Run main tournament
    tournament = FourPlayerPPOTournament(args)
    summary = tournament.run_tournament()

    # Final summary
    logger.info("\n🎯 TOURNAMENT COMPLETE! 🎯")
    logger.info("=" * 70)
    if summary["champion"]:
        logger.info(f"🏆 Champion: {summary['champion'].upper()}")
    else:
        logger.info("🏆 No clear champion (no games completed)")
    logger.info(f"📊 Total Games: {summary['total_games']}")
    logger.info(f"🎯 Combinations Tested: {summary['combinations_tested']}")
    logger.info(f"🤖 PPO Model: {summary['ppo_model'].upper()}")
    logger.info(f"🎮 Strategies: {', '.join([s.upper() for s in summary['strategies']])}")
    logger.info("\n✅ PPO vs Strategies Tournament System Ready!")
    logger.info("🔬 Advanced PPO vs Strategies evaluation complete!")
