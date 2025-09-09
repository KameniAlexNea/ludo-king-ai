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

import numpy as np
from dotenv import load_dotenv
from loguru import logger

from ludo import LudoGame, PlayerColor, StrategyFactory
from ludo_stats.game_state_saver import GameStateSaver
from ludo_tournament import BaseTournament

# Load environment configuration
load_dotenv()


def run_game_with_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


class FourPlayerTournament(BaseTournament):
    """Advanced 4-player tournament system for strategic AI evaluation."""

    def __init__(self):
        # Load configuration from .env
        self.max_turns_per_game = int(os.getenv("MAX_TURNS_PER_GAME", 500))
        self.games_per_matchup = int(os.getenv("GAMES_PER_MATCHUP", 10))
        self.tournament_seed = int(os.getenv("TOURNAMENT_SEED", 42))
        self.verbose_output = os.getenv("VERBOSE_OUTPUT", "true").lower() == "true"

        # Initialize state saver if SAVE_DIR is set
        save_dir = os.getenv("SAVE_DIR")
        state_saver = GameStateSaver(save_dir) if save_dir else None

        # Initialize parent class
        super().__init__(max_turns_per_game=self.max_turns_per_game, state_saver=state_saver)

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

        if self.verbose_output:
            logger.info("🎯 Tournament Configuration:")
            logger.info(f"   • Available strategies: {len(self.all_strategies)}")
            logger.info(
                f"   • 4-player combinations: {len(self.strategy_combinations)}"
            )
            logger.info(f"   • Games per matchup: {self.games_per_matchup}")
            logger.info(f"   • Max turns per game: {self.max_turns_per_game}")
            logger.info(
                f"   • Total games to play: {len(self.strategy_combinations) * self.games_per_matchup}"
            )

    def run_tournament(self):
        """Execute complete 4-player tournament."""
        logger.info("🏆 4-PLAYER STRATEGIC LUDO TOURNAMENT 🏆")
        logger.info("=" * 70)

        self._display_participants()
        self._run_round_robin()
        self._display_final_results()
        self._display_detailed_analysis()

        return self._get_tournament_summary()

    def _display_participants(self):
        """Show tournament participants and their strategies."""
        logger.info("\n🤖 Tournament Participants:")
        logger.info("-" * 50)

        descriptions = StrategyFactory.get_strategy_descriptions()
        for i, strategy in enumerate(self.all_strategies, 1):
            logger.info(f"{i}. {strategy.upper()}: {descriptions[strategy]}")

        logger.info("\n📋 Tournament Format:")
        logger.info(f"   • {self.games_per_matchup} games per 4-player combination")
        logger.info(f"   • {len(self.strategy_combinations)} unique combinations")
        logger.info(f"   • Maximum {self.max_turns_per_game} turns per game")
        logger.info("   • All combinations tournament with detailed analytics")

    def _run_round_robin(self):
        """Run tournament with all 4-player combinations."""
        logger.info("\n🎮 Tournament Execution:")
        logger.info("=" * 70)

        total_games = 0
        combination_results = []
        start_time = time.time()

        for combo_idx, strategy_combo in enumerate(self.strategy_combinations, 1):
            logger.info(
                f"\nCombination {combo_idx}/{len(self.strategy_combinations)}: "
                f"{' vs '.join([s.upper() for s in strategy_combo])}"
            )
            logger.info("-" * 60)

            combo_wins = {strategy: 0 for strategy in strategy_combo}

            # Play multiple games for this combination
            for game_num in range(self.games_per_matchup):
                # Randomize starting order for fairness
                game_strategies = list(strategy_combo)
                random.shuffle(game_strategies)

                if self.verbose_output:
                    logger.info(
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
                results = super()._play_four_player_game(
                    game, f"{combo_idx}.{game_num + 1}", self.verbose_output
                )
                total_games += 1

                # Track combination wins
                if results["winner"]:
                    winner_name = results["winner"].strategy_name
                    combo_wins[winner_name] += 1

                # Process results
                super()._process_game_results(results, game_strategies)

            # Show combination summary
            combo_summary = ", ".join(
                [f"{s.upper()}: {wins}" for s, wins in combo_wins.items()]
            )
            logger.info(f"  Results: {combo_summary}")
            combination_results.append((strategy_combo, combo_wins))

        elapsed = time.time() - start_time
        logger.info(f"\n⏱️  Tournament completed in {elapsed:.1f} seconds")
        logger.info(f"📊 Total games played: {total_games}")
        logger.info(f"🎯 Combinations tested: {len(self.strategy_combinations)}")

        return combination_results

    def _display_final_results(self):
        """Display comprehensive tournament results."""
        return super()._display_final_results(self.all_strategies, "4-PLAYER STRATEGIC TOURNAMENT STANDINGS")

    def _display_detailed_analysis(self):
        """Show detailed strategic analysis."""
        return super()._display_detailed_analysis(self.all_strategies)

    def _get_tournament_summary(self):
        """Return structured tournament summary."""
        summary = super()._get_tournament_summary(self.all_strategies, "4-Player Strategic Combinations")
        # Add tournament-specific information
        summary.update({
            "combinations_tested": len(self.strategy_combinations),
            "games_per_matchup": self.games_per_matchup,
        })
        return summary


if __name__ == "__main__":
    # Set random seed from environment
    run_game_with_seed(int(os.getenv("TOURNAMENT_SEED", 42)))

    logger.info("🎯 LUDO 4-PLAYER COMBINATION TOURNAMENT 🎯")
    logger.info("=" * 70)
    logger.info("Starting comprehensive all-combinations tournament...")

    # Run main tournament
    tournament = FourPlayerTournament()
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
    logger.info(
        f"🎮 Participants: {', '.join([s.upper() for s in summary['participants']])}"
    )
    logger.info("\n✅ 4-Player Strategic Tournament System Ready!")
    logger.info("🔬 Advanced AI evaluation and comparison complete!")
