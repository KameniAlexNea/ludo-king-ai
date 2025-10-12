#!/usr/bin/env python3
"""
PPO vs Strategies Two-Player Tournament System
Head-to-head tournament pitting the PPO model against each strategy individually.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import random
import time

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


class TwoPlayerPPOTournament(BaseTournament):
    """Head-to-head tournament system pitting PPO model against each strategy individually."""

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

        # Get strategies universe
        self.all_strategies = self._get_strategies()

        if self.verbose_output:
            print("🎯 PPO vs Strategies Two-Player Tournament Configuration:")
            print(f"   • PPO Model: {self.ppo_model}")
            print(f"   • Model Preference: {self.model_preference}")
            print(f"   • Available strategies: {len(self.all_strategies)}")
            print(f"   • Head-to-head matchups: {len(self.all_strategies)}")
            print(f"   • Games per matchup: {self.games_per_matchup}")
            print(f"   • Max turns per game: {self.max_turns_per_game}")
            print(f"   • Models directory: {self.models_dir}")
            print(
                f"   • Output directory: {self.output_dir if self.output_dir else 'None (no saving)'}"
            )
            print(f"   • Environment mode: {self.env_kind}")
            print(
                f"   • Total games to play: {len(self.all_strategies) * self.games_per_matchup}"
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
        """Execute PPO vs Strategies two-player tournament."""
        print("🏆 PPO vs STRATEGIES HEAD-TO-HEAD TOURNAMENT 🏆")
        print("=" * 70)

        self._display_participants()
        self._run_head_to_head()
        self._display_final_results()
        self._display_detailed_analysis()

        return self._get_tournament_summary()

    def _display_participants(self):
        """Show tournament participants."""
        print("\n🤖 Tournament Participants:")
        print("-" * 50)
        print(f"PPO: {self.ppo_model.upper()}")
        print("\nOpponent Strategies:")

        descriptions = StrategyFactory.get_strategy_descriptions()
        for i, strategy in enumerate(self.all_strategies, 1):
            desc = descriptions.get(strategy, "No description")
            print(f"{i}. {strategy.upper()}: {desc}")

        print("\n📋 Tournament Format:")
        print(f"   • {self.games_per_matchup} games per head-to-head matchup")
        print(f"   • {len(self.all_strategies)} unique opponents")
        print(f"   • Maximum {self.max_turns_per_game} turns per game")
        print("   • Two-player head-to-head with detailed analytics")

    def _run_head_to_head(self):
        """Run head-to-head matches: PPO vs each strategy."""
        print("\n🎮 Head-to-Head Execution:")
        print("=" * 70)

        total_games = 0
        matchup_results = []
        start_time = time.time()

        for matchup_idx, opponent_strategy in enumerate(self.all_strategies, 1):
            print(
                f"\nMatchup {matchup_idx}/{len(self.all_strategies)}: "
                f"{self.ppo_model.upper()} vs {opponent_strategy.upper()}"
            )
            print("-" * 60)

            # Track wins for this matchup
            matchup_wins = {self.ppo_model: 0, opponent_strategy: 0, "draw": 0}

            # Play multiple games for this matchup
            for game_num in range(self.games_per_matchup):
                # Alternate starting positions for fairness
                if game_num % 2 == 0:
                    game_players = [self.ppo_model, opponent_strategy]
                    game_colors = [ALL_COLORS[0], ALL_COLORS[2]]  # Red vs Yellow (opposite positions)
                else:
                    game_players = [opponent_strategy, self.ppo_model]
                    game_colors = [ALL_COLORS[0], ALL_COLORS[2]]  # Same colors, different order

                if self.verbose_output:
                    color_names = [ALL_COLORS[0].value, ALL_COLORS[2].value]
                    player_color_info = [
                        f"{game_players[i].upper()}({color_names[i]})" 
                        for i in range(2)
                    ]
                    print(f"  Game {game_num + 1}: {' vs '.join(player_color_info)}")

                # Create 2-player game
                game = LudoGame(game_colors)

                # Assign strategies
                for i, (player_name, colour) in enumerate(zip(game_players, game_colors)):
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
                    
                    player = game.get_player_from_color(colour)
                    player.set_strategy(strategy)
                    player.strategy_name = player_name

                # Play the game
                results = self._play_two_player_game(
                    game, f"{matchup_idx}.{game_num + 1}", self.verbose_output
                )
                total_games += 1

                # Track matchup wins
                if results["winner"]:
                    winner_name = results["winner"].strategy_name
                    matchup_wins[winner_name] += 1
                else:
                    matchup_wins["draw"] += 1

                # Process results for overall tournament tracking
                super()._process_game_results(results, game_players)

            # Show matchup summary
            ppo_wins = matchup_wins[self.ppo_model]
            opponent_wins = matchup_wins[opponent_strategy]
            draws = matchup_wins["draw"]
            
            print(f"  Results: {self.ppo_model.upper()}: {ppo_wins}, "
                  f"{opponent_strategy.upper()}: {opponent_wins}, Draws: {draws}")
            
            # Calculate win percentage for this matchup
            total_matchup_games = ppo_wins + opponent_wins + draws
            if total_matchup_games > 0:
                ppo_win_pct = (ppo_wins / total_matchup_games) * 100
                print(f"  PPO Win Rate: {ppo_win_pct:.1f}%")
            
            matchup_results.append((opponent_strategy, matchup_wins))

        elapsed = time.time() - start_time
        print(f"\n⏱️  Tournament completed in {elapsed:.1f} seconds")
        print(f"📊 Total games played: {total_games}")
        print(f"🎯 Head-to-head matchups: {len(self.all_strategies)}")

        return matchup_results

    def _play_two_player_game(self, game: LudoGame, game_id, verbose=False):
        """Play a single two-player game and return results."""
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
            dice_roll = game.roll_dice()
            
            # Get AI decision context
            context = game.get_ai_decision_context(dice_roll)
            
            if context.valid_moves:
                # Player makes strategic decision
                chosen_token = current_player.make_strategic_decision(context)
                
                # Execute the move
                move_result = game.execute_move(current_player, chosen_token, dice_roll)
                
                # Save decision if state saver is available
                if self.state_saver:
                    self.state_saver.save_decision(
                        strategy_name, context, chosen_token, move_result
                    )

                game_results["player_stats"][strategy_name]["moves_made"] += 1

                if move_result.captured_tokens:
                    captures = len(move_result.captured_tokens)
                    game_results["player_stats"][strategy_name]["tokens_captured"] += captures
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
                    if verbose:
                        print(f"  Game {game_id}: {strategy_name.upper()} WINS! ({turn_count} turns)")
                    break
                
                # Handle turn progression
                if not move_result.extra_turn and not game.game_over:
                    game.next_turn()
            else:
                # No valid moves, skip turn
                game.next_turn()
            
            turn_count += 1
            game_results["player_stats"][strategy_name]["turns_taken"] += 1

        # Handle draw/timeout case
        if not game_results["winner"]:
            if verbose:
                print(f"  Game {game_id}: DRAW ({turn_count} turns)")
            game_results["turns_played"] = turn_count

        # Save game state if state saver is available
        if self.state_saver:
            self.state_saver.save_game_state(game, game_results)

        return game_results

    def _display_final_results(self):
        """Display comprehensive tournament results."""
        # Create list of all participants including PPO model
        participants = [self.ppo_model] + self.all_strategies
        return super()._display_final_results(
            participants, "PPO vs STRATEGIES HEAD-TO-HEAD STANDINGS"
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
            participants, "PPO vs Strategies Head-to-Head Tournament"
        )
        # Add PPO-specific information
        summary.update(
            {
                "ppo_model": self.ppo_model,
                "strategies": self.all_strategies,
                "matchups_tested": len(self.all_strategies),
                "games_per_matchup": self.games_per_matchup,
                "tournament_type": "head_to_head"
            }
        )
        return summary


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Set random seed
    seed_environ(args.seed)

    print("🎯 LUDO PPO vs STRATEGIES HEAD-TO-HEAD TOURNAMENT 🎯")
    print("=" * 70)
    print("Starting comprehensive PPO vs Strategies head-to-head tournament...")

    # Run main tournament
    tournament = TwoPlayerPPOTournament(args)
    summary = tournament.run_tournament()

    # Final summary
    print("\n🎯 TOURNAMENT COMPLETE! 🎯")
    print("=" * 70)
    if summary["champion"]:
        print(f"🏆 Champion: {summary['champion'].upper()}")
    else:
        print("🏆 No clear champion (no games completed)")
    print(f"📊 Total Games: {summary['total_games']}")
    print(f"🎯 Head-to-Head Matchups: {summary['matchups_tested']}")
    print(f"🤖 PPO Model: {summary['ppo_model'].upper()}")
    print(f"🎮 Strategies: {', '.join([s.upper() for s in summary['strategies']])}")