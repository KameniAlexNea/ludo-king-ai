#!/usr/bin/env python3
"""
PPO vs Strategies Tournament System
Tournament pitting the best PPO model against all available Ludo strategies.
"""

import os
import random
import time
from collections import defaultdict
from itertools import combinations

import numpy as np
from stable_baselines3 import PPO

from ludo import LudoGame, PlayerColor, StrategyFactory
from ludo_stats.game_state_saver import GameStateSaver
from ludo_rl.envs.ludo_env import LudoGymEnv, EnvConfig


class PPOStrategy:
    """Wrapper to use PPO model as a Ludo strategy."""

    def __init__(self, model_path: str, model_name: str):
        self.model_name = model_name
        self.model = PPO.load(model_path)
        # Create a dummy env for observation building
        self.dummy_env = LudoGymEnv(EnvConfig())
        self.description = f"PPO Model: {model_name}"

    def decide(self, game_context: dict) -> int:
        """Convert game context to Gym observation and predict action."""
        # Build observation from context (similar to LudoGymEnv._build_observation)
        obs = self._build_observation_from_context(game_context)
        
        # Get action mask
        valid_moves = game_context.get("valid_moves", [])
        action_mask = np.zeros(4, dtype=np.int8)
        for move in valid_moves:
            action_mask[move["token_id"]] = 1
        
        # For regular PPO (not MaskablePPO), we need to handle masking manually
        # Keep predicting until we get a valid action
        max_attempts = 10
        for attempt in range(max_attempts):
            # Predict action (without action_masks parameter for regular PPO)
            action, _ = self.model.predict(obs, deterministic=True)
            action = int(action)
            
            # Check if action is valid
            if action_mask[action] == 1:
                return action
        
        # If we can't find a valid action after max attempts, 
        # return the first valid action as fallback
        for i in range(4):
            if action_mask[i] == 1:
                return i
        
        # Ultimate fallback: return action 0
        return 0

    def _build_observation_from_context(self, context: dict) -> np.ndarray:
        """Build Gym-style observation from game context."""
        # If context has full game state, use that
        if "players" in context:
            game_info = context.get("game_info", {})
            players = context.get("players", [])
            current_player_color = game_info.get("current_player", "")
            
            vec = []
            
            # Find current player
            current_player = None
            for player in players:
                if player.get("color") == current_player_color:
                    current_player = player
                    break
            
            if current_player is None:
                # Fallback
                return np.zeros(26, dtype=np.float32)
            
            # Agent token positions (current player's tokens)
            for token in current_player.get("tokens", []):
                pos = token.get("position", -1)
                vec.append(self.dummy_env._normalize_position(pos))
            
            # Opponents token positions (fixed order: other 3 players)
            for player in players:
                if player.get("color") != current_player_color:
                    for token in player.get("tokens", []):
                        pos = token.get("position", -1)
                        vec.append(self.dummy_env._normalize_position(pos))
            
            # Finished tokens per player (all 4 players in order)
            for player in players:
                finished = player.get("finished_tokens", 0)
                vec.append(finished / 4.0)
            
            # Can any finish
            can_finish = 0.0
            for token in current_player.get("tokens", []):
                pos = token.get("position", -1)
                if 0 <= pos < 100:
                    remaining = 105 - pos
                    if remaining <= 6:
                        can_finish = 1.0
                        break
            vec.append(can_finish)
            
            # Dice value (normalized) - need to get this from somewhere
            dice_value = context.get("dice_value", 0)  # This might need adjustment
            vec.append((dice_value - 3.5) / 3.5)
            
            # Progress stats
            agent_finished = current_player.get("finished_tokens", 0) / 4.0
            opp_progresses = []
            for player in players:
                if player.get("color") != current_player_color:
                    opp_progresses.append(player.get("finished_tokens", 0) / 4.0)
            opp_mean = np.mean(opp_progresses) if opp_progresses else 0.0
            vec.append(agent_finished)
            vec.append(opp_mean)
            
            # Turn index (scaled)
            turn_count = game_info.get("turn_count", 0)
            vec.append(min(1.0, turn_count / 1000.0))
            
            # Blocking count (simplified - placeholder for now)
            vec.append(0.0)
            
            return np.asarray(vec, dtype=np.float32)
        
        # Fallback to old method if context structure is different
        else:
            current_situation = context.get("current_situation", {})
            player_state = context.get("player_state", {})
            opponents = context.get("opponents", [])
            
            vec = []
            
            # Agent token positions (current player's tokens)
            for token in player_state.get("tokens", []):
                pos = token.get("position", -1)
                vec.append(self.dummy_env._normalize_position(pos))
            
            # Opponents token positions (fixed order: other 3 players)
            # Placeholder for opponents
            for opp in opponents:
                for i in range(4):  # 4 tokens per opponent
                    vec.append(self.dummy_env._normalize_position(-1))  # Placeholder
            
            # Finished tokens per player (current player + opponents)
            vec.append(player_state.get("finished_tokens", 0) / 4.0)
            for opp in opponents:
                vec.append(opp.get("tokens_finished", 0) / 4.0)
            
            # Can any finish
            can_finish = 0.0
            for token in player_state.get("tokens", []):
                pos = token.get("position", -1)
                if 0 <= pos < 100:
                    remaining = 105 - pos
                    if remaining <= 6:
                        can_finish = 1.0
                        break
            vec.append(can_finish)
            
            # Dice value (normalized)
            dice_value = current_situation.get("dice_value", 0)
            vec.append((dice_value - 3.5) / 3.5)
            
            # Progress stats
            agent_finished = player_state.get("finished_tokens", 0) / 4.0
            opp_progresses = [opp.get("tokens_finished", 0) / 4.0 for opp in opponents]
            opp_mean = np.mean(opp_progresses) if opp_progresses else 0.0
            vec.append(agent_finished)
            vec.append(opp_mean)
            
            # Turn index (scaled)
            turn_count = current_situation.get("turn_count", 0)
            vec.append(min(1.0, turn_count / 1000.0))
            
            # Blocking count (simplified - placeholder for now)
            vec.append(0.0)
            
            return np.asarray(vec, dtype=np.float32)


def run_game_with_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


class FourPlayerPPOTournament:
    """Tournament system pitting PPO model against Ludo strategies."""

    def __init__(self):
        # Configuration
        self.max_turns_per_game = 500
        self.games_per_matchup = 10
        self.tournament_seed = 42
        self.verbose_output = True

        # Initialize state saver
        self.state_saver = GameStateSaver("saved_states/ppo_vs_strategies")

        # Select the best PPO model (use FINAL or the one with most steps)
        self.ppo_model = self._select_best_ppo_model()
        
        # Get all available strategies
        self.all_strategies = StrategyFactory.get_available_strategies()
        
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
            print(
                f"   • Total games to play: {len(self.strategy_combinations) * self.games_per_matchup}"
            )

    def _select_best_ppo_model(self):
        """Select the best PPO model (prefer FINAL, then highest step count)."""
        models_dir = "./models"
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"Models directory {models_dir} not found")
        
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
        if not model_files:
            raise FileNotFoundError("No PPO model files found in ./models/")
        
        # Prefer FINAL model, then highest step count
        final_model = next((f for f in model_files if 'final' in f.lower()), None)
        if final_model:
            return final_model.replace('.zip', '')
        
        # Extract step numbers and find highest
        step_models = []
        for f in model_files:
            try:
                # Extract number from filename like "ppo_ludo_1000000_steps"
                parts = f.replace('.zip', '').split('_')
                for part in parts:
                    if part.isdigit():
                        step_models.append((int(part), f.replace('.zip', '')))
                        break
            except:
                continue
        
        if step_models:
            step_models.sort(reverse=True)
            return step_models[0][1]
        
        # Fallback to first model
        return model_files[0].replace('.zip', '')

    def _get_available_models(self):
        """Get list of available PPO model names from ./models/."""
        models_dir = "./models"
        if not os.path.exists(models_dir):
            print(f"Warning: {models_dir} not found. Creating empty list.")
            return []
        
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
        return [f.replace('.zip', '') for f in model_files]

    def _get_available_models(self):
        """Get list of available PPO model names from ./models/."""
        models_dir = "./models"
        if not os.path.exists(models_dir):
            print(f"Warning: {models_dir} not found. Creating empty list.")
            return []
        
        models = []
        for file in os.listdir(models_dir):
            if file.endswith(".zip") and "ppo" in file.lower():
                model_name = file.replace(".zip", "")
                models.append(model_name)
        return sorted(models)

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

                # Assign strategies to players
                for i, player_name in enumerate(game_players):
                    if player_name == self.ppo_model:
                        # This is the PPO player
                        model_path = f"./models/{player_name}.zip"
                        strategy = PPOStrategy(model_path, player_name)
                    else:
                        # This is a regular strategy player
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
            stats["total_turns"] += results["player_stats"][model_name][
                "turns_taken"
            ]
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

        # Calculate standings for all models that played
        standings = []
        for model in self.all_models:
            if model in self.detailed_stats:
                stats = self.detailed_stats[model]
                wins = stats["games_won"]
                games = stats["games_played"]
                win_rate = (wins / games * 100) if games > 0 else 0

                standings.append(
                    {
                        "model": model,
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
        print(
            f"\n{'Model':<20} {'Captures':<10} {'Finished':<10} {'Efficiency':<12}"
        )
        print("-" * 60)

        for model in self.all_models:
            if model in self.detailed_stats:
                stats = self.detailed_stats[model]
                efficiency = (
                    (stats["tokens_finished"] / stats["games_played"])
                    if stats["games_played"] > 0
                    else 0
                )

                print(
                    f"{model.upper():<20} {stats['tokens_captured']:<10} {stats['tokens_finished']:<10} {efficiency:<11.2f}"
                )

        # Head-to-head analysis (only show models with significant interactions)
        print("\n🥊 HEAD-TO-HEAD ANALYSIS 🥊")
        print("-" * 50)

        for model in self.all_models:
            if model in self.detailed_stats:
                h2h = self.detailed_stats[model]["head_to_head"]
                has_interactions = any(record["games"] > 0 for record in h2h.values())

                if has_interactions:
                    print(f"\n{model.upper()} vs Others:")
                    for opponent, record in h2h.items():
                        if record["games"] > 0:
                            win_rate = (record["wins"] / record["games"]) * 100
                            print(
                                f"  vs {opponent.upper():<18}: {record['wins']}/{record['games']} ({win_rate:.1f}%)"
                            )

    def _get_tournament_summary(self):
        """Return structured tournament summary."""
        # Find champion among models that actually played
        played_models = [
            m
            for m in self.all_models
            if m in self.detailed_stats and self.detailed_stats[m]["games_played"] > 0
        ]

        champion = (
            max(
                played_models,
                key=lambda m: (
                    self.detailed_stats[m]["games_won"],
                    self.detailed_stats[m]["games_won"]
                    / max(1, self.detailed_stats[m]["games_played"]),
                ),
            )
            if played_models
            else None
        )

        summary = {
            "tournament_type": "4-Player PPO Model Combinations",
            "participants": self.all_models,
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
    # Set random seed
    run_game_with_seed(42)

    print("🎯 LUDO 4-PLAYER PPO MODEL TOURNAMENT 🎯")
    print("=" * 70)
    print("Starting comprehensive all-combinations PPO tournament...")

    # Run main tournament
    tournament = FourPlayerPPOTournament()
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
    print(f"🎮 Participants: {', '.join([m.upper() for m in summary['participants']])}")
    print("\n✅ 4-Player PPO Model Tournament System Ready!")
    print("🔬 Advanced PPO model evaluation and comparison complete!")
# </content>
# <parameter name="filePath">/data/home/eak/learning/games/ludo-king-ai/four_player_ppo_tournament.py
