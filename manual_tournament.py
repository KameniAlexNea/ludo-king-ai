#!/usr/bin/env python3
"""
Interactive Ludo Tournament Setup
User defines number of players, strategies, LLM parameters, and other configs.
"""
import os
import random

from dotenv import load_dotenv

from ludo import LudoGame, PlayerColor, StrategyFactory
from ludo_stats.game_state_saver import GameStateSaver

load_dotenv()


def prompt_int(prompt, min_val=None, max_val=None, default=None):
    while True:
        val = input(f"{prompt}{' ['+str(default)+']' if default is not None else ''}: ")
        if not val and default is not None:
            return default
        try:
            iv = int(val)
            if (min_val is not None and iv < min_val) or (
                max_val is not None and iv > max_val
            ):
                print(f"Please enter a number between {min_val} and {max_val}.")
                continue
            return iv
        except ValueError:
            print("Invalid number, please try again.")


def prompt_choice(prompt, choices, default=None):
    choices_str = ", ".join(choices)
    while True:
        val = input(f"{prompt} ({choices_str}){' ['+default+']' if default else ''}: ")
        if not val and default:
            return default
        if val in choices:
            return val
        print(f"Please choose one of: {choices_str}")


def main():
    print("🔧 Ludo Tournament Configuration 🔧")
    num_players = prompt_int("Number of players", 2, 4, default=4)

    # Select strategies
    available = StrategyFactory.get_available_strategies()
    print(f"Available strategies: {', '.join(available)}")
    player_strategies = []
    for i in range(1, num_players + 1):
        strat = prompt_choice(f"Strategy for Player {i}", available)
        if strat.lower().startswith("llm"):
            provider = input("  LLM provider (ollama/groq) [ollama]: ") or "ollama"
            model = input("  LLM model name [gpt-oss]: ") or "gpt-oss"
            # Instantiate appropriate LLM strategy class
            from ludo.strategies.llm.strategy import LLMStrategy

            strategy_instance = LLMStrategy(provider=provider, model=model)
        else:
            strategy_instance = StrategyFactory.create_strategy(strat)
        player_strategies.append((str(strategy_instance), strategy_instance))
    # Other configs
    games_per_matchup = prompt_int("Games per matchup", 1, None, default=10)
    max_turns = prompt_int("Max turns per game", 1, None, default=500)
    seed = prompt_int("Tournament random seed", None, None, default=42)
    save_folder = input("Save folder for states [tournaments]: ") or "tournaments"

    # Setup environment
    random.seed(seed)
    os.makedirs(save_folder, exist_ok=True)
    saver = GameStateSaver(save_folder)

    print("\n🎮 Starting Tournament 🎮")
    # Play round robin among selected strategies
    results = {}
    for a in range(len(player_strategies)):
        for b in range(a + 1, len(player_strategies)):
            print(
                f"Starting matchup: {player_strategies[a][0]} vs {player_strategies[b][0]}"
            )
            key = (player_strategies[a][0], player_strategies[b][0])
            wins = {player_strategies[a][0]: 0, player_strategies[b][0]: 0}
            for g in range(games_per_matchup):
                # Initialize game with colors list limited to num_players
                print("Match Up:", g + 1)
                if num_players == 2:
                    colors = [PlayerColor.RED, PlayerColor.GREEN]
                else:
                    colors = [
                        PlayerColor.RED,
                        PlayerColor.BLUE,
                        PlayerColor.GREEN,
                        PlayerColor.YELLOW,
                    ][:num_players]
                game = LudoGame(colors)
                # Assign strategies
                for idx, (_, strat_inst) in enumerate(player_strategies):
                    game.players[idx].set_strategy(strat_inst)
                # Play until winner or max_turns
                turns = 0
                while not game.game_over and turns < max_turns:
                    player = game.get_current_player()
                    dice = game.roll_dice()
                    ctx = game.get_ai_decision_context(dice)
                    if ctx["valid_moves"]:
                        tok = player.make_strategic_decision(ctx)
                        move_res = game.execute_move(player, tok, dice)
                        saver.save_decision(str(player.strategy), ctx, tok, move_res)
                        if move_res.get("game_won"):
                            wins[str(player.strategy)] += 1
                            break
                    game.next_turn()
                    turns += 1
                name = str(key).replace("(", "_").replace(")", "_").replace("'", "")
                saver.save_game(f"{name}_{g}")
            results[key] = wins

    print("\n🏁 Tournament Results 🏁")
    for matchup, win_counts in results.items():
        print(f"{matchup[0]} vs {matchup[1]}: {win_counts}")


if __name__ == "__main__":
    main()
