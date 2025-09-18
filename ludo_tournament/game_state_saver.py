import json
import os
from datetime import datetime


class GameStateSaver:
    def __init__(self, save_dir="saved_states"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.current_game_states = []

    def save_decision(self, strategy_name, game_context, chosen_move, outcome):
        """Save a single decision with its context and outcome"""
        state = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy_name,
            "game_context": self._extract_full_context(game_context),
            "chosen_move": chosen_move,
            "outcome": outcome,
        }
        self.current_game_states.append(state)

    def _extract_full_context(self, context):
        """Extract complete game context for real evaluation"""
        return {
            "current_situation": context.get("current_situation", {}),
            "player_state": context.get("player_state", {}),
            "opponents": context.get("opponents", []),
            "valid_moves": context.get("valid_moves", []),
            "strategic_analysis": context.get("strategic_analysis", {}),
        }

    def save_game(self, game_id):
        """Save all states from current game to file"""
        if not self.current_game_states:
            return

        filename = f"{game_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.save_dir, filename)

        with open(filepath, "w") as f:
            json.dump(self.current_game_states, f, indent=2)

        print(f"Saved {len(self.current_game_states)} decisions to {filename}")
        self.current_game_states = []

    def load_states(self, strategy_name=None):
        """Load saved states, optionally filtered by strategy"""
        all_states = []

        for filename in os.listdir(self.save_dir):
            if filename.endswith(".json"):
                with open(os.path.join(self.save_dir, filename), "r") as f:
                    states = json.load(f)
                    if strategy_name:
                        states = [s for s in states if s["strategy"] == strategy_name]
                    all_states.extend(states)

        return all_states

    def analyze_strategy(self, strategy_name):
        """Simple analysis of saved decisions for a strategy"""
        states = self.load_states(strategy_name)
        if not states:
            return f"No data for {strategy_name}"

        wins = sum(1 for s in states if s["outcome"].get("game_won"))
        captures = sum(len(s["outcome"].get("captured_tokens", [])) for s in states)

        return {
            "total_decisions": len(states),
            "games_won": wins,
            "total_captures": captures,
            "avg_moves_per_decision": sum(s["valid_moves"] for s in states)
            / len(states),
        }
