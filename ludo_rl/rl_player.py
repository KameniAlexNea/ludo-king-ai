"""Simplified RL player and strategy factory."""

from typing import Dict

from .config import TRAINING_CONFIG
from .model.dqn_model import LudoDQNAgent
from .states import LudoStateEncoder


class RLPlayer:
    def __init__(self, model_path: str = None, name: str = "RLPlayer"):
        self.name = name
        self.encoder = LudoStateEncoder()
        self.agent = LudoDQNAgent(
            state_dim=self.encoder.state_dim,
            max_actions=4,
            lr=TRAINING_CONFIG.LEARNING_RATE,
            gamma=TRAINING_CONFIG.GAMMA,
            epsilon=0.0,
            use_prioritized_replay=False,
            use_double_dqn=TRAINING_CONFIG.USE_DOUBLE_DQN,
        )
        if model_path:
            self.load_model(model_path)
        self.agent.set_eval_mode()

    def choose_move(self, game_state: Dict) -> int:
        valid_moves = game_state.get("valid_moves", [])
        if not valid_moves:
            return 0
        state = self.encoder.encode_state(game_state)
        action_idx = self.agent.act(state, valid_moves)
        return action_idx if 0 <= action_idx < len(valid_moves) else 0

    def load_model(self, model_path: str):
        try:
            self.agent.load_model(model_path)
            self.agent.set_eval_mode()
            print(f"Loaded RL model from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")


class LudoRLStrategy:
    def __init__(self, model_path: str = None, name: str = "RL-DQN"):
        self.name = name
        self.rl_player = RLPlayer(model_path, name)

    def decide(self, game_context: Dict) -> int:
        game_data = {"game_context": game_context, "chosen_move": 0}
        move_index = self.rl_player.choose_move(game_data)
        valid_moves = game_context.get("valid_moves", [])
        if valid_moves and 0 <= move_index < len(valid_moves):
            return valid_moves[move_index]["token_id"]
        return valid_moves[0]["token_id"] if valid_moves else 0


def create_rl_strategy(model_path: str = None, name: str = "RL-DQN") -> LudoRLStrategy:
    return LudoRLStrategy(model_path, name)
