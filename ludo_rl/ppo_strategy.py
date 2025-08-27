import numpy as np
from stable_baselines3 import PPO

from .envs.ludo_env import EnvConfig, LudoGymEnv


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
