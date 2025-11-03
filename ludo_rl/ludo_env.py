from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config import config
from src.reward import reward_config
from src.simulator import GameSimulator


class LudoEnv(gym.Env):
    """
    A Gymnasium environment for Ludo King.

    Observation Space:
        A dictionary with:
        - "board": (10, 58) Box, representing 10 stacked channels.
        - "dice_roll": (1,) Box, representing the dice roll (0-5).

    Action Space:
        Discrete(4), representing the choice of which piece to move (0, 1, 2, or 3).
        Action masking MUST be used.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def action_masks(self):
        # Helper for sb3_contrib.common.masking.ActionMasker
        return self._get_info()["action_mask"]

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.agent_index = 0  # We are always Player 0
        self.simulator = GameSimulator(self.agent_index)
        self.render_mode = render_mode

        # --- ADDED FOR TRUNCATION ---
        self.max_game_turns = config.MAX_TURNS
        self.current_turn = 0
        # ---

        # Action Space: Choose one of 4 pieces
        self.action_space = spaces.Discrete(4)

        # Observation Space: As designed
        self.observation_space = spaces.Dict(
            {
                # (channels, path_length)
                # Channels: my, opp1, opp2, opp3, safe, move_heat, my_ko, opp_ko, blockades, reward_heat
                "board": spaces.Box(
                    low=-50.0,
                    high=50.0,
                    shape=(10, config.PATH_LENGTH),
                    dtype=np.float32,
                ),
                # Dice roll (1-6) will be 0-5 for one-hot encoding
                "dice_roll": spaces.Box(low=0, high=5, shape=(1,), dtype=np.int64)
            }
        )

    def _obs_data_to_gym_space(self, obs_data):
        """Converts the simulator's observation dict to the gym observation."""
        board_state = obs_data["board_state"]
        summary = obs_data["transition_summary"]

        board_stack = np.stack(
            [
                board_state["my_pieces"],
                board_state["opp1_pieces"],
                board_state["opp2_pieces"],
                board_state["opp3_pieces"],
                board_state["safe_zones"],
                summary["movement_heatmap"],
                summary["my_knockouts"],
                summary["opp_knockouts"],
                summary["new_blockades"],
                obs_data["reward_heatmap"],
            ],
            dtype=np.float32,
        )

        dice_val = np.array([obs_data["dice_roll"] - 1], dtype=np.int64)

        return {"board": board_stack, "dice_roll": dice_val}

    def _get_info(self):
        """Generates the info dict, including the crucial action mask."""
        valid_moves = self.simulator.game.get_valid_moves(
            self.agent_index, self.current_dice_roll
        )
        # Use bool for the mask as recommended by Gymnasium
        action_mask = np.zeros(4, dtype=np.bool_)
        self.move_map = {}

        for move in valid_moves:
            piece_id = move["piece"].piece_id
            action_mask[piece_id] = True
            # Store the first valid move found for a piece
            if piece_id not in self.move_map:
                move_copy = move.copy()
                move_copy["dice_roll"] = self.current_dice_roll
                self.move_map[piece_id] = move_copy

        return {"action_mask": action_mask}

    def _check_game_over(self):
        """
        Checks if the agent has won (terminated)
        or if the game has hit the turn limit (truncated).
        """
        # 1. Check for termination (win condition)
        player = self.simulator.game.players[self.agent_index]
        terminated = player.has_won()

        # 2. Check for truncation (turn limit)
        truncated = self.current_turn >= self.max_game_turns

        return terminated, truncated

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)

        # Re-initialize the game
        self.simulator = GameSimulator(self.agent_index)
        self.current_dice_roll = self.simulator.game.roll_dice()

        # --- RESET TURN COUNTER ---
        self.current_turn = 0

        obs_data = self.simulator.get_agent_observation(self.current_dice_roll)
        obs = self._obs_data_to_gym_space(obs_data)
        info = self._get_info()

        # Handle "no valid moves" on the very first turn
        while not np.any(info["action_mask"]):
            if self.render_mode == "human":
                print(
                    f"Agent rolled {self.current_dice_roll}, no valid moves. Skipping turn."
                )

            self.current_turn += 1
            self.simulator.step_opponents_only()
            self.current_dice_roll = self.simulator.game.roll_dice()

            obs_data = self.simulator.get_agent_observation(self.current_dice_roll)
            obs = self._obs_data_to_gym_space(obs_data)
            info = self._get_info()

            if self.current_turn >= self.max_game_turns:
                return obs, info

        if self.render_mode == "human":
            print("--- Game Reset ---")
            self.render()
            print(
                f"Agent rolled {self.current_dice_roll}. Valid pieces: {np.where(info['action_mask'])[0]}"
            )

        return obs, info

    def step(self, action: int):
        reward = 0.0
        extra_turn = False

        # 1. Check if the chosen action is valid
        if self.move_map.get(action) is None:
            # Agent chose an invalid piece. This shouldn't happen with MaskablePPO.
            # If it does, penalize and skip turn.
            reward = reward_config.lose  # Heavy penalty
            if self.render_mode == "human":
                print(f"Agent chose INVALID action {action}. Turn skipped.")
            opponent_rewards = self.simulator.step_opponents_only()
            reward += opponent_rewards[self.agent_index]
            self.current_turn += 1
            # Get observation for next turn
            self.current_dice_roll = self.simulator.game.roll_dice()
            next_obs_data = self.simulator.get_agent_observation(self.current_dice_roll)

        else:
            # 2. Execute the valid move
            chosen_move = self.move_map[action]

            if self.render_mode == "human":
                print(
                    f"Agent moving piece {chosen_move['piece'].piece_id} to {chosen_move['new_pos']}"
                )

            # `step` executes agent move, simulates opponents, and returns next obs
            next_obs_data, reward, extra_turn = self.simulator.step(chosen_move)
            if not extra_turn:
                self.current_turn += 1

        self.current_dice_roll = next_obs_data["dice_roll"]
        obs = self._obs_data_to_gym_space(next_obs_data)
        info = self._get_info()

        # 3. Check for game over (win or turn limit)
        terminated, truncated = self._check_game_over()

        if terminated:
            players = self.simulator.game.players
            rank = sum(p.has_won() for p in players)
            reward += (
                reward_config.win * (len(players) - rank + 1) / len(players)
            )  # Large bonus for winning
            if self.render_mode == "human":
                print(f"---AGENT FINISHED (Rank: {rank}) ---")
            return obs, reward, terminated, truncated, info

        if truncated:
            if self.render_mode == "human":
                print(
                    f"--- MAX TURNS ({self.max_game_turns}) REACHED. GAME TRUNCATED. ---"
                )
            # No extra reward/penalty, SB3 handles this via GAE
            return obs, reward, terminated, truncated, info

        # 4. Handle "no valid moves" for the *next* turn
        # Loop until the agent has a move to make
        while not np.any(info["action_mask"]) and not terminated and not truncated:
            reward += reward_config.skipped_turn  # Small penalty for a skipped turn
            if self.render_mode == "human":
                print(
                    f"Agent rolled {self.current_dice_roll}, no valid moves. Skipping turn."
                )
            opponent_rewards = self.simulator.step_opponents_only()
            reward += opponent_rewards[self.agent_index]
            self.current_turn += 1
            if self.current_turn >= self.max_game_turns:
                truncated = True
                break

            self.current_dice_roll = self.simulator.game.roll_dice()
            next_obs_data = self.simulator.get_agent_observation(self.current_dice_roll)
            obs = self._obs_data_to_gym_space(next_obs_data)
            info = self._get_info()
            terminated, truncated = self._check_game_over()  # Re-check every loop

        if truncated:
            if self.render_mode == "human":
                print(
                    f"--- MAX TURNS ({self.max_game_turns}) REACHED. GAME TRUNCATED. ---"
                )
            reward += reward_config.lose
            return obs, reward, terminated, truncated, info
        if self.render_mode == "human":
            self.render()
            print(
                f"Agent's turn. Rolled {self.current_dice_roll}. Valid pieces: {np.where(info['action_mask'])[0]}"
            )
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"--- Turn {self.current_turn}/{self.max_game_turns} ---")
            for i, player in enumerate(self.simulator.game.players):
                player_str = "AGENT (P0)" if i == 0 else f"Opponent (P{i})"
                piece_pos = [p.position for p in player.pieces]
                print(f"   {player_str}: {piece_pos}")

    def close(self):
        pass
