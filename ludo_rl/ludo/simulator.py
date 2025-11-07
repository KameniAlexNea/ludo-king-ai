import os
import random
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

from .config import config
from .game import LudoGame


@dataclass(slots=True)
class GameSimulator:
    """
    Manages the simulation, modified to integrate with the Gym env.
    """

    agent_index: int = 0
    game: LudoGame = field(init=False)
    transition_summary: Dict[str, List[int]] = field(init=False, repr=False)
    reward_heatmap: List[float] = field(init=False, repr=False)
    _board_buffer: np.ndarray | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self.game = LudoGame()
        self._configure_opponent_strategies()
        self.reset_summaries()

    def reset_summaries(self):
        self.transition_summary = {
            "movement_heatmap": [0] * config.PATH_LENGTH,
            "my_knockouts": [0] * config.PATH_LENGTH,
            "opp_knockouts": [0] * config.PATH_LENGTH,
            "new_blockades": [0] * config.PATH_LENGTH,
        }
        self.reward_heatmap = [0] * config.PATH_LENGTH

    def get_agent_observation(self, dice_roll):
        return {
            "dice_roll": dice_roll,
            "board_state": self.game.get_board_state(self.agent_index),
            "transition_summary": self.transition_summary,
            "reward_heatmap": self.reward_heatmap,
        }

    def update_summaries(self, mover_index, move, result):
        agent_relative_pos = self.get_agent_relative_pos_for_opp(
            mover_index, move["new_pos"]
        )
        if agent_relative_pos != -1:
            self.transition_summary["movement_heatmap"][agent_relative_pos] += 1
            self.reward_heatmap[agent_relative_pos] += result["reward"]

        for knockout in result["events"]["knockouts"]:
            knocked_agent_rel_pos = self.game.get_agent_relative_pos(
                self.agent_index, knockout["abs_pos"]
            )
            if knocked_agent_rel_pos == -1:
                continue
            if knockout["player"] == self.agent_index:
                self.transition_summary["my_knockouts"][knocked_agent_rel_pos] = 1
            else:
                self.transition_summary["opp_knockouts"][knocked_agent_rel_pos] = 1

        for blockade in result["events"]["blockades"]:
            blockade_agent_rel_pos = self.get_agent_relative_pos_for_opp(
                mover_index, blockade["relative_pos"]
            )
            if blockade_agent_rel_pos != -1:
                self.transition_summary["new_blockades"][blockade_agent_rel_pos] = 1

    def get_agent_relative_pos_for_opp(self, opp_index, opp_relative_pos):
        if opp_relative_pos == 0:
            return 0
        if opp_relative_pos > 51:
            return -1
        abs_pos = self.game.get_absolute_position(opp_index, opp_relative_pos)
        return self.game.get_agent_relative_pos(self.agent_index, abs_pos)

    def simulate_opponent_turns(self) -> List[float]:
        """Simulates turns for all 3 opponents and returns cumulative rewards."""
        with torch.autograd.profiler.record_function("simulate_opponent_turns"):
            cumulative_rewards = [0.0] * config.NUM_PLAYERS

            for i in range(1, config.NUM_PLAYERS):
                opp_index = (self.agent_index + i) % config.NUM_PLAYERS
                extra_turn = True

                while extra_turn:
                    dice_roll = self.game.roll_dice()
                    valid_moves = self.game.get_valid_moves(opp_index, dice_roll)

                    if not valid_moves:
                        extra_turn = False
                        continue

                    move = self._decide_move(opp_index, dice_roll, valid_moves)
                    result = self.game.make_move(
                        opp_index, move["piece"], move["new_pos"], dice_roll
                    )
                    self.update_summaries(opp_index, move, result)

                    for player_idx, value in result["rewards"].items():
                        cumulative_rewards[player_idx] += value

                    extra_turn = result["extra_turn"]

            return cumulative_rewards

    def _configure_opponent_strategies(self):
        opponents = os.getenv("OPPONENTS", "random").split(",")
        # STRATEGY SELECTION METHOD : 0 - RANDOM, 1 - SEQUENTIAL
        selection_method = int(os.getenv("STRATEGY_SELECTION", "0"))
        strategies = [opp.strip().lower() for opp in opponents if opp]
        pos = 0
        for idx, player in enumerate(self.game.players):
            if idx == self.agent_index:
                continue
            player.strategy_name = (
                strategies[pos % len(strategies)]
                if selection_method == 1
                else random.choice(strategies)
            )
            player._strategy = None
            pos += 1

    def _decide_move(self, player_index: int, dice_roll: int, valid_moves: list[dict]):
        player = self.game.players[player_index]
        board_stack = self._build_board_stack(player_index)
        move = player.decide(board_stack, dice_roll, valid_moves)
        if move is not None:
            return move
        return random.choice(valid_moves)

    def _build_board_stack(self, player_index: int) -> np.ndarray:
        if self._board_buffer is None:
            self._board_buffer = np.zeros((10, config.PATH_LENGTH), dtype=np.float32)
        return self.game.build_board_tensor(player_index, out=self._board_buffer)

    def step_opponents_only(self):
        """Called when agent has no moves. Resets summaries and simulates opponents."""
        self.reset_summaries()
        return self.simulate_opponent_turns()

    def step(self, agent_move):
        """
        Takes the agent's move, simulates opponents, and returns the next obs,
        the reward accumulated for the agent (including opponent reactions),
        and whether the agent earned an extra turn.
        """
        # Start a fresh transition summary for this full turn sequence.
        self.reset_summaries()

        dice_roll = agent_move.get("dice_roll")
        if dice_roll is None:
            dice_roll = self.game.roll_dice()

        # 1. Execute agent's move
        with torch.autograd.profiler.record_function("agent_make_move"):
            agent_result = self.game.make_move(
                self.agent_index, agent_move["piece"], agent_move["new_pos"], dice_roll
            )

        # Record the agent's own transition before opponents respond.
        self.update_summaries(self.agent_index, agent_move, agent_result)

        total_rewards = [0.0] * config.NUM_PLAYERS
        for player_idx, value in agent_result["rewards"].items():
            total_rewards[player_idx] += value

        extra_turn = agent_result["extra_turn"]

        # 3. Simulate opponent turns if no extra turn was earned
        if not extra_turn:
            with torch.autograd.profiler.record_function("opponent_turns_block"):
                opponent_rewards = self.simulate_opponent_turns()
                total_rewards = [a + b for a, b in zip(total_rewards, opponent_rewards)]

        # 4. Get observation for agent's *next* turn
        next_dice_roll = self.game.roll_dice()
        next_obs = self.get_agent_observation(next_dice_roll)

        return next_obs, total_rewards[self.agent_index], extra_turn
