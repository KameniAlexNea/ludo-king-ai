import random
from typing import List

from .config import config
from .game import LudoGame


class GameSimulator:
    """
    Manages the simulation, modified to integrate with the Gym env.
    """

    def __init__(self, agent_index):
        self.game = LudoGame()
        self.agent_index = agent_index
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

                move = random.choice(valid_moves)
                result = self.game.make_move(
                    opp_index, move["piece"], move["new_pos"], dice_roll
                )
                self.update_summaries(opp_index, move, result)

                for player_idx, value in result["rewards"].items():
                    cumulative_rewards[player_idx] += value

                extra_turn = result["extra_turn"]

        return cumulative_rewards

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
        dice_roll = agent_move.get("dice_roll")
        if dice_roll is None:
            dice_roll = self.game.roll_dice()

        # 1. Execute agent's move
        agent_result = self.game.make_move(
            self.agent_index, agent_move["piece"], agent_move["new_pos"], dice_roll
        )

        total_rewards = [0.0] * config.NUM_PLAYERS
        for player_idx, value in agent_result["rewards"].items():
            total_rewards[player_idx] += value

        # 2. Reset summaries (ready for opponent moves)
        self.reset_summaries()

        extra_turn = agent_result["extra_turn"]

        # 3. Simulate opponent turns if no extra turn was earned
        if not extra_turn:
            opponent_rewards = self.simulate_opponent_turns()
            total_rewards = [a + b for a, b in zip(total_rewards, opponent_rewards)]

        # 4. Get observation for agent's *next* turn
        next_dice_roll = self.game.roll_dice()
        next_obs = self.get_agent_observation(next_dice_roll)

        return next_obs, total_rewards[self.agent_index], extra_turn
