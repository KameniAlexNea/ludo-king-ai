from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from ludo_rl.ludo.config import config
from ludo_rl.ludo.game import LudoGame
from ludo_rl.ludo.simulator import GameSimulator


class GameSimulatorTests(unittest.TestCase):
    def test_configure_opponents_sequential(self) -> None:
        with mock.patch.dict(
            os.environ, {"OPPONENTS": "rusher,cautious", "STRATEGY_SELECTION": "1"}
        ):
            sim = GameSimulator(agent_index=0)
        opponent_names = [
            player.strategy_name
            for idx, player in enumerate(sim.game.players)
            if idx != sim.agent_index
        ]
        self.assertEqual(opponent_names, ["rusher", "cautious", "rusher"])

    def test_update_summaries_tracks_events(self) -> None:
        sim = GameSimulator(agent_index=0)
        sim.reset_summaries()

        move = {"new_pos": 10}
        abs_pos = sim.game.get_absolute_position(sim.agent_index, move["new_pos"])
        blockade_rel = 12
        result = {
            "reward": 1.5,
            "events": {
                "knockouts": [
                    {"player": sim.agent_index, "piece_id": 0, "abs_pos": abs_pos},
                    {
                        "player": (sim.agent_index + 1) % config.NUM_PLAYERS,
                        "piece_id": 1,
                        "abs_pos": abs_pos,
                    },
                ],
                "blockades": [
                    {"player": sim.agent_index, "relative_pos": blockade_rel},
                ],
            },
        }

        sim.update_summaries(sim.agent_index, move, result)
        self.assertEqual(sim.transition_summary["movement_heatmap"][move["new_pos"]], 1)
        self.assertAlmostEqual(sim.reward_heatmap[move["new_pos"]], result["reward"])

        rel_knockout_pos = sim.game.get_agent_relative_pos(sim.agent_index, abs_pos)
        self.assertEqual(sim.transition_summary["my_knockouts"][rel_knockout_pos], 1)
        self.assertEqual(sim.transition_summary["opp_knockouts"][rel_knockout_pos], 1)

        blockade_agent_rel = sim.get_agent_relative_pos_for_opp(
            sim.agent_index, blockade_rel
        )
        self.assertEqual(sim.transition_summary["new_blockades"][blockade_agent_rel], 1)

    def test_get_agent_relative_pos_for_opp_edge_cases(self) -> None:
        sim = GameSimulator(agent_index=0)
        self.assertEqual(sim.get_agent_relative_pos_for_opp(1, 0), 0)
        self.assertEqual(sim.get_agent_relative_pos_for_opp(1, 60), -1)

    def test_simulate_opponent_turns_accumulates_rewards(self) -> None:
        sim = GameSimulator(agent_index=0)

        rolls = iter([6] * 12)

        def fake_roll(self):
            return next(rolls)

        def fake_valid_moves(self, player_index: int, dice_roll: int):
            piece = self.players[player_index].pieces[0]
            return [{"piece": piece, "new_pos": 1, "dice_roll": dice_roll}]

        def fake_make_move(
            self, player_index: int, piece, new_pos: int, dice_roll: int
        ):
            piece.position = new_pos
            return {
                "reward": 0.5,
                "rewards": {0: 0.1, 1: 0.2, 2: 0.0, 3: 0.0},
                "events": {"knockouts": [], "blockades": [], "move_resolved": True},
                "extra_turn": False,
            }

        sim.reset_summaries()

        with (
            mock.patch.object(LudoGame, "roll_dice", fake_roll),
            mock.patch.object(LudoGame, "get_valid_moves", fake_valid_moves),
            mock.patch.object(LudoGame, "make_move", fake_make_move),
        ):
            rewards = sim.simulate_opponent_turns()

        self.assertAlmostEqual(rewards[sim.agent_index], 0.3)
        self.assertGreater(sum(sim.transition_summary["movement_heatmap"]), 0)

    def test_step_opponents_only_resets_summary(self) -> None:
        sim = GameSimulator(agent_index=0)
        sim.transition_summary["movement_heatmap"][2] = 4
        with mock.patch.object(
            GameSimulator,
            "simulate_opponent_turns",
            autospec=True,
            return_value=[0.0] * config.NUM_PLAYERS,
        ) as simulate_mock:
            rewards = sim.step_opponents_only()
        self.assertEqual(sim.transition_summary["movement_heatmap"][2], 0)
        self.assertEqual(rewards, [0.0] * config.NUM_PLAYERS)
        simulate_mock.assert_called_once_with(sim)

    def test_build_board_stack_shape(self) -> None:
        sim = GameSimulator(agent_index=0)
        stack = sim.game.build_board_tensor(sim.agent_index)
        self.assertEqual(stack.shape, (10, config.PATH_LENGTH))
        self.assertTrue(np.all(stack[5:] == 0))  # extra channels initialised to zero

    def test_step_handles_extra_turn_and_opponents(self) -> None:
        sim = GameSimulator(agent_index=0)
        piece = sim.game.players[sim.agent_index].pieces[0]
        piece.position = 10
        move = {"piece": piece, "new_pos": 13, "dice_roll": 3}

        with mock.patch.object(
            GameSimulator,
            "simulate_opponent_turns",
            autospec=True,
            return_value=[0.0] * config.NUM_PLAYERS,
        ) as simulate_mock:
            next_obs, reward, extra_turn = sim.step(move)

        self.assertIn("board_state", next_obs)
        self.assertFalse(extra_turn)
        simulate_mock.assert_called_once_with(sim)
        self.assertIsInstance(reward, float)

    def test_step_skips_opponents_on_extra_turn(self) -> None:
        sim = GameSimulator(agent_index=0)
        piece = sim.game.players[sim.agent_index].pieces[0]
        move = {"piece": piece, "new_pos": 1, "dice_roll": 6}

        with mock.patch.object(
            GameSimulator,
            "simulate_opponent_turns",
            autospec=True,
        ) as simulate_mock:
            _, _, extra_turn = sim.step(move)

        self.assertTrue(extra_turn)
        simulate_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
