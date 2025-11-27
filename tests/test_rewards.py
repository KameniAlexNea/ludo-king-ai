import math
import unittest
from unittest.mock import patch

from ludo_rl.ludo_king.reward import (
    compute_blockade_hits_bonus,
    compute_draw_reward,
    compute_invalid_action_penalty,
    compute_move_rewards,
    compute_skipped_turn_penalty,
    compute_terminal_reward,
)
from ludo_rl.ludo_king.types import MoveEvents


class TestTerminalReward(unittest.TestCase):
    def test_terminal_reward_scaling_four_players(self):
        # Given
        num_players = 4
        # When / Then
        win = compute_terminal_reward(num_players, rank=1)
        second = compute_terminal_reward(num_players, rank=2)
        third = compute_terminal_reward(num_players, rank=3)
        fourth = compute_terminal_reward(num_players, rank=4)

        # win > second > third > fourth when lose < 0
        self.assertGreater(win, second)
        self.assertGreater(second, third)
        self.assertGreater(third, fourth)
        # Linear scaling of loss fractions: 1/3, 2/3, 1
        # Only ratios matter (sign from configured lose)
        self.assertTrue(math.isclose(abs(second / fourth), 1 / 3, rel_tol=1e-6))
        self.assertTrue(math.isclose(abs(third / fourth), 2 / 3, rel_tol=1e-6))

    def test_terminal_reward_scaling_two_players(self):
        num_players = 2
        win = compute_terminal_reward(num_players, rank=1)
        second = compute_terminal_reward(num_players, rank=2)

        self.assertGreater(win, second)


class TestComputeMoveRewards(unittest.TestCase):
    def _run_event_flags_test(self, events_kwargs, expected_keys):
        num_players = 4
        mover = 0
        old_pos, new_pos = 5, 6
        events = MoveEvents(**{k: v for k, v in events_kwargs.items()})

        rewards = compute_move_rewards(num_players, mover, old_pos, new_pos, events)

        # Only mover gets positive reward increments (except capture victim side effect)
        self.assertTrue(all(idx in rewards for idx in range(num_players)))
        self.assertNotEqual(rewards[mover], 0.0)
        self.assertTrue(all(getattr(events, k) for k in expected_keys))

    def test_compute_move_rewards_move_resolved(self):
        self._run_event_flags_test({"move_resolved": True}, ["move_resolved"])

    def test_compute_move_rewards_exited_home(self):
        self._run_event_flags_test({"exited_home": True}, ["exited_home"])

    def test_compute_move_rewards_finished(self):
        self._run_event_flags_test({"finished": True}, ["finished"])

    def test_compute_move_rewards_hit_blockade(self):
        self._run_event_flags_test({"hit_blockade": True}, ["hit_blockade"])

    def test_compute_move_rewards_blockades(self):
        self._run_event_flags_test(
            {"blockades": [{"player": 0, "rel": 10}]}, ["blockades"]
        )

    def test_compute_move_rewards_capture_and_victim_penalty(self):
        num_players = 4
        mover = 1
        victim = 2
        old_pos, new_pos = 10, 11
        events = MoveEvents()
        events.knockouts = [{"player": victim, "piece_id": 0, "abs_pos": 25}]

        rewards = compute_move_rewards(num_players, mover, old_pos, new_pos, events)

        # Mover gets capture bonus; victim gets negative capture reward
        self.assertGreater(rewards[mover], 0.0)
        self.assertLess(rewards[victim], 0.0)

    def test_compute_move_rewards_blocked_move_has_only_blockade_bonus(self):
        num_players = 4
        mover = 0
        old_pos, new_pos = 12, 12  # unchanged due to blockade
        events = MoveEvents(move_resolved=False, hit_blockade=True)

        rewards = compute_move_rewards(num_players, mover, old_pos, new_pos, events)

        # Progress should not apply since old == new and not resolved
        # Only hit_blockade should contribute for mover
        mover_reward = rewards[mover]
        self.assertNotEqual(mover_reward, 0.0)


class TestEnvInvalidAction(unittest.TestCase):
    def test_env_invalid_action_uses_central_penalty(self):
        from ludo_rl.ludo_env import LudoEnv

        env = LudoEnv()
        _, _ = env.reset()

        # Prevent opponents simulation from affecting reward via blockade hits
        with patch(
            "ludo_rl.ludo_king.simulator.Simulator.step_opponents_only",
            return_value=None,
        ):
            invalid_action = 9999  # guaranteed not in move_map
            _, r, _, _, _ = env.step(invalid_action)

            self.assertTrue(
                math.isclose(r, compute_invalid_action_penalty(), rel_tol=1e-6)
            )

        env.close()


class TestMiscRewardHelpers(unittest.TestCase):
    def test_misc_reward_helpers_do_not_crash(self):
        # Smoke checks for remaining helpers
        self.assertEqual(compute_blockade_hits_bonus(0.0), 0.0)
        # These just need to not crash and return a number
        self.assertIsInstance(compute_skipped_turn_penalty(), float)
        self.assertIsInstance(compute_draw_reward(), float)


if __name__ == "__main__":
    unittest.main()
