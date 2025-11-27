"""
Tests for sparse reward functions.

This module tests the simplified sparse reward system:
- Terminal rewards (win/lose/draw)
- Sparse milestone rewards (capture, finish)
- Reward helper functions
"""

import math
import unittest
from unittest.mock import patch

from ludo_rl.ludo_king.config import config, reward_config
from ludo_rl.ludo_king.game import Game
from ludo_rl.ludo_king.player import Player
from ludo_rl.ludo_king.reward import (
    compute_draw_reward,
    compute_invalid_action_penalty,
    compute_skipped_turn_penalty,
    compute_sparse_rewards,
    compute_terminal_reward,
)
from ludo_rl.ludo_king.types import Color, KnockoutEvent, Move, MoveEvents


def _make_game() -> Game:
    """Create a standard 4-player game for testing."""
    players = [
        Player(Color.RED),
        Player(Color.GREEN),
        Player(Color.YELLOW),
        Player(Color.BLUE),
    ]
    return Game(players=players)


# =============================================================================
# Terminal Reward Tests
# =============================================================================
class TestTerminalReward(unittest.TestCase):
    """Tests for terminal (win/lose) reward calculations."""

    def test_win_reward_is_positive(self):
        """Winner should receive positive reward."""
        reward = compute_terminal_reward(num_players=4, rank=1)
        self.assertGreater(reward, 0)

    def test_lose_reward_is_negative(self):
        """Losers should receive negative reward."""
        reward = compute_terminal_reward(num_players=4, rank=4)
        self.assertLess(reward, 0)

    def test_terminal_reward_scaling_four_players(self):
        """Test reward scaling for 4 players: win > 2nd > 3rd > 4th."""
        num_players = 4
        win = compute_terminal_reward(num_players, rank=1)
        second = compute_terminal_reward(num_players, rank=2)
        third = compute_terminal_reward(num_players, rank=3)
        fourth = compute_terminal_reward(num_players, rank=4)

        # win > second > third > fourth when lose < 0
        self.assertGreater(win, second)
        self.assertGreater(second, third)
        self.assertGreater(third, fourth)

        # Linear scaling of loss fractions: 1/3, 2/3, 1
        self.assertTrue(math.isclose(abs(second / fourth), 1 / 3, rel_tol=1e-6))
        self.assertTrue(math.isclose(abs(third / fourth), 2 / 3, rel_tol=1e-6))

    def test_terminal_reward_scaling_two_players(self):
        """Test reward scaling for 2 players."""
        num_players = 2
        win = compute_terminal_reward(num_players, rank=1)
        second = compute_terminal_reward(num_players, rank=2)

        self.assertGreater(win, second)

    def test_terminal_reward_scaling_three_players(self):
        """Test reward scaling for 3 players."""
        num_players = 3
        win = compute_terminal_reward(num_players, rank=1)
        second = compute_terminal_reward(num_players, rank=2)
        third = compute_terminal_reward(num_players, rank=3)

        self.assertGreater(win, second)
        self.assertGreater(second, third)


# =============================================================================
# Sparse Reward Tests
# =============================================================================
class TestSparseRewards(unittest.TestCase):
    """Tests for compute_sparse_rewards with different event types."""

    def test_no_events_gives_zero_reward(self):
        """No significant events gives zero reward (sparse)."""
        events = MoveEvents(move_resolved=True)
        rewards = compute_sparse_rewards(num_players=4, mover_index=0, events=events)
        self.assertEqual(rewards[0], 0.0)

    def test_finish_reward(self):
        """Finishing a piece gives finish reward."""
        events = MoveEvents(finished=True, move_resolved=True)
        rewards = compute_sparse_rewards(num_players=4, mover_index=0, events=events)
        self.assertAlmostEqual(rewards[0], reward_config.finish, delta=1e-6)

    def test_capture_gives_reward(self):
        """Capturing opponent gives positive reward to mover."""
        events = MoveEvents(move_resolved=True)
        events.knockouts = [KnockoutEvent(player=2, piece_id=0, abs_pos=25)]

        rewards = compute_sparse_rewards(num_players=4, mover_index=1, events=events)

        self.assertAlmostEqual(rewards[1], reward_config.capture, delta=1e-6)
        self.assertAlmostEqual(rewards[2], reward_config.got_captured, delta=1e-6)

    def test_multiple_captures_multiply_reward(self):
        """Multiple captures give proportionally more reward."""
        events_single = MoveEvents(move_resolved=True)
        events_single.knockouts = [KnockoutEvent(player=2, piece_id=0, abs_pos=25)]

        events_double = MoveEvents(move_resolved=True)
        events_double.knockouts = [
            KnockoutEvent(player=2, piece_id=0, abs_pos=25),
            KnockoutEvent(player=3, piece_id=0, abs_pos=25),
        ]

        rewards_single = compute_sparse_rewards(
            num_players=4, mover_index=1, events=events_single
        )
        rewards_double = compute_sparse_rewards(
            num_players=4, mover_index=1, events=events_double
        )

        # Double capture should give twice the reward
        self.assertAlmostEqual(rewards_double[1], rewards_single[1] * 2, delta=1e-6)

    def test_hit_blockade_no_reward(self):
        """Hitting a blockade gives NO reward (sparse rewards)."""
        events = MoveEvents(hit_blockade=True, move_resolved=False)
        rewards = compute_sparse_rewards(num_players=4, mover_index=0, events=events)
        self.assertEqual(rewards[0], 0.0)

    def test_blockade_formation_no_reward(self):
        """Forming a blockade gives NO reward (sparse rewards)."""
        from ludo_rl.ludo_king.types import BlockadeEvent

        events = MoveEvents(move_resolved=True)
        events.blockades = [BlockadeEvent(player=0, rel_pos=10)]
        rewards = compute_sparse_rewards(num_players=4, mover_index=0, events=events)
        self.assertEqual(rewards[0], 0.0)

    def test_exit_home_no_reward(self):
        """Exiting yard gives NO reward (sparse rewards)."""
        events = MoveEvents(exited_home=True, move_resolved=True)
        rewards = compute_sparse_rewards(num_players=4, mover_index=0, events=events)
        self.assertEqual(rewards[0], 0.0)

    def test_finish_plus_capture(self):
        """Finish and capture together gives combined reward."""
        events = MoveEvents(finished=True, move_resolved=True)
        events.knockouts = [KnockoutEvent(player=2, piece_id=0, abs_pos=25)]

        rewards = compute_sparse_rewards(num_players=4, mover_index=0, events=events)

        expected = reward_config.finish + reward_config.capture
        self.assertAlmostEqual(rewards[0], expected, delta=1e-6)


# =============================================================================
# Blockade Mechanics Tests (game.py behavior - no rewards)
# =============================================================================
class TestBlockadeMechanics(unittest.TestCase):
    """Tests for blockade detection via game mechanics."""

    def test_hit_blockade_prevents_move(self):
        """Hitting blockade doesn't move piece and returns None rewards."""
        game = _make_game()

        # Red at position 5
        game.players[0].pieces[0].position = 5

        # Green forms blockade at position that maps to Red's position 10
        game.players[1].pieces[0].position = 49
        game.players[1].pieces[1].position = 49

        move = Move(player_index=0, piece_id=0, new_pos=10, dice_roll=5)
        result = game.apply_move(move)

        # Verify blockade hit
        self.assertTrue(result.events.hit_blockade)
        self.assertFalse(result.events.move_resolved)

        # game.py no longer returns rewards (decoupled)
        self.assertIsNone(result.rewards)

        # Verify piece didn't move
        self.assertEqual(game.players[0].pieces[0].position, 5)

    def test_own_blockade_blocks_piece_behind(self):
        """Own blockade blocks piece trying to land on it."""
        game = _make_game()

        agent = game.players[0]
        blockade_pos = 6
        agent.pieces[0].position = blockade_pos
        agent.pieces[1].position = blockade_pos
        agent.pieces[2].position = blockade_pos - 3  # Behind blockade
        agent.pieces[3].position = 0

        # Clear opponents
        for p in game.players[1:]:
            for pc in p.pieces:
                pc.position = 0

        # Piece 2 with dice=3 would land on own blockade
        legal_moves = game.legal_moves(0, 3)
        valid_pieces = {int(m.piece_id) for m in legal_moves}

        self.assertIn(0, valid_pieces)  # Can move
        self.assertIn(1, valid_pieces)  # Can move
        self.assertNotIn(2, valid_pieces)  # Blocked by own blockade

    def test_forming_blockade_sets_event(self):
        """Forming a blockade sets blockade event (no reward from game.py)."""
        game = _make_game()

        # Choose non-safe ring square
        target_rel = 10
        target_abs = game.board.absolute_position(int(Color.RED), target_rel)
        while target_abs in config.SAFE_SQUARES_ABS or target_rel <= 2:
            target_rel += 1
            target_abs = game.board.absolute_position(int(Color.RED), target_rel)

        # Place one piece at target, one behind
        game.players[0].pieces[0].position = target_rel
        game.players[0].pieces[1].position = target_rel - 1

        move = Move(player_index=0, piece_id=1, new_pos=target_rel, dice_roll=1)
        result = game.apply_move(move)

        self.assertTrue(result.events.move_resolved)
        self.assertTrue(result.events.blockades)
        # game.py no longer returns rewards (decoupled)
        self.assertIsNone(result.rewards)

    def test_successful_move_returns_none_rewards(self):
        """Successful moves return None rewards (computed by env)."""
        game = _make_game()
        game.players[0].pieces[0].position = 5

        move = Move(player_index=0, piece_id=0, new_pos=10, dice_roll=5)
        result = game.apply_move(move)

        # game.py no longer returns rewards
        self.assertIsNone(result.rewards)
        self.assertTrue(result.events.move_resolved)
        self.assertEqual(game.players[0].pieces[0].position, 10)


# =============================================================================
# Environment Integration Tests
# =============================================================================
class TestEnvRewardIntegration(unittest.TestCase):
    """Tests for reward handling in the environment."""

    def test_env_invalid_action_uses_central_penalty(self):
        """Invalid action returns compute_invalid_action_penalty value."""
        from ludo_rl.ludo_env import LudoEnv

        env = LudoEnv()
        _, _ = env.reset()

        with patch(
            "ludo_rl.ludo_king.simulator.Simulator.step_opponents_only",
            return_value=None,
        ):
            invalid_action = 9999
            _, r, _, _, _ = env.step(invalid_action)

            self.assertTrue(
                math.isclose(r, compute_invalid_action_penalty(), rel_tol=1e-6)
            )

        env.close()

    def test_env_rewards_always_numeric(self):
        """Environment always returns numeric rewards (not inf/nan)."""
        from ludo_rl.ludo_env import LudoEnv

        env = LudoEnv()
        _, info = env.reset(seed=42)

        for _ in range(50):
            if info["action_mask"].any():
                valid_actions = [
                    i for i, valid in enumerate(info["action_mask"]) if valid
                ]
                action = valid_actions[0]
                _, reward, terminated, truncated, info = env.step(action)

                self.assertIsInstance(reward, (float, int))
                self.assertFalse(math.isinf(reward))
                self.assertFalse(math.isnan(reward))

                if terminated or truncated:
                    break
            else:
                _, reward, terminated, truncated, info = env.step(0)
                if terminated or truncated:
                    break

        env.close()

    def test_got_captured_penalty_flows_to_agent(self):
        """When opponent captures agent's piece, agent receives got_captured penalty.
        
        This is a critical integration test - verifies the full flow:
        Simulator._process_move_result should accumulate got_captured penalty
        which then flows through sim.get_agent_reward() to the env.
        
        Instead of mocking the entire step_opponents_only, we directly test
        the reward accumulation mechanism.
        """
        from ludo_rl.ludo_env import LudoEnv
        from ludo_rl.ludo_king.types import MoveEvents, KnockoutEvent, MoveResult, Move
        from ludo_rl.ludo_king.config import reward_config

        env = LudoEnv()
        env.reset(seed=42)

        # The agent is player 0
        # Directly simulate what happens when an opponent captures agent's piece
        # by calling _process_move_result
        
        # Reset the accumulator
        env.sim._agent_reward_acc = 0.0
        
        # Create a fake knockout event where agent (player 0) is the victim
        fake_events = MoveEvents(move_resolved=True)
        fake_events.knockouts = [KnockoutEvent(player=0, piece_id=0, abs_pos=10)]
        
        # Process the fake result - opponent 1 captured agent's piece
        fake_result = MoveResult(
            old_position=5,
            new_position=10,
            extra_turn=False,
            events=fake_events,
            rewards=None,
        )
        fake_move = Move(player_index=1, piece_id=0, new_pos=10, dice_roll=6)
        env.sim._process_move_result(1, fake_move, fake_result)
        
        # Now get the accumulated reward
        accumulated_reward = env.sim.get_agent_reward()
        
        env.close()

        # Verify got_captured penalty was accumulated
        expected_penalty = reward_config.got_captured
        self.assertAlmostEqual(
            accumulated_reward, expected_penalty,
            places=5,
            msg=f"Agent should receive got_captured penalty {expected_penalty}, got {accumulated_reward}"
        )

    def test_simulator_accumulates_got_captured_rewards(self):
        """Simulator._process_move_result accumulates got_captured penalty for agent.
        
        Unit test for the simulator's reward accumulation logic.
        """
        from ludo_rl.ludo_king.game import Game
        from ludo_rl.ludo_king.player import Player
        from ludo_rl.ludo_king.simulator import Simulator
        from ludo_rl.ludo_king.types import MoveEvents, KnockoutEvent, MoveResult, Move

        # Create game with agent at index 0
        players = [Player(color=c) for c in [0, 1, 2, 3]]
        game = Game(players=players)
        sim = Simulator.for_game(game, agent_index=0)

        # Reset accumulator
        sim._agent_reward_acc = 0.0

        # Create event where opponent (player 1) captures agent's piece (player 0)
        events = MoveEvents(move_resolved=True)
        events.knockouts = [KnockoutEvent(player=0, piece_id=0, abs_pos=25)]

        result = MoveResult(
            old_position=20,
            new_position=25,
            extra_turn=False,
            events=events,
            rewards=None,  # game.py no longer computes rewards
        )
        move = Move(player_index=1, piece_id=0, new_pos=25, dice_roll=5)

        # Process the result
        sim._process_move_result(1, move, result)

        # Agent should have accumulated got_captured penalty
        self.assertAlmostEqual(
            sim._agent_reward_acc,
            reward_config.got_captured,
            delta=1e-6,
            msg=f"Agent reward accumulator should have {reward_config.got_captured}, got {sim._agent_reward_acc}"
        )


# =============================================================================
# Reward Helper Functions Tests
# =============================================================================
class TestRewardHelpers(unittest.TestCase):
    """Tests for reward helper functions."""

    def test_compute_skipped_turn_penalty_returns_float(self):
        """Skipped turn penalty returns float."""
        penalty = compute_skipped_turn_penalty()
        self.assertIsInstance(penalty, float)

    def test_compute_draw_reward_returns_float(self):
        """Draw reward returns float."""
        reward = compute_draw_reward()
        self.assertIsInstance(reward, float)

    def test_compute_invalid_action_penalty_returns_float(self):
        """Invalid action penalty returns float."""
        penalty = compute_invalid_action_penalty()
        self.assertIsInstance(penalty, float)
        self.assertEqual(penalty, reward_config.invalid_action)


if __name__ == "__main__":
    unittest.main()
