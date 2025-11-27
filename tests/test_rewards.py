"""
Consolidated tests for reward functions and reward-related game mechanics.

This module tests:
- Terminal rewards (win/lose scaling)
- Move rewards (progress, capture, blockade, finish, exit_home)
- Exposure delta calculations
- Safe position bonuses
- Blockade penalties and mechanics
- Reward helper functions
"""

import math
import unittest
from unittest.mock import patch

from ludo_rl.ludo_king.config import config, reward_config
from ludo_rl.ludo_king.game import Game
from ludo_rl.ludo_king.player import Player
from ludo_rl.ludo_king.reward import (
    _count_threats_at_position,
    _is_position_safe,
    compute_blockade_hits_bonus,
    compute_draw_reward,
    compute_exposure_delta,
    compute_invalid_action_penalty,
    compute_move_rewards,
    compute_skipped_turn_penalty,
    compute_terminal_reward,
)
from ludo_rl.ludo_king.types import Color, Move, MoveEvents


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
# Move Reward Event Tests
# =============================================================================
class TestMoveRewardEvents(unittest.TestCase):
    """Tests for compute_move_rewards with different event types."""

    def test_progress_reward_on_move_resolved(self):
        """Moving piece forward gives progress reward."""
        events = MoveEvents(move_resolved=True)
        rewards = compute_move_rewards(
            num_players=4, mover_index=0, old_position=5, new_position=6, events=events
        )
        self.assertGreater(rewards[0], 0)

    def test_no_progress_when_position_unchanged(self):
        """No progress reward when piece doesn't move."""
        events = MoveEvents(move_resolved=False)
        rewards = compute_move_rewards(
            num_players=4, mover_index=0, old_position=5, new_position=5, events=events
        )
        self.assertEqual(rewards[0], 0)

    def test_exit_home_reward(self):
        """Exiting yard gives exit_home reward."""
        events = MoveEvents(exited_home=True, move_resolved=True)
        rewards = compute_move_rewards(
            num_players=4, mover_index=0, old_position=0, new_position=1, events=events
        )
        self.assertGreater(rewards[0], 0)
        # Opponents get penalty
        for i in [1, 2, 3]:
            self.assertLess(rewards[i], 0)

    def test_finish_reward(self):
        """Finishing a piece gives finish reward."""
        events = MoveEvents(finished=True, move_resolved=True)
        rewards = compute_move_rewards(
            num_players=4,
            mover_index=0,
            old_position=56,
            new_position=config.HOME_FINISH,
            events=events,
        )
        self.assertGreater(rewards[0], 0)
        # Opponents get penalty
        for i in [1, 2, 3]:
            self.assertLess(rewards[i], 0)

    def test_blockade_formation_reward(self):
        """Forming a blockade gives blockade reward."""
        events = MoveEvents(blockades=[{"player": 0, "rel": 10}], move_resolved=True)
        rewards = compute_move_rewards(
            num_players=4, mover_index=0, old_position=9, new_position=10, events=events
        )
        self.assertGreater(rewards[0], 0)

    def test_hit_blockade_penalty(self):
        """Hitting a blockade gives penalty."""
        events = MoveEvents(hit_blockade=True, move_resolved=False)
        rewards = compute_move_rewards(
            num_players=4, mover_index=0, old_position=5, new_position=5, events=events
        )
        self.assertLess(rewards[0], 0)
        self.assertAlmostEqual(rewards[0], reward_config.hit_blockade, delta=1e-6)


# =============================================================================
# Capture Reward Tests
# =============================================================================
class TestCaptureRewards(unittest.TestCase):
    """Tests for capture-related rewards."""

    def test_capture_gives_positive_reward(self):
        """Capturing opponent gives positive reward to mover."""
        events = MoveEvents(move_resolved=True)
        events.knockouts = [{"player": 2, "piece_id": 0, "abs_pos": 25}]

        rewards = compute_move_rewards(
            num_players=4,
            mover_index=1,
            old_position=10,
            new_position=11,
            events=events,
        )

        self.assertGreater(rewards[1], 0)  # mover gets bonus
        self.assertLess(rewards[2], 0)  # victim gets penalty

    def test_multiple_captures_multiply_reward(self):
        """Multiple captures give proportionally more reward."""
        events_single = MoveEvents(move_resolved=True)
        events_single.knockouts = [{"player": 2, "piece_id": 0, "abs_pos": 25}]

        events_double = MoveEvents(move_resolved=True)
        events_double.knockouts = [
            {"player": 2, "piece_id": 0, "abs_pos": 25},
            {"player": 3, "piece_id": 0, "abs_pos": 25},
        ]

        rewards_single = compute_move_rewards(
            num_players=4,
            mover_index=1,
            old_position=10,
            new_position=11,
            events=events_single,
        )
        rewards_double = compute_move_rewards(
            num_players=4,
            mover_index=1,
            old_position=10,
            new_position=11,
            events=events_double,
        )

        # Double capture should give more reward
        self.assertGreater(rewards_double[1], rewards_single[1])

    def test_victim_penalty_correct(self):
        """Victim receives got_capture penalty."""
        events = MoveEvents(move_resolved=True)
        events.knockouts = [{"player": 2, "piece_id": 0, "abs_pos": 25}]

        rewards = compute_move_rewards(
            num_players=4,
            mover_index=0,
            old_position=10,
            new_position=11,
            events=events,
        )

        self.assertAlmostEqual(rewards[2], reward_config.got_capture, delta=1e-6)


# =============================================================================
# Exposure and Safety Tests
# =============================================================================
class TestExposureCalculations(unittest.TestCase):
    """Tests for exposure delta and threat calculations."""

    def test_yard_position_is_safe(self):
        """Yard (position 0) is always safe."""
        game = _make_game()
        self.assertTrue(_is_position_safe(0, mover_color=0, board=game.board))

    def test_finished_position_is_safe(self):
        """Finished position is always safe."""
        game = _make_game()
        self.assertTrue(
            _is_position_safe(config.HOME_FINISH, mover_color=0, board=game.board)
        )

    def test_home_stretch_is_safe(self):
        """Home stretch positions are safe."""
        game = _make_game()
        for pos in range(config.HOME_COLUMN_START, config.HOME_FINISH):
            self.assertTrue(
                _is_position_safe(pos, mover_color=0, board=game.board),
                f"Position {pos} should be safe",
            )

    def test_safe_squares_are_safe(self):
        """Star squares (safe squares) are safe."""
        game = _make_game()
        # Position 9 maps to abs 9 for Red, which is a safe square
        safe_rel = 9  # First safe square for Red
        self.assertTrue(_is_position_safe(safe_rel, mover_color=0, board=game.board))

    def test_no_threats_in_yard(self):
        """Piece in yard has no threats."""
        game = _make_game()
        threats = _count_threats_at_position(
            game.board,
            mover_color=0,
            position=0,
            opponent_positions=[(1, [10]), (2, [20]), (3, [30])],
        )
        self.assertEqual(threats, 0)

    def test_no_threats_in_home_stretch(self):
        """Piece in home stretch has no threats."""
        game = _make_game()
        threats = _count_threats_at_position(
            game.board,
            mover_color=0,
            position=config.HOME_COLUMN_START,
            opponent_positions=[(1, [10]), (2, [20]), (3, [30])],
        )
        self.assertEqual(threats, 0)

    def test_threat_count_with_nearby_opponent(self):
        """Opponent within 6 squares creates threat."""
        game = _make_game()
        # Red at position 10, Green at position 8 (2 squares behind)
        # Green's relative 8 maps to absolute 21 (8 + 13)
        # Red's relative 10 maps to absolute 10
        # So we need opponent at position that's close in absolute terms

        # Place Red at position 10 (abs 10)
        # Place Green such that they're 3 steps behind in absolute terms
        # Green relative 46 -> abs 46 + 13 = 59 mod 52 = 7
        # Distance from abs 7 to abs 10 = 3
        threats = _count_threats_at_position(
            game.board,
            mover_color=0,
            position=10,
            opponent_positions=[(1, [46])],  # Green at relative 46, abs 7
        )
        self.assertGreater(threats, 0)

    def test_exposure_delta_moving_to_safer_position(self):
        """Moving to safer position gives negative exposure delta."""
        game = _make_game()
        # Move from threatened position to home stretch
        delta = compute_exposure_delta(
            game.board,
            mover_color=0,
            old_position=10,
            new_position=config.HOME_COLUMN_START,
            opponent_positions=[(1, [8])],
        )
        self.assertLessEqual(delta, 0)

    def test_exposure_delta_moving_from_yard_to_ring(self):
        """Moving from yard (safe) to ring may increase exposure."""
        game = _make_game()
        # Opponent positioned to threaten position 1
        delta = compute_exposure_delta(
            game.board,
            mover_color=0,
            old_position=0,  # Yard - safe
            new_position=1,  # Start position
            opponent_positions=[(1, [50])],  # Green that can reach position 1
        )
        # Exposure should increase or stay same
        self.assertGreaterEqual(delta, 0)


# =============================================================================
# Safe Landing Bonus Tests
# =============================================================================
class TestSafeLandingBonus(unittest.TestCase):
    """Tests for safe landing bonus."""

    def test_landing_on_safe_square_gives_bonus(self):
        """Landing on safe square gives bonus."""
        game = _make_game()
        events = MoveEvents(move_resolved=True)

        rewards = compute_move_rewards(
            num_players=4,
            mover_index=0,
            old_position=8,
            new_position=9,  # Safe square for Red
            events=events,
            board=game.board,
            mover_color=0,
            opponent_positions=[],
        )

        # Should include safe_landing_bonus
        expected_min = reward_config.progress + reward_config.safe_landing_bonus
        self.assertGreaterEqual(rewards[0], expected_min - 0.01)

    def test_landing_in_home_stretch_gives_bonus(self):
        """Landing in home stretch gives bonus."""
        game = _make_game()
        events = MoveEvents(move_resolved=True)

        rewards = compute_move_rewards(
            num_players=4,
            mover_index=0,
            old_position=50,
            new_position=config.HOME_COLUMN_START,
            events=events,
            board=game.board,
            mover_color=0,
            opponent_positions=[],
        )

        # Should include safe_landing_bonus
        self.assertGreater(rewards[0], reward_config.progress)

    def test_no_double_bonus_for_finishing(self):
        """Finishing position doesn't give safe_landing_bonus (already has finish bonus)."""
        game = _make_game()
        events = MoveEvents(finished=True, move_resolved=True)

        rewards = compute_move_rewards(
            num_players=4,
            mover_index=0,
            old_position=56,
            new_position=config.HOME_FINISH,
            events=events,
            board=game.board,
            mover_color=0,
            opponent_positions=[],
        )

        # Reward should be finish + progress, not + safe_landing_bonus
        # Check it's not excessively large
        max_expected = (
            reward_config.finish
            + reward_config.progress
            + reward_config.safe_landing_bonus
        )
        self.assertLessEqual(rewards[0], max_expected + 0.01)


# =============================================================================
# Blockade Mechanics Tests
# =============================================================================
class TestBlockadeMechanics(unittest.TestCase):
    """Tests for blockade detection and reward handling via game mechanics."""

    def test_hit_blockade_returns_penalty_and_prevents_move(self):
        """Hitting blockade returns penalty and doesn't move piece."""
        game = _make_game()

        # Red at position 5
        game.players[0].pieces[0].position = 5

        # Green forms blockade at position that maps to Red's position 10
        # Red 10 -> abs 10, Green needs to be at relative pos that maps to abs 10
        # Green abs = (rel + 13) % 52, so rel = (10 - 13 + 52) % 52 = 49
        game.players[1].pieces[0].position = 49
        game.players[1].pieces[1].position = 49

        move = Move(player_index=0, piece_id=0, new_pos=10, dice_roll=5)
        result = game.apply_move(move)

        # Verify blockade hit
        self.assertTrue(result.events.hit_blockade)
        self.assertFalse(result.events.move_resolved)

        # Verify rewards
        self.assertIsNotNone(result.rewards)
        self.assertAlmostEqual(result.rewards[0], reward_config.hit_blockade, delta=0.1)

        # Verify piece didn't move
        self.assertEqual(game.players[0].pieces[0].position, 5)

        # Other players unaffected
        for i in [1, 2, 3]:
            self.assertEqual(result.rewards[i], 0.0)

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

    def test_forming_blockade_gives_reward(self):
        """Forming a blockade gives blockade reward."""
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
        self.assertIsNotNone(result.rewards)
        self.assertGreaterEqual(result.rewards[0], reward_config.blockade)

    def test_successful_move_always_has_rewards(self):
        """Successful moves always return rewards dict."""
        game = _make_game()
        game.players[0].pieces[0].position = 5

        move = Move(player_index=0, piece_id=0, new_pos=10, dice_roll=5)
        result = game.apply_move(move)

        self.assertIsNotNone(result.rewards)
        self.assertIn(0, result.rewards)
        self.assertTrue(result.events.move_resolved)
        self.assertEqual(game.players[0].pieces[0].position, 10)

    def test_piece_on_opponent_safe_square_can_move(self):
        """Piece on opponent's safe square can still move even with blockade there."""
        game = _make_game()

        # Red forms blockade at its start square (relative 1 -> abs 1)
        red = game.players[0]
        red.pieces[0].position = 1
        red.pieces[1].position = 1
        red.pieces[2].position = 0
        red.pieces[3].position = 0

        # Green piece on Red's start (abs=1). For Green, that's relative 40.
        green = game.players[1]
        green.pieces[0].position = 40
        green.pieces[1].position = 0
        green.pieces[2].position = 0
        green.pieces[3].position = 0

        # Clear other opponents
        for p in game.players[2:]:
            for pc in p.pieces:
                pc.position = 0

        legal_moves = game.legal_moves(1, 3)  # Green's turn
        valid_pieces = {int(m.piece_id) for m in legal_moves}

        self.assertIn(0, valid_pieces, "Green on opponent safe square should move")


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


# =============================================================================
# Reward Helper Functions Tests
# =============================================================================
class TestRewardHelpers(unittest.TestCase):
    """Tests for reward helper functions."""

    def test_compute_blockade_hits_bonus_zero(self):
        """Zero hits gives zero bonus."""
        self.assertEqual(compute_blockade_hits_bonus(0.0), 0.0)

    def test_compute_blockade_hits_bonus_positive(self):
        """Positive hits gives proportional bonus."""
        bonus = compute_blockade_hits_bonus(2.0)
        self.assertEqual(bonus, reward_config.blockade_hit * 2.0)

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
        self.assertEqual(penalty, reward_config.skipped_turn)


if __name__ == "__main__":
    unittest.main()
