"""Tests for blockade edge cases and reward handling."""

import unittest

from ludo_rl.ludo_env import LudoEnv
from ludo_rl.ludo_king.config import reward_config
from ludo_rl.ludo_king.game import Game
from ludo_rl.ludo_king.player import Player
from ludo_rl.ludo_king.types import Color, Move


class BlockadeRewardTests(unittest.TestCase):
    """Test that blockade scenarios are handled correctly."""

    def test_hit_blockade_returns_penalty_rewards(self):
        """Verify that hitting a blockade returns a penalty in rewards (not None)."""
        players = [
            Player(Color.RED),
            Player(Color.GREEN),
            Player(Color.YELLOW),
            Player(Color.BLUE),
        ]
        game = Game(players=players)

        # Setup: Create an opponent blockade
        # Red player (agent) at position 5
        game.players[0].pieces[0].position = 5

        # Green players forming blockade at position 10
        # Both at same absolute position
        game.players[1].pieces[
            0
        ].position = 49  # Green's position that maps to same abs as Red's 10
        game.players[1].pieces[1].position = 49

        # Agent tries to move through blockade
        move = Move(player_index=0, piece_id=0, new_pos=10, dice_roll=5)
        result = game.apply_move(move)

        # Verify the move failed
        self.assertTrue(result.events.hit_blockade, "Should hit blockade")
        self.assertFalse(result.events.move_resolved, "Move should not resolve")

        # Rewards are always computed centrally; mover gets hit_blockade penalty
        self.assertIsNotNone(result.rewards, "Rewards should be computed on blockade")
        self.assertIn(0, result.rewards)
        self.assertAlmostEqual(
            result.rewards[0],
            reward_config.hit_blockade,
            delta=1e-2,
            msg="Mover gets blockade penalty",
        )
        # Other players should not be affected
        for i in [1, 2, 3]:
            self.assertIn(i, result.rewards)
            self.assertEqual(result.rewards[i], 0.0)

        # Piece should not have moved
        self.assertEqual(
            game.players[0].pieces[0].position,
            5,
            "Piece should stay at original position",
        )

    def test_env_handles_none_rewards_gracefully(self):
        """Verify that the environment handles None rewards without crashing."""
        env = LudoEnv()

        # We can't easily force a blockade scenario in the environment,
        # but we can verify it doesn't crash during normal play
        obs, info = env.reset(seed=42)

        steps = 0
        max_steps = 50

        while steps < max_steps:
            # Take valid actions
            if info["action_mask"].any():
                valid_actions = [
                    i for i, valid in enumerate(info["action_mask"]) if valid
                ]
                action = valid_actions[0]
                obs, reward, terminated, truncated, info = env.step(action)

                # Reward should always be a valid float
                self.assertIsInstance(reward, (float, int), "Reward should be numeric")
                self.assertFalse(
                    float("inf") == reward, "Reward should not be infinity"
                )
                self.assertFalse(
                    float("-inf") == reward, "Reward should not be negative infinity"
                )

                if terminated or truncated:
                    break
            else:
                # No valid moves
                obs, reward, terminated, truncated, info = env.step(0)
                if terminated or truncated:
                    break

            steps += 1

    def test_blockade_penalty_applied(self):
        """Verify that hitting a blockade applies the correct penalty."""
        players = [
            Player(Color.RED),
            Player(Color.GREEN),
            Player(Color.YELLOW),
            Player(Color.BLUE),
        ]
        game = Game(players=players)

        # Setup: Create an opponent blockade
        game.players[0].pieces[0].position = 5
        game.players[1].pieces[0].position = 49  # Forms blockade at agent's position 10
        game.players[1].pieces[1].position = 49

        # Verify blockade exists (2 pieces at same position)
        abs_pos = game.board.absolute_position(0, 10)  # Red's position 10
        occupants = game.board.pieces_at_absolute(abs_pos, exclude_color=0)
        self.assertEqual(len(occupants), 2, "Should have 2 opponent pieces (blockade)")

        # Agent tries to move to blockade position
        move = Move(player_index=0, piece_id=0, new_pos=10, dice_roll=5)
        result = game.apply_move(move)

        # Verify hit_blockade flag is set
        self.assertTrue(result.events.hit_blockade, "Should detect blockade hit")
        self.assertFalse(result.events.move_resolved, "Move should fail")

        # Mover gets blockade penalty in rewards
        self.assertIsNotNone(result.rewards)
        self.assertAlmostEqual(
            result.rewards[0], reward_config.hit_blockade, delta=1e-2
        )

    def test_successful_move_has_rewards(self):
        """Verify that successful moves always have rewards dict."""
        players = [
            Player(Color.RED),
            Player(Color.GREEN),
            Player(Color.YELLOW),
            Player(Color.BLUE),
        ]
        game = Game(players=players)

        # Setup: Simple move that will succeed
        game.players[0].pieces[0].position = 5

        # Agent makes a valid move
        move = Move(player_index=0, piece_id=0, new_pos=10, dice_roll=5)
        result = game.apply_move(move)

        # Successful move should have rewards
        self.assertIsNotNone(result.rewards, "Successful move should have rewards")
        self.assertIn(0, result.rewards, "Should have reward for agent player")
        self.assertTrue(result.events.move_resolved, "Move should resolve")

        # Piece should have moved
        self.assertEqual(
            game.players[0].pieces[0].position, 10, "Piece should move to new position"
        )


if __name__ == "__main__":
    unittest.main()
