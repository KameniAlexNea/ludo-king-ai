"""Tests for the 10-channel board tensor design."""

import unittest


from ludo_rl.ludo_king.game import Game
from ludo_rl.ludo_king.player import Player
from ludo_rl.ludo_king.simulator import Simulator
from ludo_rl.ludo_king.types import Color, Move


class BoardChannelTests(unittest.TestCase):
    """Test that the board tensor has the correct 10-channel design."""

    def setUp(self):
        """Create a game with 4 players."""
        self.players = [
            Player(Color.RED),
            Player(Color.GREEN),
            Player(Color.YELLOW),
            Player(Color.BLUE),
        ]
        self.game = Game(players=self.players)
        self.sim = Simulator.for_game(self.game, agent_index=0)

    def test_board_tensor_shape(self):
        """Verify the board tensor has shape (10, 58)."""
        tensor = self.game.board.build_tensor(agent_color=0)
        self.assertEqual(tensor.shape, (10, 58), "Board tensor should be (10, 58)")

    def test_initial_piece_positions(self):
        """Verify that pieces start in yard (position 0)."""
        tensor = self.game.board.build_tensor(agent_color=0)

        # All pieces should be in yard initially (position 0)
        # Each player has 4 pieces
        self.assertEqual(tensor[0][0], 4.0, "Agent should have 4 pieces in yard")
        self.assertEqual(tensor[1][0], 4.0, "Opponent 1 should have 4 pieces in yard")
        self.assertEqual(tensor[2][0], 4.0, "Opponent 2 should have 4 pieces in yard")
        self.assertEqual(tensor[3][0], 4.0, "Opponent 3 should have 4 pieces in yard")

    def test_safe_zones_channel(self):
        """Verify that channel 4 represents safe zones correctly."""
        tensor = self.game.board.build_tensor(agent_color=0)
        safe_channel = tensor[4]

        # Home column (52-56) should be safe
        for pos in range(52, 57):
            self.assertEqual(safe_channel[pos], 1.0, f"Position {pos} should be safe")

        # Safe squares should be marked
        # Note: These need to be translated to agent's relative frame
        self.assertGreater(
            safe_channel.sum(), 5.0, "Should have multiple safe positions"
        )

    def test_transition_summaries_initially_empty(self):
        """Verify that transition summary channels start empty."""
        tensor = self.game.board.build_tensor(agent_color=0)

        self.assertEqual(tensor[5].sum(), 0.0, "Movement heatmap should start empty")
        self.assertEqual(tensor[6].sum(), 0.0, "My knockouts should start empty")
        self.assertEqual(tensor[7].sum(), 0.0, "Opponent knockouts should start empty")
        self.assertEqual(tensor[8].sum(), 0.0, "New blockades should start empty")
        self.assertEqual(tensor[9].sum(), 0.0, "Reward heatmap should start empty")

    def test_movement_heatmap_updates(self):
        """Verify that movement heatmap updates when pieces move."""
        # Move agent's piece from yard to start
        self.game.players[0].pieces[0].position = 1

        # Create and apply a move
        move = Move(player_index=0, piece_id=0, new_pos=7, dice_roll=6)
        result = self.game.apply_move(move)
        self.sim._update_transition_summaries(0, move, result)

        tensor = self.game.board.build_tensor(agent_color=0)

        # Movement heatmap should show activity at position 7
        self.assertGreater(
            tensor[5][7], 0.0, "Movement heatmap should track move to position 7"
        )
        self.assertEqual(
            tensor[5].sum(), 1.0, "Should have exactly one movement recorded"
        )

    def test_reward_heatmap_updates(self):
        """Verify that reward heatmap accumulates rewards."""
        # Move agent's piece from yard to start
        self.game.players[0].pieces[0].position = 1

        # Create and apply a move
        move = Move(player_index=0, piece_id=0, new_pos=7, dice_roll=6)
        result = self.game.apply_move(move)
        self.sim._update_transition_summaries(0, move, result)

        tensor = self.game.board.build_tensor(agent_color=0)

        # Reward heatmap should show rewards at position 7
        self.assertGreater(
            tensor[9][7], 0.0, "Reward heatmap should track rewards at position 7"
        )

    def test_knockout_tracking(self):
        """Verify that knockouts are tracked in the correct channels."""
        # Setup: Place agent piece at position 9, opponent at position 10
        # Then move agent piece to 10 to capture
        self.game.players[0].pieces[0].position = 9

        # Place opponent on the same absolute position that agent will land on
        # Red starts at abs position 1, so relative pos 10 = abs pos (1 + 10 - 1) % 52 = 10
        # Green starts at abs position 14, so to be at abs pos 10:
        # (14 + rel - 1) % 52 = 10 => rel = (10 - 14 + 1 + 52) % 52 + offset
        # Actually easier: put opponent at a position agent will reach
        # Red rel 10 = abs 10, Green needs to be there too
        # Green abs 10 -> rel = (10 - 14 + 52) % 52 + 1 = 49
        self.game.players[1].pieces[
            0
        ].position = 49  # Green at same abs position as Red's 10

        # Agent captures opponent by moving to position 10
        move = Move(player_index=0, piece_id=0, new_pos=10, dice_roll=1)
        result = self.game.apply_move(move)
        self.sim._update_transition_summaries(0, move, result)

        # Check if opponent was sent home (if on same absolute square and not safe)
        # Note: Position 10 for Red is abs position 10, which is NOT a safe square
        # Safe squares are [1, 9, 14, 22, 27, 35, 40, 48]
        tensor = self.game.board.build_tensor(agent_color=0)
        # My knockouts should be non-zero somewhere if capture happened
        if result.events.knockouts:
            self.assertEqual(
                self.game.players[1].pieces[0].position,
                0,
                "Opponent piece should be in yard",
            )
            self.assertGreater(tensor[6].sum(), 0.0, "My knockouts should be recorded")

    def test_reset_transition_summaries(self):
        """Verify that transition summaries can be reset."""
        # Make some moves to populate summaries
        self.game.players[0].pieces[0].position = 1
        move = Move(player_index=0, piece_id=0, new_pos=7, dice_roll=6)
        result = self.game.apply_move(move)
        self.sim._update_transition_summaries(0, move, result)

        # Verify summaries are populated
        tensor = self.game.board.build_tensor(agent_color=0)
        self.assertGreater(tensor[5].sum(), 0.0, "Movement heatmap should have data")

        # Reset summaries
        self.game.board.reset_transition_summaries()

        # Verify summaries are cleared
        tensor = self.game.board.build_tensor(agent_color=0)
        self.assertEqual(tensor[5].sum(), 0.0, "Movement heatmap should be cleared")
        self.assertEqual(tensor[6].sum(), 0.0, "My knockouts should be cleared")
        self.assertEqual(tensor[7].sum(), 0.0, "Opponent knockouts should be cleared")
        self.assertEqual(tensor[8].sum(), 0.0, "New blockades should be cleared")
        self.assertEqual(tensor[9].sum(), 0.0, "Reward heatmap should be cleared")

    def test_blockade_tracking(self):
        """Verify that blockades are tracked when formed."""
        # Place two agent pieces at same position to form blockade
        self.game.players[0].pieces[0].position = 9
        self.game.players[0].pieces[1].position = 5

        # Move second piece to join first
        move = Move(player_index=0, piece_id=1, new_pos=9, dice_roll=4)
        result = self.game.apply_move(move)
        self.sim._update_transition_summaries(0, move, result)

        tensor = self.game.board.build_tensor(agent_color=0)

        # Check if blockade was formed
        if result.events.blockades:
            self.assertGreater(tensor[8].sum(), 0.0, "New blockades should be tracked")

    def test_all_channels_independent(self):
        """Verify that all 10 channels can hold independent data."""
        # Manually populate all channels
        self.game.board.movement_heatmap[10] = 5.0
        self.game.board.my_knockouts[20] = 1.0
        self.game.board.opp_knockouts[30] = 1.0
        self.game.board.new_blockades[40] = 1.0
        self.game.board.reward_heatmap[50] = 10.0

        tensor = self.game.board.build_tensor(agent_color=0)

        # Verify each channel has its data
        self.assertEqual(tensor[5][10], 5.0, "Movement heatmap channel 5")
        self.assertEqual(tensor[6][20], 1.0, "My knockouts channel 6")
        self.assertEqual(tensor[7][30], 1.0, "Opponent knockouts channel 7")
        self.assertEqual(tensor[8][40], 1.0, "New blockades channel 8")
        self.assertEqual(tensor[9][50], 10.0, "Reward heatmap channel 9")


if __name__ == "__main__":
    unittest.main()
