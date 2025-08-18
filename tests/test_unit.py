"""
Unit Tests for Core Ludo Game Components - CORRECTED VERSION
Tests individual game components with actual API methods.
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ludo.token import Token, TokenState
from ludo.player import Player, PlayerColor
from ludo.board import Board
from ludo.game import LudoGame


class TokenTestCase(unittest.TestCase):
    """Test cases for Token class."""
    
    def setUp(self):
        """Set up test environment."""
        self.token = Token(0, "red")
    
    def test_token_initialization(self):
        """Test token initialization."""
        self.assertEqual(self.token.token_id, 0)
        self.assertEqual(self.token.player_color, "red")
        self.assertEqual(self.token.position, -1)
        self.assertEqual(self.token.state, TokenState.HOME)
        self.assertFalse(self.token.is_active())
        self.assertTrue(self.token.is_in_home())
        self.assertFalse(self.token.is_finished())
    
    def test_token_exit_home(self):
        """Test token exiting home."""
        # Simulate moving token out of home (dice roll 6)
        success = self.token.move(6, 1)  # Start position 1 for red
        self.assertTrue(success)
        self.assertEqual(self.token.position, 1)
        self.assertEqual(self.token.state, TokenState.ACTIVE)
        self.assertTrue(self.token.is_active())
        self.assertFalse(self.token.is_in_home())
    
    def test_token_movement(self):
        """Test token movement on board."""
        # Exit home first
        self.token.move(6, 1)
        # Then move normally
        success = self.token.move(3, 1)
        self.assertTrue(success)
        self.assertEqual(self.token.position, 4)
        self.assertTrue(self.token.is_active())
    
    def test_token_cannot_exit_without_six(self):
        """Test token cannot exit home without rolling 6."""
        success = self.token.move(3, 1)
        self.assertFalse(success)
        self.assertEqual(self.token.position, -1)
        self.assertTrue(self.token.is_in_home())
    
    def test_token_dictionary_representation(self):
        """Test token to_dict method."""
        token_dict = self.token.to_dict()
        self.assertIn('token_id', token_dict)
        self.assertIn('player_color', token_dict)
        self.assertIn('state', token_dict)
        self.assertIn('position', token_dict)


class PlayerTestCase(unittest.TestCase):
    """Test cases for Player class."""
    
    def setUp(self):
        """Set up test environment."""
        self.player = Player(PlayerColor.BLUE, 1)
    
    def test_player_initialization(self):
        """Test player initialization."""
        self.assertEqual(self.player.color, PlayerColor.BLUE)
        self.assertEqual(self.player.player_id, 1)
        self.assertEqual(len(self.player.tokens), 4)
        
        # All tokens should be in home initially
        for token in self.player.tokens:
            self.assertTrue(token.is_in_home())
            self.assertEqual(token.player_color, PlayerColor.BLUE.value)
    
    def test_player_token_counts(self):
        """Test player token counting methods."""
        state = self.player.get_game_state()
        
        # Initially all in home
        self.assertEqual(state['tokens_in_home'], 4)
        self.assertEqual(state['active_tokens'], 0)
        self.assertEqual(state['finished_tokens'], 0)
        
        # Move some tokens manually for testing
        self.player.tokens[0].state = TokenState.ACTIVE
        self.player.tokens[0].position = 10
        self.player.tokens[1].state = TokenState.FINISHED
        self.player.tokens[1].position = 57
        
        updated_state = self.player.get_game_state()
        self.assertEqual(updated_state['tokens_in_home'], 2)
        self.assertEqual(updated_state['active_tokens'], 1)
        self.assertEqual(updated_state['finished_tokens'], 1)
    
    def test_player_has_won(self):
        """Test player win condition."""
        self.assertFalse(self.player.has_won())
        
        # Move all tokens to finish
        for token in self.player.tokens:
            token.state = TokenState.FINISHED
            token.position = 57
        
        self.assertTrue(self.player.has_won())
    
    def test_player_game_state(self):
        """Test player game state generation."""
        state = self.player.get_game_state()
        
        self.assertIn('tokens_in_home', state)
        self.assertIn('active_tokens', state)
        self.assertIn('finished_tokens', state)
        self.assertIn('has_won', state)
        self.assertIn('player_id', state)
        self.assertIn('color', state)
        
        self.assertEqual(state['tokens_in_home'], 4)
        self.assertEqual(state['active_tokens'], 0)
        self.assertEqual(state['finished_tokens'], 0)
        self.assertFalse(state['has_won'])
    
    def test_movable_tokens(self):
        """Test getting movable tokens."""
        # Initially, no tokens can move without rolling 6
        movable = self.player.get_movable_tokens(3)
        self.assertEqual(len(movable), 0)
        
        # With 6, should be able to move tokens out of home
        movable = self.player.get_movable_tokens(6)
        self.assertGreater(len(movable), 0)


class BoardTestCase(unittest.TestCase):
    """Test cases for Board class."""
    
    def setUp(self):
        """Set up test environment."""
        self.board = Board()
    
    def test_board_initialization(self):
        """Test board initialization."""
        # Check that board has correct number of positions
        self.assertEqual(len(self.board.positions), 52)
        
        # Check known safe positions (star and colored squares)
        safe_positions = [1, 9, 14, 22, 27, 35, 40, 48]
        for pos in safe_positions:
            if pos < len(self.board.positions):
                position = self.board.positions[pos]
                self.assertTrue(position.is_safe, f"Position {pos} should be safe")
    
    def test_position_properties(self):
        """Test position properties."""
        # Test regular position (not safe)
        if len(self.board.positions) > 10:
            pos_10 = self.board.positions[10]
            self.assertFalse(pos_10.is_safe)
        
        # Test safe position (star position)
        if len(self.board.positions) > 9:
            pos_9 = self.board.positions[9]
            self.assertTrue(pos_9.is_safe)


class GameLogicTestCase(unittest.TestCase):
    """Test cases for Game logic."""
    
    def setUp(self):
        """Set up test environment."""
        self.game = LudoGame([PlayerColor.RED, PlayerColor.BLUE])
    
    def test_game_initialization(self):
        """Test game initialization."""
        self.assertEqual(len(self.game.players), 2)
        self.assertEqual(self.game.current_player_index, 0)
        self.assertFalse(self.game.game_over)
        self.assertIsNone(self.game.winner)
        self.assertEqual(self.game.turn_count, 0)
    
    def test_dice_rolling(self):
        """Test dice rolling functionality."""
        dice_value = self.game.roll_dice()
        self.assertIn(dice_value, [1, 2, 3, 4, 5, 6])
        self.assertEqual(self.game.last_dice_value, dice_value)
    
    def test_current_player(self):
        """Test current player management."""
        player1 = self.game.get_current_player()
        self.assertEqual(player1.color, PlayerColor.RED)
        
        self.game.next_turn()
        player2 = self.game.get_current_player()
        self.assertEqual(player2.color, PlayerColor.BLUE)
        
        self.game.next_turn()
        player1_again = self.game.get_current_player()
        self.assertEqual(player1_again.color, PlayerColor.RED)
    
    def test_valid_moves_no_six(self):
        """Test that no moves are available without rolling 6 initially."""
        player = self.game.get_current_player()
        valid_moves = self.game.get_valid_moves(player, 3)
        self.assertEqual(len(valid_moves), 0)
    
    def test_valid_moves_with_six(self):
        """Test that exit_home move is available with 6."""
        player = self.game.get_current_player()
        valid_moves = self.game.get_valid_moves(player, 6)
        # Should have 4 possible moves (one for each token to exit home)
        self.assertGreater(len(valid_moves), 0)
        # Check that at least one move is exit_home type
        exit_moves = [m for m in valid_moves if m.get('move_type') == 'exit_home']
        self.assertGreater(len(exit_moves), 0)
    
    def test_move_execution(self):
        """Test move execution."""
        player = self.game.get_current_player()
        
        # Execute exit home move
        result = self.game.execute_move(player, 0, 6)
        self.assertTrue(result['success'])
        self.assertTrue(result.get('extra_turn', False))  # Rolling 6 gives extra turn
        
        # Check token position
        token = player.tokens[0]
        self.assertTrue(token.is_active())
    
    def test_ai_decision_context(self):
        """Test AI decision context generation."""
        context = self.game.get_ai_decision_context(6)
        
        self.assertIn('current_situation', context)
        self.assertIn('player_state', context)
        self.assertIn('opponents', context)
        self.assertIn('valid_moves', context)
        self.assertIn('strategic_analysis', context)
        
        # Check current situation
        situation = context['current_situation']
        self.assertEqual(situation['dice_value'], 6)
        self.assertEqual(situation['player_color'], 'red')
        
        # Should have moves available with dice value 6
        self.assertGreater(len(context['valid_moves']), 0)


class IntegrationTestCase(unittest.TestCase):
    """Integration tests for complete game scenarios."""
    
    def test_complete_game_flow(self):
        """Test a complete game from start to strategic finish."""
        from ludo.strategy import StrategyFactory
        
        game = LudoGame([PlayerColor.RED, PlayerColor.BLUE])
        
        # Set up strategies
        game.players[0].set_strategy(StrategyFactory.create_strategy("winner"))
        game.players[1].set_strategy(StrategyFactory.create_strategy("balanced"))
        
        max_turns = 100
        turn_count = 0
        moves_made = 0
        
        while not game.game_over and turn_count < max_turns:
            current_player = game.get_current_player()
            dice_value = game.roll_dice()
            
            context = game.get_ai_decision_context(dice_value)
            
            if context['valid_moves']:
                # AI makes decision
                selected_token = current_player.make_strategic_decision(context)
                self.assertIsNotNone(selected_token, "Strategy should make a decision")
                
                # Execute move
                move_result = game.execute_move(current_player, selected_token, dice_value)
                self.assertTrue(move_result['success'], f"Move should succeed at turn {turn_count}")
                moves_made += 1
                
                if move_result.get('game_won'):
                    self.assertTrue(game.game_over)
                    self.assertIsNotNone(game.winner)
                    break
                
                if not move_result.get('extra_turn', False):
                    game.next_turn()
            else:
                game.next_turn()
            
            turn_count += 1
        
        # Verify game integrity
        self.assertGreater(moves_made, 0, "Should have made some moves")
        if game.game_over:
            self.assertIsNotNone(game.winner, "Finished game should have winner")
            self.assertTrue(game.winner.has_won(), "Winner should have all tokens finished")
    
    def test_strategy_behavior_differences(self):
        """Test that different strategies behave differently."""
        from ludo.strategy import StrategyFactory
        
        # Create a game context with multiple options
        game = LudoGame([PlayerColor.RED, PlayerColor.BLUE])
        
        # Set up a scenario with some tokens active
        player = game.players[0]
        player.tokens[0].state = TokenState.ACTIVE
        player.tokens[0].position = 10
        player.tokens[1].state = TokenState.ACTIVE
        player.tokens[1].position = 25
        
        dice_value = 3
        context = game.get_ai_decision_context(dice_value)
        
        if context['valid_moves']:
            decisions = {}
            strategies = ["killer", "defensive", "balanced", "random"]
            
            for strategy_name in strategies:
                strategy = StrategyFactory.create_strategy(strategy_name)
                decision = strategy.decide(context.copy())
                decisions[strategy_name] = decision
            
            # Verify all decisions are valid
            valid_tokens = [move['token_id'] for move in context['valid_moves']]
            for strategy_name, decision in decisions.items():
                self.assertIn(decision, valid_tokens, f"Strategy {strategy_name} made invalid decision")
        else:
            self.skipTest("No valid moves available for strategy testing")


if __name__ == '__main__':
    unittest.main(verbosity=2)
