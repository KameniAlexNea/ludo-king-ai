"""
Simple example showing how to integrate your AI model with the Ludo environment.
This demonstrates the minimal interface needed for AI integration.
"""

from ludo import LudoGame, PlayerColor
import json


def random_ai_example():
    """Example showing how to integrate a random AI with the Ludo environment."""
    import random
    
    print("=== Random AI Example ===")
    
    # Create a simple 2-player game
    game = LudoGame([PlayerColor.RED, PlayerColor.BLUE])
    
    # Play for a few turns to show the interface
    for turn in range(10):
        if game.game_over:
            break
            
        current_player = game.get_current_player()
        print(f"\nTurn {turn + 1}: {current_player.color.value}")
        
        # Roll dice
        dice_value = game.roll_dice()
        print(f"Dice: {dice_value}")
        
        # Get AI context
        context = game.get_ai_decision_context(dice_value)
        
        # Random AI decision
        valid_moves = context['valid_moves']
        if valid_moves:
            # Random choice (replace this with your AI model)
            selected_move = random.choice(valid_moves)
            token_id = selected_move['token_id']
            
            print(f"AI chooses token {token_id}")
            print(f"Move details: {selected_move}")
            
            # Execute move
            result = game.execute_move(current_player, token_id, dice_value)
            print(f"Result: {result}")
            
            if not result['extra_turn']:
                game.next_turn()
        else:
            print("No valid moves")
            game.next_turn()


def neural_network_interface_example():
    """Example showing how to interface with a neural network model."""
    
    print("\n=== Neural Network Interface Example ===")
    
    class MockNeuralNetwork:
        """Mock neural network that shows the expected interface."""
        
        def predict(self, game_state):
            """
            Expected interface for your neural network.
            
            Args:
                game_state: Dictionary containing complete game state
                
            Returns:
                token_id: Integer (0-3) indicating which token to move
            """
            # Your neural network would process the game_state here
            # and return the best token_id to move
            
            # For this example, just choose the first valid move
            valid_moves = game_state.get('valid_moves', [])
            if valid_moves:
                return valid_moves[0]['token_id']
            return 0
    
    # Create game and AI model
    game = LudoGame([PlayerColor.RED, PlayerColor.BLUE])
    ai_model = MockNeuralNetwork()
    
    # Example of one complete turn
    current_player = game.get_current_player()
    dice_value = game.roll_dice()
    
    # Get complete game state for AI
    game_state = game.get_ai_decision_context(dice_value)
    
    print("Game state structure:")
    print(json.dumps(game_state, indent=2))
    
    # AI makes decision
    if game_state['valid_moves']:
        token_id = ai_model.predict(game_state)
        print(f"\nAI selected token: {token_id}")
        
        # Execute the move
        result = game.execute_move(current_player, token_id, dice_value)
        print(f"Move result: {result}")


def feature_extraction_example():
    """Example showing how to extract features for machine learning."""
    
    print("\n=== Feature Extraction Example ===")
    
    def extract_features(game_context):
        """
        Extract numerical features from game context for ML models.
        
        Returns:
            dict: Numerical features ready for ML
        """
        player_state = game_context['player_state']
        valid_moves = game_context['valid_moves']
        current_situation = game_context['current_situation']
        opponents = game_context['opponents']
        
        features = {
            # Player state features
            'tokens_in_home': player_state['tokens_in_home'],
            'active_tokens': player_state['active_tokens'], 
            'finished_tokens': player_state['finished_tokens'],
            
            # Dice and turn features
            'dice_value': current_situation['dice_value'],
            'consecutive_sixes': current_situation['consecutive_sixes'],
            'turn_count': current_situation['turn_count'],
            
            # Move opportunity features
            'num_valid_moves': len(valid_moves),
            'can_capture': any(move['captures_opponent'] for move in valid_moves),
            'can_finish': any(move['move_type'] == 'finish' for move in valid_moves),
            'can_exit_home': any(move['move_type'] == 'exit_home' for move in valid_moves),
            'has_safe_moves': any(move['is_safe_move'] for move in valid_moves),
            
            # Opponent threat features
            'opponent_threat_total': sum(opp['threat_level'] for opp in opponents),
            'opponent_finished_tokens': sum(opp['tokens_finished'] for opp in opponents),
            'opponent_active_tokens': sum(opp['tokens_active'] for opp in opponents),
            
            # Strategic features
            'max_strategic_value': max([move['strategic_value'] for move in valid_moves], default=0),
            'avg_strategic_value': sum(move['strategic_value'] for move in valid_moves) / len(valid_moves) if valid_moves else 0,
        }
        
        return features
    
    # Example usage
    game = LudoGame([PlayerColor.RED, PlayerColor.BLUE])
    dice_value = game.roll_dice()
    context = game.get_ai_decision_context(dice_value)
    
    features = extract_features(context)
    print("Extracted features for ML:")
    for feature, value in features.items():
        print(f"  {feature}: {value}")


def training_data_generation_example():
    """Example showing how to generate training data."""
    
    print("\n=== Training Data Generation Example ===")
    
    def collect_training_data(num_games=5):
        """Collect training data from multiple games."""
        training_data = []
        
        for game_num in range(num_games):
            print(f"Generating data from game {game_num + 1}...")
            
            game = LudoGame([PlayerColor.RED, PlayerColor.BLUE])
            game_data = []
            
            turn_count = 0
            while not game.game_over and turn_count < 50:  # Limit for example
                current_player = game.get_current_player()
                dice_value = game.roll_dice()
                context = game.get_ai_decision_context(dice_value)
                
                if context['valid_moves']:
                    # Record the game state and a random move (replace with actual AI decision)
                    import random
                    selected_move = random.choice(context['valid_moves'])
                    
                    # Create training example
                    training_example = {
                        'game_state': context,
                        'action_taken': selected_move['token_id'],
                        'move_details': selected_move,
                        'game_id': game_num,
                        'turn': turn_count
                    }
                    
                    game_data.append(training_example)
                    
                    # Execute the move
                    result = game.execute_move(current_player, selected_move['token_id'], dice_value)
                    
                    if not result['extra_turn']:
                        game.next_turn()
                else:
                    game.next_turn()
                
                turn_count += 1
            
            training_data.extend(game_data)
        
        return training_data
    
    # Generate some training data
    data = collect_training_data(2)  # Small example
    print(f"Generated {len(data)} training examples")
    
    # Show structure of first example
    if data:
        print("\nExample training data structure:")
        example = data[0]
        print(f"Game ID: {example['game_id']}")
        print(f"Turn: {example['turn']}")
        print(f"Action taken: {example['action_taken']}")
        print(f"Move type: {example['move_details']['move_type']}")
        print(f"Strategic value: {example['move_details']['strategic_value']}")


if __name__ == "__main__":
    # Run all examples
    random_ai_example()
    neural_network_interface_example()
    feature_extraction_example()
    training_data_generation_example()
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETED!")
    print("="*60)
    print("\nThese examples show how to:")
    print("1. Make random moves (baseline)")
    print("2. Interface with neural networks")
    print("3. Extract features for ML")
    print("4. Generate training data")
    print("\nYou can now build your own AI using these patterns!")
