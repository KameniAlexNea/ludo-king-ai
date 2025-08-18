# Ludo King AI Environment 🎲

A comprehensive Python environment for training AI agents to play Ludo King. This implementation provides a structured, observable game state perfect for machine learning applications.

## 🎯 Features

- **Complete Ludo Implementation**: Full game rules with proper token movement, capturing, and winning conditions
- **AI-Friendly Interface**: Rich game state representation optimized for ML models
- **Structured Design**: Clean, modular code that's easy to understand and extend
- **Strategic Analysis**: Built-in move evaluation and strategic context for AI decision making
- **Multiple Player Support**: 2-4 players with different colors
- **Comprehensive Documentation**: Well-documented code with clear examples

## 🏗️ Architecture

The environment is built with clear separation of concerns:

```
ludo/
├── __init__.py       # Package initialization
├── token.py          # Token/piece representation and movement
├── player.py         # Player management and strategy
├── board.py          # Game board and position management  
├── game.py           # Main game engine and rules
└── main.py           # Example usage and AI demonstrations
```

## 🚀 Quick Start

### Basic Usage

```python
from ludo import LudoGame, PlayerColor

# Create a game with 2-4 players
game = LudoGame([PlayerColor.RED, PlayerColor.BLUE])

# Roll dice and get AI decision context
dice_value = game.roll_dice()
context = game.get_ai_decision_context(dice_value)

# Your AI makes a decision
token_id = your_ai_model.predict(context)

# Execute the move
current_player = game.get_current_player()
result = game.execute_move(current_player, token_id, dice_value)

# Check results
if result['success']:
    print(f"Move successful! Extra turn: {result['extra_turn']}")
```

### AI Interface Example

```python
class YourAI:
    def make_decision(self, game_context):
        """
        game_context contains:
        - current_situation: dice value, player state
        - valid_moves: all possible moves with analysis
        - opponents: opponent states and threat levels
        - strategic_analysis: capture opportunities, safe moves, etc.
        """
        valid_moves = game_context['valid_moves']
        
        # Implement your strategy here
        # Return token_id (0-3) to move
        return best_move_token_id
```

## 🎮 Game State Representation

The environment provides rich state information perfect for AI learning:

### Game Context for AI
```python
{
    "current_situation": {
        "player_color": "red",
        "dice_value": 6,
        "consecutive_sixes": 1,
        "turn_count": 15
    },
    "player_state": {
        "tokens": [
            {
                "token_id": 0,
                "state": "active",
                "position": 23,
                "is_in_home": false,
                "is_active": true
            }
        ],
        "tokens_in_home": 2,
        "active_tokens": 2,
        "finished_tokens": 0
    },
    "valid_moves": [
        {
            "token_id": 0,
            "current_position": 23,
            "target_position": 29,
            "move_type": "advance_main_board",
            "is_safe_move": true,
            "captures_opponent": false,
            "strategic_value": 8.5
        }
    ],
    "strategic_analysis": {
        "can_capture": false,
        "can_finish_token": false,
        "can_exit_home": true,
        "safe_moves": [...],
        "risky_moves": [...]
    }
}
```

## 🎯 Game Rules Implementation

### Core Rules
- ✅ Roll 6 to exit home
- ✅ Extra turns for rolling 6, capturing, or finishing tokens
- ✅ Three consecutive 6s forfeit turn
- ✅ Token capturing and home return
- ✅ Safe squares (stars and colored squares)
- ✅ Home column exact count movement
- ✅ Stacking and blocking mechanics

### Strategic Elements
- ✅ Move type classification (exit_home, advance, finish)
- ✅ Safety analysis for each move
- ✅ Capture opportunity detection  
- ✅ Strategic value calculation
- ✅ Threat level assessment of opponents

## 🧠 AI Training Features

### State Representation
- **Observable**: Complete game state without hidden information
- **Structured**: Consistent format across all game phases
- **Rich Context**: Strategic analysis and move evaluation included
- **Scalable**: Efficient representation suitable for large-scale training

### Move Analysis
Each possible move includes:
- Current and target positions
- Move type and safety analysis
- Capture opportunities
- Strategic value scoring
- Risk assessment

### Training Support
- Game state serialization for dataset creation
- Performance benchmarking tools
- Reproducible game seeds
- Comprehensive logging

## 📊 Example AI Implementation

A simple AI is included that demonstrates the interface:

```python
class SimpleAI:
    def make_decision(self, game_context):
        valid_moves = game_context['valid_moves']
        
        # Priority system:
        # 1. Finish tokens
        # 2. Capture opponents  
        # 3. Exit home with 6
        # 4. Highest strategic value
        
        for move in valid_moves:
            if move['move_type'] == 'finish':
                return move['token_id']
        
        for move in valid_moves:
            if move['captures_opponent']:
                return move['token_id']
                
        # ... more strategy logic
        
        best_move = max(valid_moves, key=lambda m: m['strategic_value'])
        return best_move['token_id']
```

## 🏃‍♂️ Running the Demo

Run the complete demonstration:

```bash
python main.py
```

This will show:
1. **Full Game Demo**: Complete game with AI players
2. **AI Interface Demo**: Detailed state representation examples
3. **Performance Benchmark**: Speed and efficiency metrics

## 🔧 Advanced Usage

### Custom Game Rules
```python
# Modify rules by subclassing
class CustomLudoGame(LudoGame):
    def _calculate_strategic_value(self, token, dice_value):
        # Implement custom strategy evaluation
        return custom_value
```

### Training Data Generation
```python
# Generate training data
games_data = []
for _ in range(1000):
    game = LudoGame([PlayerColor.RED, PlayerColor.BLUE])
    game_history = play_complete_game(game)
    games_data.append(game_history)
```

### Model Integration
```python
import torch

class LudoAI(torch.nn.Module):
    def forward(self, game_state):
        # Process game_state features
        # Return move probabilities
        pass

# Use with the environment
model = LudoAI()
token_id = model.predict(game.get_ai_decision_context(dice_value))
```

## 🎯 Key Benefits for AI Development

1. **No Visual Processing**: Pure state-based representation
2. **Complete Information**: No hidden states or ambiguity  
3. **Rich Features**: Strategic analysis built-in
4. **Fast Execution**: Optimized for training speed
5. **Flexible**: Easy to modify rules or add features
6. **Well-Tested**: Comprehensive rule implementation

## 🤝 Contributing

This environment is designed to be easily extensible. Areas for enhancement:
- Advanced strategic analysis algorithms
- Different AI agent architectures
- Tournament and rating systems
- Rule variations and game modes
- Performance optimizations

## 📈 Roadmap

- [ ] Neural network training examples
- [ ] Reinforcement learning integration
- [ ] Tournament system
- [ ] Rule variations (team play, different board sizes)
- [ ] Advanced strategic analysis
- [ ] Performance optimizations for large-scale training

---

Perfect for chess AI developers looking to expand into board games, researchers studying game AI, or anyone building multi-agent systems! 🚀
