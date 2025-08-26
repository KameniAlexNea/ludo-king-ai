# Ludo King AI Environment 🎲

A comprehensive Python environment for training AI agents to play Ludo King. This implementation provides a structured, observable game state perfect for machine learning applications.

## 🎯 Features

- **Complete Ludo Implementation**: Full game rules with proper token movement, capturing, and winning conditions
- **AI-Friendly Interface**: Rich game state representation optimized for ML models
- **Strategic Tournament System**: Multi-strategy tournaments with comprehensive analysis
- **Game State Learning**: Automatic saving and analysis of AI decisions and outcomes
- **Move Evaluation Framework**: Real-time move quality assessment and correlation analysis
- **Structured Design**: Clean, modular code that's easy to understand and extend
- **Strategic Analysis**: Built-in move evaluation and strategic context for AI decision making
- **Multiple Player Support**: 2-4 players with different colors
- **Comprehensive Documentation**: Well-documented code with clear examples

## 🏗️ Architecture

The environment is built with clear separation of concerns:

```
ludo/
├── __init__.py              # Package initialization
├── token.py                 # Token/piece representation and movement
├── player.py                # Player management and strategy
├── board.py                 # Game board and position management  
├── game.py                  # Main game engine and rules
├── strategy.py              # Strategy framework and implementations
└── strategies/              # Individual strategy implementations
    ├── base.py              # Base strategy class
    ├── killer.py            # Aggressive capture-focused strategy
    ├── winner.py            # Win-focused finishing strategy
    ├── defensive.py         # Conservative safe-play strategy
    ├── optimist.py          # Risk-taking strategy
    ├── balanced.py          # Adaptive balanced strategy
    └── cautious.py          # Ultra-conservative strategy

# Tournament and Analysis Systems
four_player_tournament.py   # Comprehensive tournament system
strategic_tournament.py     # Strategic analysis demonstrations
game_state_saver.py         # Decision tracking and persistence
real_move_evaluator.py      # Move quality analysis system
main.py                     # Example usage and AI demonstrations
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
- **Reinforcement Learning Ready**: Complete RL training pipeline with DQN implementation

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

## 🤖 Reinforcement Learning Training

The environment includes a complete RL training framework using Deep Q-Networks (DQN):

### Quick RL Training
```python
# 1. Generate training data
python four_player_tournament.py

# 2. Train RL agent
python examples/train_rl_agent.py

# 3. Use trained agent
from ludo_rl import create_rl_strategy
rl_strategy = create_rl_strategy("models/ludo_dqn_model.pth")
```

### RL Features
- **State Encoding**: Converts game states to 908-dimensional vectors
- **DQN Architecture**: Feed-forward network with experience replay
- **Reward Engineering**: Multi-factor reward function including strategic value
- **Training Pipeline**: Automated training on saved game data
- **Game Integration**: Seamless integration with existing strategy system

### Training Data Format
Uses existing `GameStateSaver` output:
```json
{
  "timestamp": "2025-01-21T10:00:00.000000",
  "strategy": "killer",
  "game_context": { ... },
  "chosen_move": 1,
  "outcome": { ... }
}
```

See `ludo_rl/README.md` for detailed documentation.

## 📊 Game State Learning System

### Automatic Decision Tracking
The environment automatically captures every AI decision for later analysis:

```python
from game_state_saver import GameStateSaver
from four_player_tournament import FourPlayerTournament

# Run tournament with automatic state saving
tournament = FourPlayerTournament()
tournament.run_full_tournament()  # Saves all decisions to JSON files

# Load and analyze saved states
saver = GameStateSaver()
states = saver.load_states("killer")  # Load decisions for specific strategy
analysis = saver.analyze_strategy("killer")  # Get performance statistics
```

### Rich Context Capture
Each saved decision includes:
- **Complete game context**: Board state, valid moves, strategic analysis
- **Decision made**: Which token was chosen and why
- **Outcome**: Results of the move (captures, extra turns, game won)
- **Timestamp**: For temporal analysis

### Move Quality Evaluation
```python
from real_move_evaluator import GameStateAnalyzer

# Analyze move quality vs outcomes
analyzer = GameStateAnalyzer()
analyzer.analyze_decisions_with_context("optimist")

# Results include:
# - Safety vs Progress vs Capture scores
# - Correlation between move quality and outcomes  
# - Move type performance analysis
# - Strategic pattern identification
```

### Performance Insights
The evaluation system provides:
- **Move Type Analysis**: Exit home vs advance vs finish performance
- **Safety Assessment**: Safe moves vs risky moves outcomes
- **Strategy Comparison**: Which strategies make better decisions
- **Correlation Validation**: Does good evaluation predict good outcomes?

## 🏆 Tournament Analysis

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

### Basic Game Demo
Run the complete demonstration:

```bash
python main.py
```

This will show:
1. **Full Game Demo**: Complete game with AI players
2. **AI Interface Demo**: Detailed state representation examples
3. **Performance Benchmark**: Speed and efficiency metrics

### Strategic Tournament
Run comprehensive multi-strategy tournaments:

```bash
python four_player_tournament.py
```

Features:
- **35 strategy combinations**: Tests all 4-player combinations
- **Performance analysis**: Win rates, capture efficiency, strategic behavior
- **Automatic state saving**: All decisions saved for later analysis
- **Head-to-head comparison**: Strategy vs strategy performance matrices

### Move Evaluation Analysis
Analyze saved tournament data:

```bash
python real_move_evaluator.py
```

Provides:
- **Move quality scoring**: Safety, progress, capture, blocking analysis
- **Strategy comparison**: Which strategies make better decisions
- **Correlation analysis**: Move quality vs actual outcomes
- **Performance validation**: Evaluation system accuracy assessment

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

- [x] **Strategic tournament system** - Multi-strategy competitions with comprehensive analysis
- [x] **Game state learning** - Automatic decision tracking and analysis
- [x] **Move evaluation framework** - Real-time move quality assessment
- [x] **Performance correlation analysis** - Move quality vs outcome validation
- [x] **Reinforcement learning integration** - Complete RL agent training framework with DQN
- [ ] **Neural network training examples** - Deep learning integration examples beyond RL
- [ ] **Advanced move evaluation** - True board position analysis (not meta-evaluation)
- [ ] **Predictive modeling** - Outcome prediction from game states
- [ ] **Rule variations** - Team play, different board sizes, custom rules
- [ ] **Real-time strategy adaptation** - Dynamic strategy switching
- [ ] **Performance optimizations** - Large-scale training efficiency improvements

---

Perfect for chess AI developers looking to expand into board games, researchers studying game AI, or anyone building multi-agent systems! 🚀
