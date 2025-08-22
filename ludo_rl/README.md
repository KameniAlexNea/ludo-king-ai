# Improved Ludo Reinforcement Learning

This directory contains an enhanced reinforcement learning implementation for Ludo King AI with significant improvements over the original system.

## Major Improvements

### 1. Enhanced State Representation
- **Compact State Encoding**: Reduced from variable-length to fixed 64-feature representation
- **Strategic Features**: Token safety, relative positioning, game phase awareness
- **Better Normalization**: All features properly normalized for neural network training
- **Contextual Information**: Opponent threat levels, move diversity, tactical opportunities

### 2. Advanced DQN Architecture
- **Dueling DQN**: Separate value and advantage streams for better Q-value estimation
- **Double DQN**: Reduces overestimation bias in Q-learning
- **Prioritized Experience Replay**: Learns from important experiences more frequently
- **Layer Normalization**: Improves training stability
- **Gradient Clipping**: Prevents exploding gradients

### 3. Improved Reward Engineering
- **Balanced Rewards**: Better scaling between different reward components
- **Progress-based Rewards**: Encourages token advancement
- **Safety Considerations**: Rewards safe moves, penalizes unnecessary risks
- **Game Phase Awareness**: Different rewards for early/mid/late game
- **Relative Positioning**: Rewards based on advantage over opponents

### 4. Enhanced Training Pipeline
- **Better Sequence Detection**: Improved game boundary identification
- **Validation Split**: Built-in train/validation splitting
- **Early Stopping**: Prevents overfitting
- **Comprehensive Metrics**: Tracks loss, reward, and accuracy
- **Model Checkpointing**: Saves best models automatically

### 5. Model Validation and Interpretation
- **Expert Move Validation**: Compare against strategic move choices
- **Decision Pattern Analysis**: Understand model behavior
- **Comprehensive Reporting**: Detailed performance analysis
- **Model Comparison**: Compare multiple trained models
- **Visualization Tools**: Plot training progress and model behavior

## Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy matplotlib
```

### 2. Generate Training Data

Run tournaments to collect game data:

```bash
python four_player_tournament.py
```

This saves game decisions to the `saved_states/` directory.

### 3. Train an RL Agent

```bash
python examples/train_rl_agent.py
```

This will:
- Load saved game data
- Train a DQN agent
- Save the trained model to `models/ludo_dqn_model.pth`
- Generate training progress plots

### 4. Use the Trained Agent

```python
from ludo_rl import create_rl_strategy

# Create RL strategy
rl_strategy = create_rl_strategy("models/ludo_dqn_model.pth")

# Use in games
player.set_strategy(rl_strategy)
```

## Architecture

### State Representation

The state encoder converts game contexts into fixed-length numerical vectors containing:

- **Token Positions** (864 dims): One-hot encoding of all tokens' positions
- **Game Context** (4 dims): Dice value, consecutive sixes, turn count, current player
- **Player Statistics** (16 dims): Tokens in home/active/finished/won for each player  
- **Valid Moves** (24 dims): Features for up to 4 possible moves

**Total State Dimension**: 908

### Neural Network

The DQN uses a feed-forward architecture:

```
Input (908) → Linear(512) → ReLU → Dropout(0.2) →
Linear(512) → ReLU → Dropout(0.2) →  
Linear(256) → ReLU → Linear(4)
```

Output represents Q-values for each token (0-3).

### Training Process

1. **Data Loading**: Load saved game decisions from JSON files
2. **State Encoding**: Convert game contexts to numerical vectors
3. **Reward Calculation**: Assign rewards based on outcomes and strategic value
4. **Experience Replay**: Train on mini-batches of (state, action, reward, next_state)
5. **Target Network**: Stabilize training with periodic target network updates

## Reward Function

The reward function combines multiple factors:

- **Basic Success**: +1 for valid moves, -2 for invalid moves
- **Captures**: +10 per captured opponent token
- **Token Finishing**: +25 for getting a token home
- **Extra Turns**: +3 for earning extra turns
- **Strategic Value**: +0.1 × strategic_value from game analysis
- **Optimal Moves**: +5 for choosing the best strategic move
- **Game Winning**: +100 for winning the game
- **Progress**: +0.1 × (new_position - old_position) for advancement

## Usage Examples

### Basic Training

```python
from ludo_rl import LudoRLTrainer

# Initialize trainer with saved game data
trainer = LudoRLTrainer(state_saver_dir="saved_states")

# Train for 1000 epochs
stats = trainer.train(epochs=1000, model_save_path="my_model.pth")

# Plot training progress
trainer.plot_training_progress("training_plots.png")
```

### Using Trained Agent

```python
from ludo_rl import create_rl_strategy
from ludo import LudoGame, PlayerColor

# Load trained strategy
rl_strategy = create_rl_strategy("my_model.pth", name="MyRL")

# Create game and set strategy
game = LudoGame([PlayerColor.RED, PlayerColor.GREEN])
game.players[0].set_strategy(rl_strategy)

# Play game normally
while not game.game_over:
    dice_value = game.roll_dice()
    game.play_turn()
```

### Custom Training Configuration

```python
from ludo_rl import LudoStateEncoder, LudoDQNAgent, LudoRLTrainer

# Custom state encoder
encoder = LudoStateEncoder(board_size=52, max_tokens=4, num_players=4)

# Custom DQN agent
agent = LudoDQNAgent(
    state_dim=encoder.state_dim,
    lr=0.001,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    memory_size=20000,
    batch_size=64
)

# Custom trainer
trainer = LudoRLTrainer("my_game_data.json")
trainer.agent = agent

# Train with custom parameters
stats = trainer.train(
    epochs=2000,
    target_update_freq=50,
    save_freq=100
)
```

## File Structure

```
ludo_rl/
├── __init__.py           # Module exports
├── state_encoder.py      # Game state to vector conversion
├── dqn_model.py          # DQN architecture and agent
├── trainer.py            # Training pipeline
└── rl_player.py          # Game system integration

examples/
├── train_rl_agent.py     # Complete training example
└── play_with_rl_agent.py # Gameplay with trained agent
```

## Integration with Existing Code

The RL module is designed to work seamlessly with the existing Ludo codebase:

- Uses existing `GameStateSaver` for data collection
- Compatible with existing strategy system
- Leverages existing game context generation
- No modifications needed to core game logic

## Performance Tips

1. **Data Collection**: Collect diverse game data from multiple strategies
2. **Training Duration**: Start with 1000-2000 epochs, increase if needed
3. **Hyperparameters**: Adjust learning rate, batch size, and network size based on data size
4. **Evaluation**: Use the evaluation methods to assess agent performance
5. **Model Comparison**: Train multiple models with different configurations

## Troubleshooting

### Common Issues

**No training data found**:
- Run tournaments first to generate saved game states
- Check that `saved_states/` directory contains JSON files

**Poor performance**:
- Collect more diverse training data
- Increase training epochs
- Adjust reward function weights
- Try different network architectures

**Training instability**:
- Reduce learning rate
- Increase target network update frequency
- Add gradient clipping (already implemented)

### Dependencies

- **torch**: PyTorch for neural networks
- **numpy**: Numerical computations
- **matplotlib**: Training progress visualization
- **json**: Data loading (built-in)

## Future Enhancements

Potential improvements to explore:

- **Policy Gradient Methods**: PPO, A3C for better policy learning
- **Multi-Agent Training**: Train multiple agents simultaneously
- **Curriculum Learning**: Progressive difficulty increase
- **Self-Play**: Agent training against itself
- **Transfer Learning**: Pre-train on one game variant, fine-tune on another
- **Attention Mechanisms**: Better handling of variable-length valid moves
- **Opponent Modeling**: Explicit modeling of opponent strategies