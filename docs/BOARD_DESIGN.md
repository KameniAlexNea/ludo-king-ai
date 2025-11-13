# Ludo Board Tensor Design

## Overview

The Ludo board state is represented as a **10-channel tensor** with shape `(10, 58)`, where:
- **10 channels** represent different aspects of the game state
- **58 positions** represent the complete path: yard (0), main track (1-51), home column (52-56), and finish (57)

## Channel Layout

### Channels 0-3: Piece Positions (Static)
These channels track the current positions of all pieces on the board.

- **Channel 0**: Agent's pieces (my tokens)
- **Channel 1**: Opponent 1's pieces (next player)
- **Channel 2**: Opponent 2's pieces (player across)
- **Channel 3**: Opponent 3's pieces (previous player)

Each position contains a count of how many pieces are at that location.

### Channel 4: Safe Zones (Fixed)
This channel marks positions where pieces are safe from capture:
- Home column positions (52-56): Always safe
- Safe squares on the main track (1, 9, 14, 22, 27, 35, 40, 48): Safe for all players
- Values: 1.0 for safe positions, 0.0 otherwise

### Channels 5-9: Transition Summaries (Dynamic)
These channels track what happened between the agent's previous turn and current turn. They **accumulate all activity** from when the agent last interacted with the board until its next turn, including:
- When the agent makes a move and opponents respond
- When the agent has no valid moves and opponents play multiple rounds

The summaries are reset only when the agent successfully takes an action.

#### Channel 5: Movement Heatmap
Tracks all movements (by agent and opponents) since the last agent turn:
- Increments by 1.0 for each piece movement to a position
- Shows which parts of the board have seen activity

#### Channel 6: My Knockouts
Marks positions where the agent knocked out opponent pieces:
- Set to 1.0 at positions where the agent captured an opponent
- Helps the agent learn offensive strategies

#### Channel 7: Opponent Knockouts
Marks positions where opponents knocked out the agent's pieces:
- Set to 1.0 at positions where the agent was captured
- Helps the agent learn defensive awareness

#### Channel 8: New Blockades
Marks positions where blockades (2 pieces of the same color) were formed:
- Set to 1.0 at positions where blockades were created
- Helps identify strategic obstacles

#### Channel 9: Reward Heatmap
Accumulates rewards at each position:
- Adds the reward value for each move to that position
- Shows which positions generated positive or negative outcomes

## Position Mapping

All positions are represented in the **agent-relative frame**:

- **Position 0**: Yard (starting area)
- **Positions 1-51**: Main circular track (relative to agent's color)
- **Positions 52-56**: Home column (final approach)
- **Position 57**: Finish (goal)

### Coordinate Transformation

Opponent piece positions are automatically translated from their perspective to the agent's perspective, ensuring the agent always sees the board from its own viewpoint.

## Implementation Details

### Board Class (`ludo_rl/ludo_king/board.py`)

```python
class Board:
    # Transition summary tracking (channels 5-9)
    movement_heatmap: np.ndarray      # Channel 5
    my_knockouts: np.ndarray          # Channel 6
    opp_knockouts: np.ndarray         # Channel 7
    new_blockades: np.ndarray         # Channel 8
    reward_heatmap: np.ndarray        # Channel 9
    
    def build_tensor(self, agent_color: int) -> np.ndarray:
        """Build a (10, PATH_LENGTH) tensor representing the full board state."""
        # Returns (10, 58) tensor with all channels populated
        
    def reset_transition_summaries(self) -> None:
        """Reset channels 5-9 to zero for a new turn cycle."""
```

### Simulator Class (`ludo_rl/ludo_king/simulator.py`)

The simulator manages the transition summaries with careful control over when to reset:

```python
class Simulator:
    def step(self, agent_move: Move) -> tuple[bool, bool]:
        # Reset transition summaries at start of agent's turn
        self.game.board.reset_transition_summaries()
        
        # Apply agent's move and update summaries
        result = self.game.apply_move(agent_move)
        self._update_transition_summaries(self.agent_index, agent_move, result)
        
        # Simulate opponents if no extra turn
        if not extra_turn:
            # Update summaries for each opponent move
            ...
    
    def step_opponents_only(self, reset_summaries: bool = True) -> None:
        # Control whether to reset summaries
        # Set reset_summaries=False to accumulate across multiple opponent rounds
        if reset_summaries:
            self.game.board.reset_transition_summaries()
        # Simulate all opponents...
```

### Environment Usage (`ludo_rl/ludo_env.py`)

The environment correctly manages summary accumulation:

```python
# When agent successfully takes action:
self.sim.step_opponents_only(reset_summaries=True)  # Reset for fresh cycle

# When agent has no valid moves (in while loop):
self.sim.step_opponents_only(reset_summaries=False)  # Accumulate activity
```

## Usage Example

```python
from ludo_rl.ludo_king.game import Game
from ludo_rl.ludo_king.player import Player
from ludo_rl.ludo_king.types import Color

# Create game
players = [Player(Color.RED), Player(Color.GREEN), 
           Player(Color.YELLOW), Player(Color.BLUE)]
game = Game(players=players)

# Get board tensor for agent (Red player)
board_tensor = game.board.build_tensor(agent_color=0)

# Shape: (10, 58)
print(board_tensor.shape)  # (10, 58)

# Access specific channels
my_pieces = board_tensor[0]           # My piece positions
opponent_pieces = board_tensor[1:4]   # Opponent positions
safe_zones = board_tensor[4]          # Safe positions
movement_heat = board_tensor[5]       # Recent movements
my_knockouts = board_tensor[6]        # My captures
opp_knockouts = board_tensor[7]       # Captures against me
blockades = board_tensor[8]           # New blockades
rewards = board_tensor[9]             # Reward distribution
```

## Benefits

This 10-channel design provides:

1. **Complete state information**: Current positions of all pieces
2. **Strategic context**: Safe zones for planning
3. **Historical awareness**: What happened since last turn
4. **Reward shaping**: Direct feedback on move quality
5. **Temporal dynamics**: Movement patterns and interactions
6. **Agent-centric view**: Everything relative to agent's perspective

The transition summaries (channels 5-9) are particularly valuable for reinforcement learning, as they provide the agent with immediate feedback about the consequences of actions and help it learn patterns of successful play.
