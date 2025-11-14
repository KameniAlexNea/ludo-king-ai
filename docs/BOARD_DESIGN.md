# Ludo Token-Sequence Observation Design

## Overview

The Ludo board state is now represented as a **temporal token-sequence observation** that tracks the last 10 atomic moves in the game. This design captures the dynamic evolution of piece positions over time rather than a static spatial representation.

## Observation Space Structure

The observation is a dictionary with the following components:

### Core Components

- **`positions`**: `(10, 16)` int64 array - Last 10 frames of all 16 token positions (0..57)
- **`dice_history`**: `(10,)` int64 array - Dice rolls for each of the last 10 frames (0 for padding, 1..6 actual)
- **`token_mask`**: `(10, 16)` bool array - Validity mask (1 if frame/token exists, 0 for padding)
- **`token_colors`**: `(16,)` int64 array - Color ID for each token block (0..3)
- **`current_dice`**: `(1,)` int64 array - Current dice roll for agent's decision (1..6)

### Token Organization

The 16 tokens are organized in color blocks:

- **Tokens 0-3**: Agent's pieces (red player)
- **Tokens 4-7**: Opponent 1's pieces (next player)
- **Tokens 8-11**: Opponent 2's pieces (player across)
- **Tokens 12-15**: Opponent 3's pieces (previous player)

Each token position uses the same mapping as before:

- **Position 0**: Yard (starting area)
- **Positions 1-51**: Main circular track
- **Positions 52-56**: Home column (final approach)
- **Position 57**: Finish (goal)

## Temporal Design

### History Buffer

The simulator maintains a circular buffer of the last 10 atomic moves:

```python
class Simulator:
    history_T: int = 10
    _pos_hist: np.ndarray  # (10, 16) - position history
    _dice_hist: np.ndarray # (10,) - dice history
    _mask_hist: np.ndarray # (10, 16) - validity masks
    _hist_len: int = 0    # current history length
    _hist_ptr: int = 0    # circular buffer pointer
```

### Frame Updates

Each atomic move (piece movement) appends a new frame to the history:

```python
def _append_history(self, dice: int) -> None:
    # Capture current board state for agent
    frame_pos = self.game.board.all_token_positions(agent_color)
    i = self._hist_ptr
    self._pos_hist[i, :] = frame_pos
    self._dice_hist[i] = int(dice)
    self._mask_hist[i, :] = self._token_exists_mask
    # Advance circular buffer
    self._hist_ptr = (self._hist_ptr + 1) % self.history_T
    self._hist_len = min(self._hist_len + 1, self.history_T)
```

### Observation Construction

The observation returns the most recent frames in chronological order:

```python
def get_token_sequence_observation(self, current_dice: int) -> dict:
    # Return last T frames, oldest first, with padding for early game
    # Handles circular buffer wraparound automatically
    return {
        "positions": out_pos,      # (10, 16)
        "dice_history": out_dice,  # (10,)
        "token_mask": out_mask,    # (10, 16)
        "token_colors": self._token_colors,  # (16,)
        "current_dice": np.asarray([current_dice], dtype=np.int64),  # (1,)
    }
```

## Implementation Details

### Simulator Class (`ludo_rl/ludo_king/simulator.py`)

The simulator maintains the token-sequence history buffer:

```python
@dataclass(slots=True)
class Simulator:
    # Token-sequence observation buffers
    history_T: int = 10
    _pos_hist: np.ndarray = field(default=None, init=False, repr=False)  # (10, 16)
    _dice_hist: np.ndarray = field(default=None, init=False, repr=False) # (10,)
    _mask_hist: np.ndarray = field(default=None, init=False, repr=False) # (10, 16)
    _token_colors: np.ndarray = field(default=None, init=False, repr=False) # (16,)
    _hist_len: int = field(default=0, init=False, repr=False)
    _hist_ptr: int = field(default=0, init=False, repr=False)

    @classmethod
    def for_game(cls, game: Game, agent_index: int = 0) -> "Simulator":
        """Create simulator with initialized history buffers."""
        obj = object.__new__(cls)
        obj.agent_index = agent_index
        obj.game = game
        obj.history_T = 10
        agent_color = int(game.players[agent_index].color)
        obj._pos_hist = np.zeros((10, 16), dtype=np.int64)
        obj._dice_hist = np.zeros((10,), dtype=np.int64)
        obj._mask_hist = np.zeros((10, 16), dtype=np.bool_)
        obj._token_colors = game.board.token_colors(agent_color)
        obj._token_exists_mask = game.board.token_exists_mask(agent_color)
        obj._hist_len = 0
        obj._hist_ptr = 0
        return obj
```

### Environment Integration (`ludo_rl/ludo_env.py`)

The environment builds observations from the simulator:

```python
class LudoEnv(gym.Env):
    observation_space = spaces.Dict({
        "positions": spaces.Box(low=0, high=57, shape=(10, 16), dtype=np.int64),
        "dice_history": spaces.Box(low=0, high=6, shape=(10,), dtype=np.int64),
        "token_mask": spaces.Box(low=0, high=1, shape=(10, 16), dtype=np.bool_),
        "token_colors": spaces.Box(low=0, high=3, shape=(16,), dtype=np.int64),
        "current_dice": spaces.Box(low=1, high=6, shape=(1,), dtype=np.int64),
    })

    def _build_observation(self) -> Dict[str, np.ndarray]:
        """Build token-sequence observation from simulator."""
        return self.sim.get_token_sequence_observation(self.current_dice_roll)
```

### Board Class Support (`ludo_rl/ludo_king/board.py`)

The board class provides token position utilities:

```python
class Board:
    def all_token_positions(self, agent_color: int) -> np.ndarray:
        """Return (16,) array of all token positions in agent-relative frame."""
        # Returns positions for all 16 tokens (4 per player)
        
    def token_colors(self, agent_color: int) -> np.ndarray:
        """Return (16,) array of color IDs for each token."""
        # Returns color mapping for all tokens
        
    def token_exists_mask(self, agent_color: int) -> np.ndarray:
        """Return (16,) boolean mask of which tokens exist."""
        # Returns True for tokens that are present in the game
```

## Usage Example

```python
from ludo_rl.ludo_env import LudoEnv

# Create environment
env = LudoEnv()

# Reset to get initial observation
obs, info = env.reset()

# Observation structure
print(obs["positions"].shape)      # (10, 16) - last 10 frames of 16 token positions
print(obs["dice_history"].shape)   # (10,) - dice for last 10 frames
print(obs["token_mask"].shape)     # (10, 16) - validity mask
print(obs["token_colors"].shape)   # (16,) - color ID per token
print(obs["current_dice"].shape)   # (1,) - current dice roll

# Access specific data
last_frame_positions = obs["positions"][-1]  # Most recent positions
my_pieces = last_frame_positions[0:4]        # Agent's 4 pieces
opp1_pieces = last_frame_positions[4:8]      # Opponent 1's pieces
current_dice = obs["current_dice"][0]        # Current dice value

# History tracking
for frame_idx in range(10):
    if obs["token_mask"][frame_idx].any():
        frame_positions = obs["positions"][frame_idx]
        frame_dice = obs["dice_history"][frame_idx]
        print(f"Frame {frame_idx}: dice={frame_dice}, positions={frame_positions}")
```

## Benefits

This token-sequence design provides:

1. **Temporal dynamics**: Captures movement patterns over the last 10 moves
2. **Complete state history**: Agent can see how the game evolved
3. **Efficient representation**: Only tracks actual token positions, not full board
4. **Padding support**: Early-game frames are zero-masked for consistent shape
5. **Agent-centric view**: All positions relative to agent's perspective
6. **Action context**: Current dice helps inform valid moves

The temporal sequence is particularly valuable for reinforcement learning with Transformers, as the model can:

- Learn movement patterns and strategies over time
- Understand opponent behavior from their recent moves
- Make decisions based on game momentum and flow
- Capture long-term dependencies between moves
