# Exposure Delta Reward Shaping

This document explains the exposure-based reward shaping used to encourage safer play while maintaining simplicity and computational efficiency.

## Background

We initially explored a complex potential-based R/O (Risk/Opportunity) shaping approach with multiple weighted components (progress, capture opportunity, capture risk, finish opportunity). After extensive experimentation with heuristic strategies, we found that a **simpler exposure delta approach** provides effective learning signal without the complexity and computational overhead.

## Current Approach: Exposure Delta

The key insight is: **it's not about where you end up, but whether you made yourself MORE vulnerable than before**.

### Formula

```
exposure_delta = threats_AFTER - threats_BEFORE
reward_adjustment = -exposure_delta * capture_exposure_penalty
```

- **Positive delta** (became more exposed) → penalty
- **Negative delta** (became safer) → bonus
- **Zero delta** (no change) → no adjustment

### How Threats Are Counted

For each position, we count how many dice rolls (1-6) could result in an opponent capturing the piece:

```python
def _count_threats_at_position(board, mover_color, position, opponent_positions):
    # Returns 0-6: number of dice values that would allow capture
```

Safe positions (yard, home stretch, safe squares) have 0 threats.

## Why This Works

1. **Intuitive**: A player already in danger who captures doesn't pay extra—they were already at risk. A player who leaves safety to capture pays the exposure cost.

2. **Computationally cheap**: Simple distance calculations, no recursive depth search.

3. **Discovered through heuristic analysis**: Studying cautious/defensive strategies showed that managing exposure change (not absolute exposure) correlates with winning play.

## Additional Bonuses

### Safe Landing Bonus
Small reward for landing on safe positions (safe squares, home stretch):

```python
if _is_position_safe(new_position, mover_color, board):
    mover_reward += reward_config.safe_landing_bonus
```

## Configuration

Parameters in `ludo_rl/ludo_king/config.py` (class `Reward`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `capture_exposure_penalty` | `0.2 * COEF` | Penalty multiplier for increased exposure |
| `safe_landing_bonus` | `0.02 * COEF` | Bonus for landing on safe positions |

## Sparse Event Rewards (Unchanged)

The exposure shaping is layered on top of existing sparse rewards:

| Event | Reward |
|-------|--------|
| `progress` | Small positive per move |
| `exit_home` | Bonus for leaving yard |
| `finish` | Large bonus for finishing piece |
| `capture` | Bonus for capturing opponent |
| `got_capture` | Penalty for being captured |
| `blockade` | Bonus for forming blockade |
| `hit_blockade` | Penalty for hitting blockade |
| `win` / `lose` | Terminal rewards |

## Interpreting Logged Rewards

- `~0.50`: Typical `exit_home` (0.5 with default COEF) + progress
- `-0.50`: Typical `hit_blockade` penalty
- `+5.0`: Likely a `finish` reward
- Small variations around base rewards: exposure delta adjustments

## Implementation

All reward logic is centralized in `ludo_rl/ludo_king/reward.py`:

- `compute_move_rewards()` - Main reward computation
- `compute_exposure_delta()` - Exposure change calculation
- `_count_threats_at_position()` - Threat counting helper
- `_is_position_safe()` - Safety check helper

## Testing

Functional tests in `tests/test_rewards.py` cover:

- Terminal reward scaling (win/lose for 2-4 players)
- Move event rewards (progress, exit, finish, capture, blockade)
- Exposure calculations (safe positions, threat counting, delta)
- Safe landing bonus
- Blockade mechanics
- Environment integration
- Helper functions

## Migration Notes

The previous R/O shaping approach was removed because:

1. **Complexity**: Four weighted components with depth parameter
2. **Performance**: Recursive risk calculations were slow
3. **Tuning difficulty**: Many hyperparameters to balance
4. **Marginal benefit**: Exposure delta achieved similar results with simpler code

If you have old configs with `SHAPING_*` or `RO_*` environment variables, they are no longer used.
