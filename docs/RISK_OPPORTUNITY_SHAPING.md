# Risk/Opportunity (RO) Shaping

This document explains the potential-based shaping we add on top of sparse event rewards to provide dense learning signal that encourages safer, more opportunistic play.

## Summary
- Reward used by the agent for learning:
  - R_total = R_sparse + alpha * (gamma * Phi(s') - Phi(s))
- Phi(s) is a potential function combining risk and opportunity signals from the current state.
- R_sparse remains the canonical event rewards (win/lose, finish, capture, blockade, exit home, progress).
- Shaping preserves optimal policies (potential-based) while providing a smoother gradient during training.

## Components of Phi(s)
Phi(s) is computed per agent from four normalized signals in [0, 1], combined with configurable weights. We clip the final Phi to [-1, 1] for stability.

1) Progress (ro_w_progress)
- Normalized piece progress toward finish: average over the 4 pieces of position/57 (capped).

2) Capture Opportunity (next move) (ro_w_cap_opp)
- For each of the agent's tokens on the ring, we count dice in {1..6} that would land on an opponent (excluding global safe squares), and average across tokens.

3) Capture Risk (within depth) (ro_w_cap_risk)
- For each of the agent's tokens on the ring (and not on a global safe square), we compute the set of dice in {1..6} that would allow any opponent to land exactly on that token next move (p1).
- We convert to a depth-k approximation: 1 - (1 - p1)^depth, then average over tokens.

4) Finish Opportunity (next move) (ro_w_finish_opp)
- For each token in the home column (52..56), if the exact dice to finish exists in {1..6}, add 1/6; average across the 4 pieces.

Phi(s) = ro_w_progress * progress
        + ro_w_cap_opp * cap_opp
        - ro_w_cap_risk * cap_risk
        + ro_w_finish_opp * finish_opp

## Shaping Delta
Given Phi(s) and Phi(s'):
- shaping_delta = gamma * Phi(s') - Phi(s)
- We add alpha * shaping_delta to the mover's sparse reward.
- For a blocked move (no state change), shaping_delta simplifies to (gamma - 1) * Phi(s), typically negative (a gentle discouragement).

## Configuration
All parameters are in `ludo_rl/ludo_king/config.py` (class Reward) and can be overridden via environment variables:

- SHAPING_USE (default 1): 1 = on, 0 = off
- SHAPING_ALPHA (default 0.5): scales shaping contribution
- SHAPING_GAMMA (default 0.99): discount for potential
- RO_DEPTH (default 2): depth for risk approximation
- RO_W_PROGRESS (default 0.3)
- RO_W_CAP_OPP (default 0.4)
- RO_W_CAP_RISK (default 0.6)
- RO_W_FINISH_OPP (default 0.3)

Example:

```bash
export SHAPING_USE=1
export SHAPING_ALPHA=1.0
export SHAPING_GAMMA=0.99
export RO_DEPTH=3
export RO_W_CAP_RISK=1.2
export RO_W_CAP_OPP=0.8
export RO_W_PROGRESS=0.2
export RO_W_FINISH_OPP=0.5
```

## Practical Guidance
- Training impact: RO shaping primarily affects training. A fixed model's actions in simulation won’t change from RO alone; you’ll see RO in per-step rewards/logs.
- Start conservative: alpha ∈ [0.3, 1.0], depth ∈ [1, 3].
- Anneal alpha down late in training to let sparse events dominate.
- Watch for perverse incentives:
  - Add time/skip penalties (already present) to discourage stalling.
  - Keep `ro_w_cap_risk` reasonably strong so risky exposure is penalized.

## Interpreting Logged Rewards
- ~0.50 often corresponds to `exit_home` (0.5 with default COEF) plus tiny progress.
- -0.50 is typically `hit_blockade` (defaults to -0.5 * COEF).
- Large +5.0 is likely a `finish` (+5 with default COEF).
- RO contributions are added on top; increasing SHAPING_ALPHA and RO weights will make them visible in logs.

## Limitations
- Risk computation is analytic and shallow for speed; RO_DEPTH scales the "compounded" risk probabilistically as 1 - (1 - p1)^depth (an approximation).
- Only global safe squares are excluded from capture events; local/special rules are already enforced by the engine.

## A/B Experiments
- Baseline: `SHAPING_USE=0`
- RO on: `SHAPING_USE=1`, bump `SHAPING_ALPHA` and tune RO weights.
- Compare rollouts or reward traces from `tools/simulate.py`.

## Implementation Pointers
- Potential and shaping are implemented in `ludo_rl/ludo_king/reward.py`:
  - `compute_state_potential(game, agent_index, depth)`
  - `shaping_delta(phi_before, phi_after, gamma)`
- Integration occurs in `ludo_rl/ludo_king/game.py` inside `apply_move`.
- Env and UI consume `MoveResult.rewards` directly; no local reward math.

## Testing
- Functional tests for rewards live under `tests/`:
  - `test_rewards.py` covers terminal scaling, event rewards, invalid action, and helpers.
  - `test_blockade_rewards.py` covers blockade scenarios and penalties.
- Consider adding A/B shape delta assertions for synthetic risky boards if you enable a debug flag.
