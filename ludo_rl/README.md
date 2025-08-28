# Ludo RL Package

Provides reinforcement learning environments and training scripts for the Ludo game.

## Structure
```
ludo_rl/
  envs/
    ludo_env.py        # Gymnasium environment wrapper
  train_sb3.py         # Stable-Baselines3 training script
  train_rllib.py       # RLlib training script
  __init__.py
```

## Environment ID
Custom environment `LudoGymEnv` (not auto-registered globally). For RLlib we register `LudoGymEnv-v0` at runtime.

## Observation (v0)
Vector combining agent & opponents token positions (normalized), finished counts, finishing feasibility flag, dice normalization, progress stats, (optional) turn fraction & blocking placeholder.

## Rewards
Modular shaping in `RewardConfig`:
- capture / got_captured
- finish_token / win / lose
- progress shaping placeholder (extend as needed)
- time penalty
- illegal action penalty
- extra turn bonus
- diversity bonus for activating new tokens

Tune these for stability.

## Usage
### Stable Baselines 3
```
python -m ludo_rl.train_sb3 --total-steps 500000 --n-envs 8
```
Checkpoints & logs under `models/` and `logs/`.

### RLlib
```
python -m ludo_rl.train_rllib --stop-timesteps 500000
```

## Extension Ideas
- Action masking (invalid moves) via custom policy
- Richer observation (threat distances, safe squares, home column depth)
- Self-play population: swap opponent strategies with latest policy snapshot
- Curriculum: progressively harder opponent mixes
- Eval harness comparing against built-in heuristic strategies

## License
See project root LICENSE.
