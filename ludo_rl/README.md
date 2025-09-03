# Ludo RL (Classic Multi-Seat Framework)

This module (`ludo_rl`) provides the original multi-seat Ludo reinforcement learning environment and training scripts. A single shared policy can control every color sequentially. For the newer single-seat fast-forward environment see `../ludo_rls`.

## Why This Exists
The classic design exposes every player turn to the learner. It is useful for:
* Studying symmetric self-play dynamics
* Debugging reward shaping across all roles
* Backward compatibility with older logs & models

Trade‑offs: credit assignment noise (capture vs got_captured), longer episodes (~4x agent steps), slower convergence vs single-seat.

## Directory Layout
```
ludo_rl/
  envs/
    ludo_env.py                    # Core Gymnasium environment (all seats exposed)
    model.py                       # Env + reward dataclasses (RewardConfig, ObservationConfig, OpponentsConfig)
    builders/observation_builder.py# Feature vector assembly
    calculators/
      simple_reward_calculator.py  # Deterministic event+progress shaping
      reward_calculator.py         # (Alternate / legacy shaping logic)
      probabilistic_calculator.py  # Optional stochastic reward modulation
    simulators/opponent_simulator.py # Drives opponent turns between agent moves
    utils/move_utils.py            # Dice rolls, valid move logic helpers
  ppo_strategy.py                  # Wrapper to deploy a trained PPO as a heuristic strategy
  train_sb3.py                     # Stable-Baselines3 PPO training script
  train_rllib.py                   # RLlib training script
  README.md
  __init__.py
```

## Environment: `LudoGymEnv`
Implements Gymnasium API: `reset(seed=...) -> (obs, info)` and `step(action) -> (obs, reward, terminated, truncated, info)`.

Turn mechanics:
1. One agent (configured `agent_color`, default red) acts per `step()`.
2. If no extra turn, all opponents simulate (including their chained extra turns) via `OpponentSimulator`.
3. Episode continues until someone wins (`terminated`) or `max_turns` exceeded (`truncated`).

### Opponent Strategies
Three distinct opponent strategies are sampled each reset from `EnvConfig.opponents.candidates` (e.g. killer, winner, optimist, balanced, defensive, random, cautious). They are assigned to the non-agent colors in fixed order.

### Observation Vector (default schema)
Includes (normalized):
* Agent token positions (4)
* Aggregated agent token state counts (active/home/home_column/finished) if implemented in builder
* Opponent token positions (3 * 4)
* Finished token counts per player (4)
* Dice value (scaled)
* Can any agent token finish this turn flag
* Agent progress fraction & opponent mean progress
* Optional turn index fraction, blocking placeholder

### Action Space
`Discrete(4)` (choose token). Invalid actions incur `illegal_action` penalty; environment still advances using a fallback valid token to avoid deadlocks.

## Reward System
Two layers:
* `SimpleRewardConstants` (fixed large-magnitude event constants used inside simple calculator)
* `RewardConfig` (tunable shaping values for diversity, illegal penalties, etc.)

Event components (simple calculator):
* Movement progress (scaled, home column multiplier, safe square bonus)
* Capture (+10 each by default constant)
* Got captured (-6 each)
* Token finished (derived from progress constant, e.g. *10)
* Extra turn (+2)
* Diversity bonus (first activation of each token) from `reward_cfg`
* Illegal action penalty (`reward_cfg.illegal_action`)

Terminal:
* Win: +200
* Loss: -200 (if any opponent wins first)

Optional modules: `probabilistic_calculator.py` can modulate risk-based shaping if enabled (`reward_cfg.use_probabilistic_rewards`).

## Configuration (`EnvConfig`)
Key fields:
* `max_turns`: truncation limit (default 1000 in model, script can override)
* `agent_color`: training seat (defaults red; PPO strategy wrapper expects this)
* `reward_cfg`: `RewardConfig` instance (win/lose, capture, progress scaling, penalties)
* `obs_cfg`: toggles for turn index, blocking count, raw dice
* `opponents.candidates`: list of heuristic strategy names used for sampling
* `seed`: base RNG seed

## Training (Stable-Baselines3 PPO)
Script: `train_sb3.py` features:
* Vectorized env support (`--n-envs` -> SubprocVecEnv if >1)
* Checkpoints every `--checkpoint-freq` (saves model, replay buffer, VecNormalize stats)
* Evaluation callback every `--eval-freq` using separate VecNormalize wrapper
* CLI adjustable entropy coefficient (`--ent-coef`)
* Optional disabling of probabilistic rewards (`--no-probabilistic-rewards`)

Example:
```
python -m ludo_rl.train_sb3 --total-steps 2000000 --n-envs 8 --max-turns 600 \
  --checkpoint-freq 100000 --eval-freq 50000
```

## Using a Trained Model as Strategy
`ppo_strategy.py` wraps a saved PPO checkpoint so it can participate as a heuristic opponent or evaluation target. It reconstructs observation features from engine context and applies an action mask before argmax selection.

## RLlib Script
`train_rllib.py` offers a legacy alternative; may require updating for latest observation changes. Kept for experimentation with multi-algorithm training.

## Known Limitations
* Shared-policy reward interference (capture vs got_captured) reduces learning signal clarity.
* Longer wall-clock per useful agent decision vs single-seat variant.
* Large terminal reward magnitudes can dominate gradients; tune if instability observed.

## Migrating to `ludo_rls`
Switch import path (`from ludo_rls.envs.ludo_env import LudoGymEnv`) and adopt single-seat documentation there. Expect faster convergence and clearer terminal signals.

## Extension Ideas
* Integrate true action masking into policy network to avoid fallback penalties
* Add richer opponent modeling features (threat proximity, capture risk)
* Periodic opponent strategy resampling curriculum
* Population-based training with multiple concurrently evolving checkpoints

## License
See project root `LICENSE`.
