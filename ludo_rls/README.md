# Ludo RL (Single-Seat Self-Play Suite)

Advanced reinforcement learning environments, reward shaping, and evaluation tooling for Ludo.

The `ludo_rls` variant implements **Option B single-seat training**: only one randomly chosen player color per episode is controlled by the learning agent; other seats are simulated internally using a frozen snapshot of the current policy (or scripted strategies where configured). This eliminates self-canceling rewards and concentrates learning signal.

## Key Features
* Single-seat PPO friendly environment (stable credit assignment)
* Internal opponent fast-forwarding for hidden turns (≈ 4x fewer external steps)
* Modular reward shaping (movement, capture, token finish, win/lose, draw, progress, diversity, time, illegals)
* Action masking with optional masked penalty scaling
* Tournament callback evaluating against heuristic strategy mixes with rich metrics (win rate, capture stats, illegal rates, offensive/defensive ratios)
* Deterministic seeding & random per-episode training color (configurable)

## Structure
```
ludo_rls/
  envs/
    ludo_env.py                 # Option B single-seat Gymnasium environment
    calculators/simple_reward_calculator.py
    model.py                    # Env & reward configuration dataclasses
    builders/                   # Observation assembly utilities
    utils/                      # Move utilities & helpers
  callbacks/
    tournament_callback.py      # Periodic evaluation logic
  train_sb3.py                  # Stable-Baselines3 (PPO) training entrypoint
  train_rllib.py                # (Legacy) RLlib script (may not use single-seat changes)
  __init__.py
```

## Environment
`LudoGymEnv` (not globally registered) returns only agent seat turns. Opponent turns are simulated internally until it's the agent's turn again. Each `env.step()` corresponds to exactly one decision for the active training color.

### Turn Flow
1. Reset selects a training color (random if `randomize_training_color=True`)
2. Opponents are simulated (frozen policy snapshot) until agent's turn
3. Observation emitted; agent acts
4. If extra turn (rolled 6 / rule-based), another step immediately
5. Otherwise opponents fast-forward again

### Observations
Vector (normalized) including:
* Token positions & statuses (agent + opponents)
* Finished token counts
* Dice value (raw + normalized flag)
* Progress aggregates / turn index (if enabled)
* Blocking / safety features (configurable flags)

### Action Space
`Discrete(4)` choosing which token to move. Invalid selections optionally auto-corrected when `use_action_mask=True` (with scaled penalty).

## Reward Design
Centralized in `RewardConfig` + `SimpleRewardCalculator`:

Event Rewards:
* Win: +60 (sparse terminal)
* Lose: -60
* Draw (timeout): -2 (applied via terminal path with truncation)
* Finish token: +8
* Capture: +6 per opponent token
* Got captured: -6 per own token

Shaping / Signals:
* Movement progress (scaled incremental progress)
* Diversity bonus (first activation of each token)
* Extra turn bonus (+2)
* Time penalty (-0.01 per agent step)
* Illegal action: -8 (scaled by `illegal_masked_scale`=0.25 when auto-corrected)

Terminal reward logic now consolidates win / lose / draw evaluation via `get_terminal_reward(truncated=...)`.

## Configuration (excerpt)
See `model.py`:
```
EnvConfig(
  max_turns=1000,
  randomize_training_color=True,
  use_action_mask=True,
  reward_cfg=RewardConfig(...)
)
```
`single_seat_training` is deprecated (always single-seat now).

## Evaluation
`tournament_callback.py` runs periodic multi-game evaluations versus heuristic strategies (killer, winner, balanced, defensive, random, etc.). Metrics logged:
* win_rate, mean_rank
* offensive_captures / defensive_captures
* capture_diff, capture_ratio
* illegal_rate (overall & PPO only)
* avg_turns

Integrate by passing `--tournament-freq` and `--tournament-games` to `train_sb3.py`.

## Usage (Stable-Baselines3 PPO)
Basic training:
```
python -m ludo_rls.train_sb3 --total-steps 1000000 --n-envs 8 \
  --tournament-freq 100000 --tournament-games 240
```
Models saved under `models/`, logs under `logs_self/` (TensorBoard compatible).

Resume training (example):
```
python -m ludo_rls.train_sb3 --load-path models/ppo_latest.zip --total-steps 500000
```

## Design Rationale
Single-seat fast-forwarding removes ambiguous credit assignment caused by one policy controlling all seats (capture vs got_captured neutrality). It yields cleaner gradients and higher sample efficiency (each step is an actual agent decision).

## Extending Further
* Add richer spatial features (danger proximity, escape routes)
* Opponent model lag (freeze snapshot every N updates instead of per episode)
* Curriculum of scripted opponent mixtures
* Population-based training (multiple evolving checkpoints)
* Policy distillation from diverse heuristic ensembles

## Troubleshooting
* Flat rewards early: verify terminal events occurring; increase win magnitude cautiously
* High illegal rate: enable `use_action_mask`; inspect `illegal_masked_scale`
* Slow improvement: ensure tournament frequency not too high (evaluation pauses training)

## License
See root `LICENSE`.

