# Ludo King AI Suite

Modern Ludo engine + reinforcement learning environments + evaluation tooling.

This repository contains three coherent layers:

1. Core Engine (`ludo/`) – deterministic Python implementation of game rules & heuristic strategies.
2. Classic Multi‑Seat RL (`ludo_rl/`) – original environment where a single policy sequentially controls all colors (higher credit noise, full visibility of every turn).
3. Single‑Seat Self‑Play RL (`ludo_rls/`) – optimized Option B environment: one randomly chosen seat per episode is controlled by the learner; other seats are internally fast‑forwarded using a frozen snapshot or scripted strategies (cleaner credit assignment, ~4x fewer external steps).

Consult the dedicated READMEs in each subpackage for deep details. This root file gives a high‑level map and quick onboarding path.

## Feature Highlights
* Deterministic game core (seedable) with complete rule coverage (captures, safe squares, chained extra turns, three sixes rule, home column exact finish)
* Rich strategy ecosystem (aggressive, defensive, balanced, probabilistic variants, LLM bridge)
* Two RL training paradigms (multi‑seat vs single‑seat) with modular reward shaping
* Stable-Baselines3 PPO training scripts, checkpointing & tournament evaluation
* Tournament + statistics utilities (JSON state saving, capture & legality metrics, win/rank tracking)
* Action masking & illegal action penalty scaling (single‑seat env)
* Extensible reward configuration (progress, capture, finish, diversity, time, illegals, terminal win/lose/draw)

## Repository Layout (Condensed)
```
README.md              # (you are here)
ludo/                  # Core engine (see ludo/README.md)
ludo_rl/               # Classic multi-seat RL env (see ludo_rl/README.md)
ludo_rls/              # Single-seat fast-forward RL env (see ludo_rls/README.md)
four_player_tournament.py  # Heuristic / mixed strategy tournaments
manual_tournament.py       # Manual / scripted runs
ludo_evaluator.py          # Evaluation harness / stats
ludo_stats/                # Game state saving & analysis utilities
models/                    # Saved PPO model artifacts
logs/                      # TensorBoard & evaluation logs
tests/                     # Unit & integration tests
ludo_gr/                   # Gradio AI-vs-AI visualization interface
```

## Choosing an Environment
| Goal | Use | Reason |
|------|-----|--------|
| Analyze symmetric self-play, see every move | `ludo_rl` | Full sequence exposure, debugging shaping across seats |
| Faster convergence, cleaner signal | `ludo_rls` | Single-seat random color, internal opponent fast-forward |
| Pure rule testing / custom heuristics | `ludo` | Lightweight, no RL wrappers |

## Reward Philosophy (Summary)
Single-seat (`ludo_rls`): moderate terminal magnitudes (Win +60 / Lose -60 / Draw -2) + shaped components (capture ±6, finish +8, progress, diversity, extra turn +2, time penalty, illegal penalties with mask scaling).

Classic multi-seat (`ludo_rl`): larger terminal magnitudes (Win +200 / Lose -200) with similar shaping themes (progress scaling, capture/got_captured, finish, diversity, extra turn, illegal penalty).

Both centralize terminal logic inside reward calculators; shaping is additive but tuned for gradient stability.

## Quick Start (Single-Seat PPO Recommended)
Train PPO (Stable-Baselines3):
```bash
python -m ludo_rls.train_sb3 --total-steps 1000000 --n-envs 8 \
    --tournament-freq 100000 --tournament-games 240
```

Resume from checkpoint:
```bash
python -m ludo_rls.train_sb3 --load-path models/ppo_latest.zip --total-steps 500000
```

Classic environment training example:
```bash
python -m ludo_rl.train_sb3 --total-steps 2000000 --n-envs 8 --max-turns 600 \
    --checkpoint-freq 100000 --eval-freq 50000
```

TensorBoard:
```bash
tensorboard --logdir=logs_self
```

## Tournament & Evaluation
Run a heuristic tournament (multi-strategy):
```bash
python four_player_tournament.py
```

Visualize an AI vs AI game (interactive board):
```bash
python -m ludo_gr.app
```

Single-seat periodic evaluation is integrated via `--tournament-freq` / `--tournament-games` flags (see `ludo_rls/README.md`). Metrics include win rate, mean rank, capture stats (offensive/defensive), capture ratio, PPO & overall illegal rates, average turns.

## Strategy System
Create and register a custom strategy:
```python
from ludo.strategy import StrategyFactory, Strategy

class MyHeuristic(Strategy):
        def decide(self, game_context):
                # Return token index 0..3
                return 0

# Register (edit ludo/strategies/__init__.py STRATEGIES dict)
```
List available built-ins:
```python
from ludo.strategy import StrategyFactory
print(StrategyFactory.get_available_strategies())
```

Use a trained PPO as a heuristic (classic multi-seat opponents):
```python
from ludo_rl.ppo_strategy import PPOStrategy
opp = PPOStrategy("models/ppo_ludo_final.zip")
```

## Extending / Ideas
* Richer spatial features (threat proximity, safety gradients)
* Periodic frozen opponent snapshot lag (single-seat)
* Population-based training / league systems
* Curriculum opponent sampling distributions
* Unified reward scaling across env variants
* Improved action masking integration into NN architecture

## Testing
Run the test suite:
```bash
pytest -q
```

Tests cover engine invariants, environment API, strategy integration, and RL wrappers.

## Troubleshooting
| Symptom | Suggestion |
|---------|------------|
| Flat early rewards | Confirm terminal events; allow longer horizon; adjust win magnitude cautiously |
| High illegal rate | Ensure `use_action_mask=True` (single-seat) & review `illegal_masked_scale` |
| Slow convergence | Use single-seat env; reduce evaluation frequency; verify reward magnitudes |
| Instability (value loss spikes) | Lower terminal magnitudes or clip rewards |

## Roadmap (High-Level)
Completed:
* Single-seat Option B environment
* Centralized terminal win/lose/draw rewards
* Tournament callback with extended metrics
* Comprehensive layered documentation

Planned / Potential:
* Policy snapshot lag cadence configuration
* Advanced opponent modeling features
* League / population training utilities
* Reward normalization experiments

## Recent Tournament Results
Excerpt of latest 200-game evaluations (see `ludo_experiments/results.md` for full tables):

### PPO vs Other Strategies
| Rank | Model | Win % | Avg Turns |
|------|-------|------:|----------:|
| 1 | PPO_LUDO_FINAL | 36.0 | 127.5 |
| 2 | BALANCED | 26.0 | 127.8 |
| 3 | OPTIMIST | 25.5 | 128.3 |
| 4 | PROBABILISTIC_V3 | 12.5 | 128.2 |

### PPO Self-Improvement (vs Frozen PPO)
| Rank | Model | Win % | Avg Turns |
|------|-------|------:|----------:|
| 1 | PPO_LUDO_FINAL | 38.5 | 131.3 |
| 2 | BALANCED | 26.5 | 131.3 |
| 3 | OPTIMIST | 22.0 | 131.9 |
| 4 | PROBABILISTIC_V3 | 13.0 | 131.9 |

PPO now leads both the mixed-strategy and frozen self-play benchmarks; next focus: curriculum graduation scheduling, snapshot lag stabilization, and late-game efficiency shaping.

## License
See root `LICENSE`.

## Acknowledgements
Inspired by classic board game RL research and community open-source Ludo implementations; expanded for modern self-play experimentation.

---
For deeper details jump into: `ludo/README.md`, `ludo_rl/README.md`, `ludo_rls/README.md`.
