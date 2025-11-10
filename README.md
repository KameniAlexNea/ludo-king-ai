# Ludo King AI

A modern Ludo rules engine wrapped in a Gymnasium-compatible reinforcement learning (RL) environment. The project couples a differentiable feature extractor and Stable Baselines3's MaskablePPO agent to explore self-play training for Ludo King.

## Highlights

- Full Ludo simulator with movement validation, captures, blockades, and reward shaping.
- `LudoEnv` Gymnasium environment exposing rich observations and mandatory action masking.
- Custom 1D CNN feature extractor (`LudoCnnExtractor`) tailored to the stacked board representation.
- Ready-to-run PPO training script that handles vectorised environments, checkpoints, and TensorBoard logging.
- Configuration-first design (`ludo_rl/ludo/config.py`, `reward.py`) to tweak board, network, and reward parameters in one place.

## Project Layout

```
ludo_rl
├─ __init__.py → loads .env so simulator/env can read opponent strategy settings
├─ ludo_env.LudoEnv (Gymnasium Env)
│  ├─ wraps GameSimulator to expose observation dict & masked Discrete(4) actions
│  ├─ handles invalid moves, turn limit, reward shaping, and rendering snapshots
│  └─ converts simulator telemetry into 10-channel board tensor + dice_roll token
├─ .ludo_king.simulator.GameSimulator
│  ├─ owns LudoGame, tracks agent_index, and keeps heatmaps/summary channels
│  ├─ steps agent move, then rolls opponents (respecting extra turns)
│  └─ injects scripted opponent strategies via environment variables
├─ .ludo_king.game.LudoGame
│  ├─ instantiates 4 Player objects and provides dice + move plumbing
│  ├─ delegates legal-move logic to MoveManagement
│  └─ builds per-agent board_state channels consumed by env + strategies
├─ .ludo_king.moves.MoveManagement
│  ├─ enforces rules: entry, home stretch, captures, blockades, extra turns
│  └─ invokes reward.compute_move_rewards for dense shaping events
├─ .ludo_king.player.Player
│  ├─ keeps Piece state, win detection, and chooses moves via strategies
│  └─ falls back to random if requested heuristic unavailable
├─ strategy package
│  ├─ features.build_move_options turns env observation into StrategyContext
│  ├─ BaseStrategy + concrete heuristics (defensive, killer, etc.) score MoveOption
│  └─ registry.create/available expose factories to simulator & players
├─ extractor.LudoCnnExtractor / LudoTransformerExtractor
│  ├─ convert observation dict into feature vectors for MaskablePPO
│  └─ fuse CNN/Transformer encodings with per-piece embeddings and dice token
├─ tools (arguments, scheduler, evaluate, etc.)
│  └─ supporting scripts for training, tournaments, imitation, scheduling
└─ train.py
   ├─ parses CLI args, configures MaskablePPO w/ custom extractor
   └─ runs vectorized envs, callbacks (checkpoints, entropy annealing, profiler)
```

- LudoEnv mediates RL interaction: builds masked actions, enforces rewards, loops until player or opponents advance, and emits 10-channel observations.
- GameSimulator orchestrates turns: applies agent move, simulates opponents with heuristic strategies, and updates heatmaps consumed by both env and feature extractors.
- Core rules live in LudoGame + MoveManagement + Piece/Player, with reward.compute_move_rewards producing shaped returns for PPO.
- Strategy module supplies configurable heuristics; features.build_move_options transforms env data into StrategyContext so Player.decide can score moves consistently.
- extractor.py houses CNN/Transformer feature pipelines that embed board channels, per-piece context, and dice roll before feeding MaskablePPO during training (train.py).


## Getting Started

### Prerequisites

- Python 3.11+
- A virtual environment is recommended (`python -m venv .venv && source .venv/bin/activate`).

### Installation

Install the package and dependencies in editable mode:

```bash
pip install -e .
```

Alternatively, install the raw dependencies:

```bash
pip install -r requirements.txt
```

## Training the Agent

The `train.py` script configures MaskablePPO with the custom feature extractor and launches multi-process self-play training.

```bash
python train.py
```

What the script does:

1. Creates timestamped subdirectories under `training/ludo_logs/` and `training/ludo_models/`.
2. Spawns `SubprocVecEnv` workers (half the available CPU cores) and wraps them with `VecMonitor`.
3. Sets up checkpointing every 10k steps and periodic evaluation (20k step cadence).
4. Trains for 1,000,000 timesteps, saves the initial and final policies, and performs a short interactive rollout.

TensorBoard logs end up in the run-specific `training/ludo_logs/<run_id>/` directory:

```bash
tensorboard --logdir training/ludo_logs
```

## Customisation

- **Rewards**: Adjust per-event incentives in `ludo_rl/ludo/reward.py`.
- **Network**: Tune convolution and MLP widths in `ludo_rl/ludo/config.py` (`NetworkConfig`).
- **Environment**: Modify truncation length (`MAX_TURNS`) or add render hooks in `ludo_rl/ludo_env.py`.
- **Training Hyperparameters**: Tweak PPO arguments and callback intervals in `train.py`.

## Current Status & Roadmap

- ✅ Environment, simulator, and training loop are in place.
- ✅ Evaluation tooling (e.g., scripted benchmarks, head-to-head matches).

## Development

Static analysis and linting use `tox`:

```bash
tox
```

The default configuration runs formatting checks (via Ruff/Black if installed) and unit tests once they are introduced.

## License

Released under the Apache License. See `LICENSE` for details.
