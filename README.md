# Ludo King AI

A modern Ludo rules engine wrapped in a Gymnasium-compatible reinforcement learning (RL) environment. The project couples a differentiable feature extractor and Stable Baselines3's MaskablePPO agent to explore self-play training for Ludo King.

## Highlights

- Full Ludo simulator with movement validation, captures, blockades, and reward shaping.
- `LudoEnv` Gymnasium environment exposing rich observations and mandatory action masking.
- Custom 1D CNN feature extractor (`LudoCnnExtractor`) tailored to the stacked board representation.
- Ready-to-run PPO training script that handles vectorised environments, checkpoints, and TensorBoard logging.
- Configuration-first design (`ludo_rl/ludo/config.py`, `reward.py`) to tweak board, network, and reward parameters in one place.

## Project Layout

```text
ludo_rl/
  extractor.py          # Torch CNN feature extractor used by MaskablePPO
  ludo_env.py           # Gymnasium environment bridging the simulator and SB3
  ludo/                 # Pure game logic (rules, rewards, simulator, players)
train.py                # Example training entry point using MaskablePPO
requirements.txt        # Runtime dependencies (mirrors pyproject[project.dependencies])
```

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
- ⏳ Evaluation tooling (e.g., scripted benchmarks, head-to-head matches) is not implemented yet and is tracked for future work.

## Development

Static analysis and linting use `tox`:

```bash
tox
```

The default configuration runs formatting checks (via Ruff/Black if installed) and unit tests once they are introduced.

## License

Released under the MIT License. See `LICENSE` for details.
