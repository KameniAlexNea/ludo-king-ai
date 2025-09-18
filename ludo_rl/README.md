# ludo_rl (clean training package)

A simplified, clear training stack for Ludo using `ludo_engine`.
Focuses on: clean env wrapper, stable observation design, action masking, and a minimal curriculum.

## Install

Ensure project deps are installed (sb3, sb3-contrib, gymnasium, numpy, ludo_engine).

## Train

Run training with MaskablePPO:

```bash
python -m ludo_rl.train --total-steps 2000000 --n-envs 8
```

Artifacts:

- Logs: `training/logs`
- Models: `training/models`

## Design

- `config.py` — dataclasses for reward/env/curriculum/train
- `ludo_env/ludo_env.py` — single-seat Gym env with clean step/reset and action masking
- `env/observation.py` — stable, order-preserving features mapped to [-1,1]
- `utils/move_utils.py` — action mask builder
- `callbacks/curriculum.py` — push normalized training progress to envs
- `train.py` — vectorized training loop with VecNormalize and curriculum callback

## Notes

- Illegal actions are penalized mildly (-5) and a valid move is executed to keep play flowing.
- Opponent curriculum advances with training progress (25%/60%/90% boundaries), with 25% uniform sampling for diversity.
- Evaluation can be wired similarly to the existing tournament callback if needed.
