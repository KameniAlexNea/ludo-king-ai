from __future__ import annotations
from sb3_contrib import MaskablePPO

def linear_interp(start: float, end: float, frac: float) -> float:
    return start + (end - start) * min(1.0, max(0.0, frac))

def apply_linear_lr(model: MaskablePPO, start: float, end: float, frac: float) -> float:
    new_lr = linear_interp(start, end, frac)
    try:
        for g in model.policy.optimizer.param_groups:
            g["lr"] = new_lr
    except Exception:
        pass
    return new_lr
