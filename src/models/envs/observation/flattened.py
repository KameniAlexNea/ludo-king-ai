"""Flattened observation builder that produces vector observations."""

from __future__ import annotations

import numpy as np

from .continuous import ContinuousObservationBuilder


class FlattenedObservationBuilder(ContinuousObservationBuilder):
    def build(self, dice: int) -> np.ndarray:
        obs_dict = super().build(dice)
        flat_obs = np.concatenate(list(obs_dict.values())).astype(np.float32)
        return flat_obs


__all__ = ["FlattenedObservationBuilder"]
