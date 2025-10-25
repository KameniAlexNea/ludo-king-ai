"""Opponent pool management for self-play training."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class OpponentPoolManager:
    """Manages a pool of opponent models for self-play training."""

    def __init__(self, pool_dir: str, pool_size: int = 5):
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.pool_size = pool_size
        self.opponents: list[str] = []
        self._load_existing_opponents()

    def _load_existing_opponents(self) -> None:
        if not self.pool_dir.exists():
            return

        opponent_files = sorted(
            self.pool_dir.glob("opponent_*.zip"), key=lambda p: p.stat().st_mtime
        )
        self.opponents = [str(f) for f in opponent_files[-self.pool_size :]]

    def add_opponent(self, model_path: str, timestep: int) -> None:
        opponent_path = self.pool_dir / f"opponent_{timestep}.zip"

        import shutil

        shutil.copy(model_path, opponent_path)

        self.opponents.append(str(opponent_path))

        if len(self.opponents) > self.pool_size:
            old_opponent = Path(self.opponents.pop(0))
            if old_opponent.exists():
                old_opponent.unlink()

    def sample_opponent(self) -> Optional[str]:
        if not self.opponents:
            return None
        return np.random.choice(self.opponents)

    def get_all_opponents(self) -> list[str]:
        return self.opponents.copy()
