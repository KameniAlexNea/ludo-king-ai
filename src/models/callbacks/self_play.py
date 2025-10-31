"""Callback utilities for self-play training."""

from __future__ import annotations

import os
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback

from ..envs.ludo_env_aec import OpponentPoolManager


class SelfPlayCallback(CheckpointCallback):
    """Callback that saves models to the opponent pool at checkpoints.

    Optionally also tracks and keeps only the top N models based on evaluation metrics.
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        opponent_pool: OpponentPoolManager,
        name_prefix: str = "ludo_ppo",
        best_model_dir: str | None = None,
        max_best_models: int = 20,
        metric_name: str = "eval/mean_reward",
        **kwargs,
    ) -> None:
        super().__init__(
            save_freq=save_freq, save_path=save_path, name_prefix=name_prefix, **kwargs
        )
        self.opponent_pool = opponent_pool
        self.best_model_dir = best_model_dir
        self.max_best_models = max_best_models
        self.metric_name = metric_name

        # Track best models: {model_path: metric_value}
        self.saved_models: dict[str, float] = {}
        if self.best_model_dir:
            self._load_existing_best_models()

    def _load_existing_best_models(self) -> None:
        """Load info about existing best models."""
        best_path = Path(self.best_model_dir)
        if not best_path.exists():
            best_path.mkdir(parents=True, exist_ok=True)
            return

        # Look for models matching pattern: best_12345_r42.3.zip
        for model_file in best_path.glob("best_*.zip"):
            try:
                parts = model_file.stem.split("_r")
                if len(parts) == 2:
                    reward = float(parts[1])
                    self.saved_models[str(model_file)] = reward
            except (ValueError, IndexError):
                pass

    def _save_best_model(self, metric_value: float) -> None:
        """Save model to best_models/ if it ranks in top N."""
        # Check if this model should be saved
        if len(self.saved_models) < self.max_best_models:
            should_save = True
        else:
            # Check if better than worst saved model
            worst_value = min(self.saved_models.values())
            should_save = metric_value > worst_value

        if should_save:
            # Create path with metric in filename
            best_path = Path(self.best_model_dir)
            model_path_with_metric = (
                best_path / f"best_{self.num_timesteps}_r{metric_value:.1f}.zip"
            )

            # Save model
            self.model.save(str(model_path_with_metric))
            self.saved_models[str(model_path_with_metric)] = metric_value

            if self.verbose >= 2:
                print(
                    f"Saved best model: {model_path_with_metric.name} (metric={metric_value:.3f})"
                )

            # Remove worst model if over limit
            if len(self.saved_models) > self.max_best_models:
                worst_path = min(self.saved_models.items(), key=lambda x: x[1])[0]
                worst_path_obj = Path(worst_path)
                if worst_path_obj.exists():
                    worst_path_obj.unlink()
                    if self.verbose >= 2:
                        print(f"Removed worst model: {worst_path_obj.name}")
                del self.saved_models[worst_path]

    def _on_step(self) -> bool:
        result = super()._on_step()

        # When a checkpoint is written, handle both opponent pool and best models
        if self.n_calls % self.save_freq == 0:
            latest_model = os.path.join(
                self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
            )
            if os.path.exists(latest_model):
                # Add to opponent pool
                self.opponent_pool.add_opponent(latest_model, self.num_timesteps)
                print(f"Added opponent to pool at timestep {self.num_timesteps}")

                # Save to best models if enabled and metric available
                if (
                    self.best_model_dir
                    and self.metric_name in self.logger.name_to_value
                ):
                    metric_value = self.logger.name_to_value[self.metric_name]
                    self._save_best_model(metric_value)
                elif self.best_model_dir and self.verbose >= 1:
                    print(
                        f"Warning: Metric '{self.metric_name}' not available for best model tracking"
                    )

        return result


__all__ = ["SelfPlayCallback"]
