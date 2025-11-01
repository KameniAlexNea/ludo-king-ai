import copy

import torch
from stable_baselines3.common.callbacks import BaseCallback

from ..analysis.eval_utils import evaluate_against_many
from ..configs.config import EnvConfig


class PeriodicEvalCallback(BaseCallback):
    """Run lightweight policy evaluations at a fixed timestep cadence."""

    def __init__(
        self,
        env_cfg: EnvConfig,
        opponents: tuple[str, ...],
        episodes: int,
        eval_freq: int,
        deterministic: bool,
    ) -> None:
        if eval_freq <= 0:
            raise ValueError("eval_freq must be positive")
        super().__init__(verbose=1)
        self.base_cfg = copy.deepcopy(env_cfg)
        self.opponents = opponents
        self.episodes = episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self._next_eval = eval_freq
        self._last_eval_step = 0

    @torch.no_grad()
    def _run_eval(self) -> None:
        summaries = evaluate_against_many(
            self.model,
            self.opponents,
            self.episodes,
            self.base_cfg,
            self.deterministic,
        )
        step = int(self.num_timesteps)
        if self.verbose > 0:
            print(f"\n[Eval] timesteps={step}")
            for summary in summaries:
                print(
                    f"  vs {summary.opponent}: win_rate={summary.win_rate:.3f} "
                    f"avg_reward={summary.avg_reward:.2f} avg_length={summary.avg_length:.1f} "
                    f"terminal={summary.avg_terminal_score():.2f}"
                )
        for summary in summaries:
            prefix = f"eval/{summary.opponent}"
            self.logger.record(f"{prefix}/win_rate", summary.win_rate)
            self.logger.record(f"{prefix}/avg_reward", summary.avg_reward)
            self.logger.record(f"{prefix}/avg_length", summary.avg_length)
            self.logger.record(
                f"{prefix}/avg_terminal_score", summary.avg_terminal_score()
            )
            # Log key breakdown components to help diagnose eval/training mismatch
            summary_dict = summary.as_dict()
            self.logger.record(
                f"{prefix}/bd/progress", summary_dict["breakdown/progress"]
            )
            self.logger.record(
                f"{prefix}/bd/capture", summary_dict["breakdown/capture"]
            )
            self.logger.record(
                f"{prefix}/bd/finish", summary_dict["breakdown/finish"]
            )
            self.logger.record(
                f"{prefix}/bd/got_captured", summary_dict["breakdown/got_captured"]
            )
            self.logger.record(
                f"{prefix}/bd/time_penalty", summary_dict["breakdown/time_penalty"]
            )
        self.logger.dump(step)
        self._last_eval_step = step

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._run_eval()
            self._next_eval += self.eval_freq
        return True

    def _on_training_end(self) -> None:
        if int(self.num_timesteps) != self._last_eval_step:
            self._run_eval()
