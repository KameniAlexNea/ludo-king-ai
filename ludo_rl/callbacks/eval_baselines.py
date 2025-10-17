import os
import re
import zlib
from typing import Dict, Optional, Sequence

import numpy as np
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from ludo_rl.config import EnvConfig
from ludo_rl.utils.opponents import build_opponent_combinations


class SimpleBaselineEvalCallback(MaskableEvalCallback):
    """Periodically evaluate the current policy vs fixed baselines.

    Plays 1v3 games where the agent always sits in one seat and the other 3
    seats are filled by scripted strategies specified in `baselines`. We run
    a number of games and compute win rate and average turns. Results are
    logged to TensorBoard.

    Notes:
    - Uses a separate eval env with VecNormalize sharing obs_rms for parity.
    - Sampling tries different opponent order permutations for diversity.
    - Keeps it simple: no ranks, just win/lose rate.
    """

    def __init__(
        self,
        baselines: Sequence[str],
        n_games: int = 60,
        eval_freq: int = 100_000,
        verbose: int = 0,
        env_cfg: Optional[EnvConfig] = None,
        eval_env: VecNormalize = None,
        best_model_save_path: Optional[str] = None,
        callback_on_new_best=None,
        callback_after_eval=None,
        log_path: Optional[str] = None,
        deterministic_eval: bool = True,
    ):
        self.n_games = int(n_games)
        self.env_cfg = env_cfg or EnvConfig()
        self.setups = self.env_cfg.allowed_player_counts
        self.deterministic_eval = deterministic_eval

        if eval_env is None:
            raise ValueError("eval_env must be provided")

        self.eval_env = eval_env

        # Initialize parent with eval_env and other parameters. We'll override _on_step
        # to run a custom evaluation loop that also logs win-rate and reward breakdowns.
        super().__init__(
            eval_env=self.eval_env,
            n_eval_episodes=self.n_games,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=False,
            render=False,
            verbose=verbose,
            warn=True,
            use_masking=True,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
        )
        # Pre-build a list of opponent combinations per setup size with the correct
        # number of opponents (players-1). Example: 2-player -> 1 opponent, 4-player -> 3 opponents.
        # Use deterministic mode to cycle through opponents sequentially instead of random sampling.
        self.baselines_per_setup = []
        for setup in self.setups:
            n_comb = max(0, int(setup) - 1)
            combos = build_opponent_combinations(
                list(baselines),
                n_games=self.n_games,
                n_comb=n_comb,
                deterministic=self.deterministic_eval,
            )
            self.baselines_per_setup.append(combos)
        self.executed = 0

        # Early stopping: track win rate improvements
        self.best_win_rate = -1.0
        self.non_improving_evals = 0
        self.max_non_improving_evals = 10

    def _short_metric_key(self, key: str) -> str:
        """Create a compact, unique metric key to avoid SB3 logger truncation collisions."""
        tokens = re.findall(r"[A-Za-z0-9]+", str(key).lower())
        if not tokens:
            tokens = ["term"]
        core = "_".join(t[:4] for t in tokens[:3])
        h = format(zlib.adler32(str(key).encode("utf-8")) & 0xFFFFFFFF, "x")[:4]
        return f"{core}_{h}"

    def _on_step(self) -> bool:
        # Evaluate every eval_freq steps
        if not (self.eval_freq > 0 and self.n_calls % self.eval_freq == 0):
            return True
        total_setups = len(self.setups)
        setup_idx = (self.executed // self.n_games) % max(1, total_setups)
        setup = self.setups[setup_idx]
        idx_in_setup = self.executed % self.n_games
        opponents = self.baselines_per_setup[setup_idx][idx_in_setup]

        continue_training = True

        # Set attributes on the eval env for this setup
        self.eval_env.set_attr("opponents", opponents)
        self.eval_env.set_attr("fixed_num_players", setup)

        # Sync normalization stats if applicable (keeps parity with parent)
        if self.model.get_vec_normalize_env() is not None:
            try:
                sync_envs_normalization(self.training_env, self.eval_env)
            except AttributeError as e:
                raise AssertionError(
                    "Training and eval env are not wrapped the same way, "
                    "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                    "and warning above."
                ) from e

        # Aggregators for breakdowns and wins
        episodes_breakdowns: list[Dict[str, float]] = []
        wins = 0

        def trace_callback(locals_, _globals=None):
            nonlocal episodes_breakdowns, wins
            infos = locals_.get("infos")
            dones = locals_.get("dones")
            # accumulate step-level breakdown into a temp episode dict
            if not hasattr(trace_callback, "_ep_bd"):
                trace_callback._ep_bd = {}
            if infos is not None:
                info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
                rb = info0.get("reward_breakdown") if isinstance(info0, dict) else None
                if rb:
                    for k, v in rb.items():
                        trace_callback._ep_bd[k] = trace_callback._ep_bd.get(
                            k, 0.0
                        ) + float(v)
            # handle episode end
            if dones is not None:
                done0 = bool(
                    dones[0] if isinstance(dones, (list, np.ndarray)) else dones
                )
                if done0:
                    # determine win/lose from info dict (which is populated by _build_step_info)
                    info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
                    is_win = (
                        info0.get("agent_won", False)
                        if isinstance(info0, dict)
                        else False
                    )
                    wins += 1 if is_win else 0
                    episodes_breakdowns.append(trace_callback._ep_bd)
                    trace_callback._ep_bd = {}

        # Use sb3-contrib evaluate_policy to ensure masking is applied
        episode_rewards, episode_lengths = evaluate_policy(
            self.model,  # type: ignore[arg-type]
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            render=self.render,
            deterministic=self.deterministic,
            return_episode_rewards=True,
            warn=self.warn,
            callback=trace_callback,
            use_masking=True,
        )

        # Aggregate metrics
        win_rate = wins / max(1, self.n_eval_episodes)
        # average per-term across episodes (missing terms treated as 0)
        agg_keys = (
            set().union(*(bd.keys() for bd in episodes_breakdowns))
            if episodes_breakdowns
            else set()
        )
        avg_breakdown: Dict[str, float] = {}
        for k in agg_keys:
            s = sum(bd.get(k, 0.0) for bd in episodes_breakdowns)
            avg_breakdown[k] = s / max(1, len(episodes_breakdowns))

        mean_reward, std_reward = (
            float(np.mean(episode_rewards)),
            float(np.std(episode_rewards)),
        )
        mean_ep_length, std_ep_length = (
            float(np.mean(episode_lengths)),
            float(np.std(episode_lengths)),
        )
        self.last_mean_reward = float(mean_reward)

        # Check win-rate improvement for early stopping
        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            self.non_improving_evals = 0
        else:
            self.non_improving_evals += 1

        # Logging (stdout)
        if self.verbose > 0:
            print(
                f"Eval num_timesteps={self.num_timesteps}, episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
            )
            print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            print(
                f"Win rate: {win_rate * 100:.2f}% (best: {self.best_win_rate * 100:.2f}%, non-improving: {self.non_improving_evals}/{self.max_non_improving_evals})"
            )

        # Tensorboard logs
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_ep_length", mean_ep_length)
        self.logger.record("eval/win_rate", win_rate)
        # Log breakdown terms using compact, unique keys to avoid truncation collisions
        for k, v in avg_breakdown.items():
            sk = self._short_metric_key(k)
            self.logger.record(f"eval/bd/{sk}", float(v))

        # Save eval arrays similar to parent callback
        if self.log_path is not None:
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(episode_rewards.tolist())
            self.evaluations_length.append(episode_lengths.tolist())
            np.savez(
                self.log_path,
                timesteps=self.evaluations_timesteps,
                results=self.evaluations_results,
                ep_lengths=self.evaluations_length,
            )

        # Best-model checkpoint logic
        if mean_reward > self.best_mean_reward:
            if self.verbose > 0:
                print("New best mean reward!")
            if self.best_model_save_path is not None:
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            self.best_mean_reward = float(mean_reward)
            if self.callback_on_new_best is not None:
                continue_training = self.callback_on_new_best.on_step()

        # Trigger callback after every evaluation, if needed (parity with parent)
        if self.callback is not None:
            continue_training = continue_training and self._on_event()

        # Dump log
        self.logger.record(
            "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
        )
        self.logger.dump(self.num_timesteps)

        self.eval_env.set_attr("opponents", None)
        self.eval_env.set_attr("fixed_num_players", None)
        self.executed += 1

        # Early stopping: if win rate hasn't improved for N consecutive evals, stop training
        if self.non_improving_evals >= self.max_non_improving_evals:
            if self.verbose > 0:
                print(
                    f"[Early Stop] No win rate improvement for {self.max_non_improving_evals} consecutive evaluations. "
                    f"Best win rate: {self.best_win_rate * 100:.2f}%. Stopping training."
                )
            return False  # Signal training to stop

        return continue_training
