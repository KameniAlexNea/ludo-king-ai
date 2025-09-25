import os
from typing import List, Optional, Sequence

import numpy as np
from loguru import logger
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from ludo_rl.config import EnvConfig
from ludo_rl.ludo_env.ludo_env import LudoRLEnv
from ludo_rl.ludo_env.ludo_env_base import StepInfo
from ludo_rl.utils.move_utils import MoveUtils
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
        log_prefix: str = "eval/",
        verbose: int = 0,
        env_cfg: Optional[EnvConfig] = None,
        eval_env: VecNormalize = None,
        best_model_save_path: Optional[str] = None,
        callback_on_new_best=None,
        callback_after_eval=None,
        log_path: Optional[str] = None,
    ):
        self.baselines = list(baselines)
        self.n_games = int(n_games)
        self.eval_freq = int(eval_freq)
        self.log_prefix = log_prefix.rstrip("/") + "/"
        self.env_cfg = env_cfg or EnvConfig()

        if eval_env is None:
            raise ValueError("eval_env must be provided")

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Track best win rate per player-count setup (e.g., 2,3,4)
        self.best_win_rate = {}

        # Initialize parent with eval_env and other parameters
        super().__init__(
            eval_env=self.eval_env,
            n_eval_episodes=1,  # Not used since we override evaluation
            eval_freq=self.eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=True,
            render=False,
            verbose=verbose,
            warn=True,
            use_masking=True,  # Enable masking for Ludo
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
        )

    def _on_step(self) -> bool:
        # Evaluate every eval_freq steps
        if self.eval_freq <= 0:
            return True
        if self.num_timesteps == 0 or (self.num_timesteps % self.eval_freq) != 0:
            return True

        # Sync training and eval env normalization (from parent)
        if self.model.get_vec_normalize_env() is not None:
            try:
                from stable_baselines3.common.callbacks import sync_envs_normalization

                sync_envs_normalization(self.training_env, self.eval_env)
            except AttributeError as e:
                raise AssertionError(
                    "Training and eval env are not wrapped the same way, "
                    "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                    "and warning above."
                ) from e

        try:
            episode_rewards, episode_lengths, successes = self._run_eval()
        except Exception as e:
            if self.verbose:
                logger.error(f"[Eval] Error during evaluation: {e}")
            return True

        continue_training = self._log_results(episode_rewards, episode_lengths, successes)
        return continue_training

    def _log_results(self, episode_rewards, episode_lengths, successes):
        continue_training = True

        if self.log_path is not None:
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(episode_rewards)
            self.evaluations_length.append(episode_lengths)
            kwargs = {}
            if successes:
                self.evaluations_successes.append(successes)
                kwargs = dict(successes=self.evaluations_successes)
            np.savez(
                self.log_path,
                timesteps=self.evaluations_timesteps,
                results=self.evaluations_results,
                ep_lengths=self.evaluations_length,
                **kwargs,
            )

        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        self.last_mean_reward = float(mean_reward)

        if self.verbose > 0:
            print(f"Eval num_timesteps={self.num_timesteps}, "
                  f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
        # Add to current Logger
        self.logger.record("eval/mean_reward", float(mean_reward))
        self.logger.record("eval/mean_ep_length", mean_ep_length)

        if successes:
            success_rate = np.mean(successes)
            if self.verbose > 0:
                print(f"Success rate: {100 * success_rate:.2f}%")
            self.logger.record("eval/success_rate", success_rate)

        # Dump log so the evaluation results are printed with the correct timestep
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(self.num_timesteps)

        if mean_reward > self.best_mean_reward:
            if self.verbose > 0:
                print("New best mean reward!")
            if self.best_model_save_path is not None:
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            self.best_mean_reward = float(mean_reward)
            # Trigger callback on new best model, if needed
            if self.callback_on_new_best is not None:
                continue_training = self.callback_on_new_best.on_step()

        # Trigger callback after every evaluation, if needed
        if self.callback_after_eval is not None:
            continue_training = continue_training and self.callback_after_eval.on_step()

        return continue_training

    def _log_episode(self, step_info: StepInfo, episode_reward, episode_breakdown, prefix):
        # Use record_mean for automatic TensorBoard averaging
        self.logger.record_mean(
            prefix + "avg_offensive_captures",
            step_info.episode_captured_opponents,
        )
        self.logger.record_mean(
            prefix + "avg_defensive_captures",
            step_info.episode_captured_by_opponents,
        )
        self.logger.record_mean(
            prefix + "avg_finished_tokens",
            step_info.finished_tokens,
        )
        self.logger.record_mean(
            prefix + "avg_episode_reward", episode_reward
        )
        # Record opportunity rates per game for averaging
        cap_rate = (
            step_info.episode_capture_ops_taken
            / step_info.episode_capture_ops_available
            if step_info.episode_capture_ops_available > 0
            else 0.0
        )
        fin_rate = (
            step_info.episode_finish_ops_taken
            / step_info.episode_finish_ops_available
            if step_info.episode_finish_ops_available > 0
            else 0.0
        )
        exit_rate = (
            step_info.episode_home_exit_ops_taken
            / step_info.episode_home_exit_ops_available
            if step_info.episode_home_exit_ops_available > 0
            else 0.0
        )
        self.logger.record_mean(
            prefix + "capture_opportunity_rate", cap_rate
        )
        self.logger.record_mean(
            prefix + "finish_opportunity_rate", fin_rate
        )
        self.logger.record_mean(
            prefix + "exit_opportunity_rate", exit_rate
        )
        # Log reward breakdown components if available
        if episode_breakdown:
            for key, val in episode_breakdown.items():
                try:
                    self.logger.record_mean(
                        f"{prefix}reward_breakdown/{key}", float(val)
                    )
                except Exception:
                    pass

    def _log_setup_results(self, setup, win_rate, avg_turns):
        # Log to TB if available (setup-specific keys)
        self.logger.record(f"{self.log_prefix}setup_{setup}/win_rate", win_rate)
        self.logger.record(f"{self.log_prefix}setup_{setup}/avg_turns", avg_turns)

        # Save best model if win_rate improved for this setup
        prev_best = self.best_win_rate.get(setup, -np.inf)
        if win_rate > prev_best:
            if self.verbose:
                print(f"New best win rate for setup {setup}: {win_rate:.3f}!")
            if self.best_model_save_path is not None:
                save_dir = os.path.join(self.best_model_save_path, f"setup_{setup}")
                os.makedirs(save_dir, exist_ok=True)
                self.model.save(os.path.join(save_dir, "best_model"))
            self.best_win_rate[setup] = win_rate

    def _run_eval(self):
        # Overall lists for logging
        episode_rewards = []
        episode_lengths = []
        successes = []

        # Decide which player-count setups to evaluate. If env is fixed, only
        # evaluate that setup; otherwise evaluate all allowed counts.
        if self.env_cfg.fixed_num_players is not None:
            setups = [self.env_cfg.fixed_num_players]
        else:
            setups = list(self.env_cfg.allowed_player_counts)

        # Evaluate separately for each setup and log/save results per-setup
        for setup in setups:
            wins = 0
            turns_list: List[int] = []

            # Build a small pool of opponent combinations using permutations and sampling
            triplets = build_opponent_combinations(self.baselines, self.n_games, setup)

            # Evaluate games for this setup
            for opp in triplets:
                # Work directly with the underlying base env for precise control
                base_env: LudoRLEnv = self.eval_env.envs[0]
                # Force the player-count for this eval run and the opponents
                obs, _ = base_env.reset(
                    options={"opponents": opp, "fixed_num_players": setup}
                )

                done = False
                total_turns = 0
                episode_reward = 0.0
                # Accumulate per-step reward breakdown into episode totals
                episode_breakdown: dict = {}
                while not done:
                    # Build action mask from pending valid moves
                    action_masks = MoveUtils.get_action_mask_for_env(base_env)

                    action, _ = self.model.predict(
                        obs, deterministic=True, action_masks=action_masks
                    )
                    # Step base env directly and keep obs normalized via VecNormalize
                    next_obs, reward, terminated, truncated, info = base_env.step(
                        int(action)
                    )
                    # accumulate step-level breakdown if present
                    br = (
                        info.get("reward_breakdown") if isinstance(info, dict) else None
                    )
                    if br:
                        for k, v in br.items():
                            episode_breakdown[k] = episode_breakdown.get(
                                k, 0.0
                            ) + float(v)
                    episode_reward += float(reward)
                    obs = self.eval_env.normalize_obs(next_obs)
                    total_turns += 1
                    done = bool(terminated or truncated)
                    if done:
                        try:
                            won = (
                                base_env.game.winner is not None
                                and base_env.game.game_over
                                and base_env.game.winner.color == base_env.agent_color
                            )
                        except Exception:
                            won = reward > 0
                        wins += 1 if won else 0
                        turns_list.append(total_turns)
                        # Collect for overall logging
                        episode_rewards.append(episode_reward)
                        episode_lengths.append(total_turns)
                        successes.append(1 if won else 0)
                        # Pop reward_breakdown from info before building StepInfo
                        if isinstance(info, dict) and "reward_breakdown" in info:
                            info.pop("reward_breakdown")
                        # Convert info dict to StepInfo dataclass for type safety
                        step_info = StepInfo(**info)

                        # Setup-specific prefix for TensorBoard keys
                        prefix = f"{self.log_prefix}setup_{setup}/"

                        self._log_episode(step_info, episode_reward, episode_breakdown, prefix)

            win_rate = wins / float(self.n_games)
            avg_turns = float(np.mean(turns_list)) if turns_list else 0.0

            self._log_setup_results(setup, win_rate, avg_turns)

        return episode_rewards, episode_lengths, successes
