from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from ludo_rl.config import EnvConfig
from ludo_rl.ludo_env.ludo_env import LudoRLEnv
from ludo_rl.utils.move_utils import MoveUtils
from ludo_rl.utils.opponents import build_opponent_triplets


class SimpleBaselineEvalCallback(BaseCallback):
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
    ):
        super().__init__(verbose=verbose)
        self.baselines = list(baselines)
        self.n_games = int(n_games)
        self.eval_freq = int(eval_freq)
        self.log_prefix = log_prefix.rstrip("/") + "/"
        self.env_cfg = env_cfg or EnvConfig()

        # Build eval env (1 process) and wrap
        def _make_eval():
            return LudoRLEnv(self.env_cfg)

        self.eval_env = DummyVecEnv([_make_eval])
        self.eval_env = VecMonitor(self.eval_env)
        # We'll set VecNormalize and tie obs_rms in _on_training_start
        self.eval_env = VecNormalize(
            self.eval_env, training=False, norm_obs=True, norm_reward=False
        )

    def _on_training_start(self) -> None:
        # Share normalization stats if training env has them
        try:
            if hasattr(self.model.env, "obs_rms"):
                self.eval_env.obs_rms = self.model.env.obs_rms
        except Exception:
            pass

    def _on_step(self) -> bool:
        # Evaluate every eval_freq steps
        if self.eval_freq <= 0:
            return True
        if self.num_timesteps == 0 or (self.num_timesteps % self.eval_freq) != 0:
            return True
        try:
            self._run_eval()
        except Exception as e:
            if self.verbose:
                print(f"[Eval] Error during evaluation: {e}")
        return True

    def _run_eval(self):
        wins = 0
        turns_list: List[int] = []
        total_offensive = 0  # tokens the agent captured
        total_defensive = 0  # times agent got captured
        total_finished_tokens = 0
        cumulative_reward = 0.0

        # Build a small pool of opponent triplets using permutations and sampling
        triplets = build_opponent_triplets(self.baselines, self.n_games)

        # Share obs_rms again in case it changed
        try:
            if hasattr(self.model.env, "obs_rms"):
                self.eval_env.obs_rms = self.model.env.obs_rms
        except Exception:
            pass

        # Evaluate games
        for opp in triplets:
            # Work directly with the underlying base env for precise control
            base_env: LudoRLEnv = self.eval_env.envs[0]
            obs, _ = base_env.reset(options={"opponents": opp})
            # Normalize initial obs using shared VecNormalize stats
            obs = self.eval_env.normalize_obs(obs)

            done = False
            total_turns = 0
            episode_reward = 0.0
            while not done:
                # Build action mask from pending valid moves
                action_masks = MoveUtils.get_action_mask_for_env(base_env)

                action, _ = self.model.predict(
                    obs, deterministic=False, action_masks=action_masks
                )
                # Step base env directly and keep obs normalized via VecNormalize
                next_obs, reward, terminated, truncated, info = base_env.step(
                    int(action)
                )
                episode_reward += float(reward)
                obs = self.eval_env.normalize_obs(next_obs)
                total_turns += 1
                done = bool(terminated or truncated)
                if done:
                    try:
                        won = (
                            base_env.game.game_over
                            and base_env.game.winner == base_env.agent_color
                        )
                    except Exception:
                        won = reward > 0
                    wins += 1 if won else 0
                    turns_list.append(total_turns)
                    # Aggregate stats from final info
                    # Use cumulative episode stats if provided (fallback to last-step stats)
                    total_offensive += int(
                        info.get(
                            "episode_captured_opponents",
                            info.get("captured_opponents", 0),
                        )
                    )
                    total_defensive += int(
                        info.get(
                            "episode_captured_by_opponents",
                            info.get("captured_by_opponents", 0),
                        )
                    )
                    total_finished_tokens += int(info.get("finished_tokens", 0))
                    cumulative_reward += episode_reward

        win_rate = wins / float(self.n_games)
        avg_turns = float(np.mean(turns_list)) if turns_list else 0.0
        avg_offensive = total_offensive / float(self.n_games)
        avg_defensive = total_defensive / float(self.n_games)
        avg_finished_tokens = total_finished_tokens / float(self.n_games)
        avg_reward = cumulative_reward / float(self.n_games)
        # Log to TB if available
        try:
            if hasattr(self, "logger") and self.logger is not None:
                self.logger.record(self.log_prefix + "win_rate", win_rate)
                self.logger.record(self.log_prefix + "avg_turns", avg_turns)
                self.logger.record(
                    self.log_prefix + "avg_offensive_captures", avg_offensive
                )
                self.logger.record(
                    self.log_prefix + "avg_defensive_captures", avg_defensive
                )
                self.logger.record(
                    self.log_prefix + "avg_finished_tokens", avg_finished_tokens
                )
                self.logger.record(self.log_prefix + "avg_episode_reward", avg_reward)
        except Exception:
            pass
        if self.verbose:
            logger.info(
                f"[Eval] win_rate={win_rate:.3f} avg_turns={avg_turns:.1f} off_cap={avg_offensive:.2f} def_cap={avg_defensive:.2f} fin_tokens={avg_finished_tokens:.2f} avg_reward={avg_reward:.2f} over {self.n_games} games"
            )
