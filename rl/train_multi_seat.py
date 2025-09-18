"""Stable-Baselines3 training script for Ludo (Multi-Seat).

Usage (after installing requirements):
    python training/train_multi_seat.py --total-steps 2000000 --n-envs 8
    python training/train_multi_seat.py --load-model ./models/ppo_ludo_final.zip --total-steps 1000000

Produces models/, logs/ and tensorboard metrics.
"""

from __future__ import annotations

import argparse
import copy
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from .callbacks.progress_curriculum import ProgressCurriculumCallback
from .callbacks.tournament_callback import ClassicTournamentCallback
from .envs.ludo_env.ludo_env_multi_seat import EnvConfig, LudoGymEnv


def make_env(rank: int, seed: int, base_cfg: EnvConfig):
    def _init():
        # Deep copy to avoid shared nested dataclass instances (reward_cfg, obs_cfg, opponents)
        cfg = copy.deepcopy(base_cfg)
        cfg.seed = seed + rank
        env = LudoGymEnv(cfg)

        # Action mask getter compatible with sb3-contrib ActionMasker
        def mask_fn(env_inst):
            # Use the env's last computed valid moves to build mask
            pending = getattr(env_inst, "_pending_valid_moves", None)
            mask = env_inst.move_utils.action_masks(pending)
            return mask.astype(bool)  # Convert to boolean for ActionMasker

        # Only wrap if ActionMasker is available and algorithm requires it
        try:
            wrapped = ActionMasker(env, mask_fn)
            return wrapped
        except Exception:
            return env
        return env

    return _init


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--model-dir", type=str, default="./models")
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to pre-trained model to continue training from (e.g., ./models/ppo_ludo_final.zip)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=500,
        help="Max turns per episode (reduce to increase variance)",
    )
    parser.add_argument(
        "--no-probabilistic-rewards",
        action="store_true",
        help="Disable probabilistic reward shaping for debugging",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,  # Increased from 512 for better stability
        help="Number of steps per PPO update",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,  # Increased from 256 for better stability
        help="Minibatch size for PPO updates",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.1,  # Increased from 0.01 for better exploration
        help="Entropy coefficient for exploration (higher = more exploration)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,  # Reduced from 5e-4 for more stable learning
        help="Learning rate for PPO",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="maskable_ppo",
        choices=["ppo", "maskable_ppo", "ddpg"],
        help="RL algorithm to use (ppo, maskable_ppo or ddpg)",
    )
    parser.add_argument(
        "--tournament-freq",
        type=int,
        default=100_000,
        help="Run tournament evaluation every N environment timesteps",
    )
    parser.add_argument(
        "--tournament-games",
        type=int,
        default=240,
        help="Total tournament games per evaluation (evenly split across 4 seats)",
    )
    parser.add_argument(
        "--tournament-baselines",
        type=str,
        default="hybrid_prob,balanced,cautious,killer,probabilistic",
        help="Comma separated baseline strategy names (must be valid StrategyFactory names)",
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Disable graduated opponent curriculum (sample uniformly from candidates)",
    )
    return parser.parse_args()


def main(args):
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    base_cfg = EnvConfig(max_turns=args.max_turns)
    base_cfg.reward_cfg.use_probabilistic_rewards = not args.no_probabilistic_rewards

    # Configure opponent curriculum
    base_cfg.opponent_curriculum.enabled = not args.no_curriculum
    # Use default progress-based curriculum; no advanced CLI overrides

    if args.n_envs == 1:
        env_fns = [make_env(0, 42, base_cfg)]
        vec_env = DummyVecEnv(env_fns)
    else:
        # Use different seeds for each environment to add stochasticity
        env_fns = [make_env(i, 42 + i * 100, base_cfg) for i in range(args.n_envs)]
        vec_env = SubprocVecEnv(env_fns)

    vec_env = VecMonitor(vec_env, filename=os.path.join(args.logdir, "monitor.csv"))
    vec_env = VecNormalize(
        vec_env,
        norm_reward=False,
    )

    # Build evaluation env with identical wrapper stack (Monitor + VecNormalize) so
    # normalization stats can be synchronized without assertion errors.
    eval_env_raw = DummyVecEnv([make_env(999, 42, base_cfg)])
    eval_env_raw = VecMonitor(eval_env_raw)
    # For evaluation we typically want raw rewards; keep norm_reward False always
    eval_env = VecNormalize(
        eval_env_raw,
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    # Sync observation normalization stats from training env
    try:
        eval_env.obs_rms = copy.deepcopy(vec_env.obs_rms)
    except Exception:
        pass
    policy_kwargs = {
        "net_arch": dict(pi=[512, 256, 128], vf=[512, 256, 128])
    }  # Increased capacity
    if args.algorithm.lower() == "ppo":
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            learning_rate=args.learning_rate,  # Use CLI argument
            n_steps=args.n_steps,  # Use CLI argument
            batch_size=args.batch_size,  # Use CLI argument
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,  # Use CLI argument
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=args.logdir,
            device="auto",
            policy_kwargs=policy_kwargs,
        )
    elif args.algorithm.lower() == "maskable_ppo":
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            learning_rate=args.learning_rate,  # Use CLI argument
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=args.logdir,
            device="auto",
            policy_kwargs=policy_kwargs,
        )
    elif args.algorithm.lower() == "ddpg":
        model = DDPG(
            "MlpPolicy",
            vec_env,
            verbose=0,
            learning_rate=args.learning_rate,
            buffer_size=1_000_000,
            learning_starts=100,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            tensorboard_log=args.logdir,
            device="auto",
            policy_kwargs=policy_kwargs,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")

    # Load pre-trained model if specified
    if args.load_model:
        if os.path.exists(args.load_model):
            print(f"Loading pre-trained model from: {args.load_model}")
            try:
                if args.algorithm.lower() == "ppo":
                    model = PPO.load(args.load_model, env=vec_env)
                elif args.algorithm.lower() == "maskable_ppo":
                    model = MaskablePPO.load(args.load_model, env=vec_env)
                elif args.algorithm.lower() == "ddpg":
                    model = DDPG.load(args.load_model, env=vec_env)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Starting training from scratch...")
        else:
            print(f"Model path not found: {args.load_model}")
            print("Starting training from scratch...")

    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=args.model_dir,
        name_prefix=f"{args.algorithm.lower()}_ludo",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.model_dir,
        log_path=args.logdir,
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        deterministic=False,  # Use stochastic evaluation
        n_eval_episodes=100,  # More evaluation episodes
        verbose=1,
    )

    baseline_names = [
        s.strip() for s in args.tournament_baselines.split(",") if s.strip()
    ]

    # Build a normalization function using training VecNormalize statistics
    def _normalize_obs(obs_np):
        # VecNormalize expects batch shape [n_envs, obs_dim]; use single-env shape
        # and only normalize observations (reward=False)
        try:
            obs = np.asarray(obs_np, dtype=np.float32)
            obs = obs[None, :]  # add batch dim
            obs_norm = vec_env.normalize_obs(obs)
            return obs_norm[0]
        except Exception:
            return obs_np

    tournament_cb = ClassicTournamentCallback(
        baselines=baseline_names,
        n_games=args.tournament_games,
        eval_freq=args.tournament_freq,
        max_turns=args.max_turns,
        log_prefix="tournament/",
        verbose=1,
        normalize_obs_fn=_normalize_obs,
    )

    progress_cb = None
    if (
        base_cfg.opponent_curriculum.enabled
        and base_cfg.opponent_curriculum.use_progress
    ):
        # Update envs with progress every 10k steps (lightweight)
        progress_cb = ProgressCurriculumCallback(
            total_timesteps=args.total_steps, update_freq=10_000
        )

    callbacks = [checkpoint_cb, eval_cb, tournament_cb]
    if progress_cb:
        callbacks.append(progress_cb)

    model.learn(
        total_timesteps=args.total_steps,
        callback=callbacks,
    )
    model.save(os.path.join(args.model_dir, f"{args.algorithm.lower()}_ludo_final"))


if __name__ == "__main__":
    args = get_args()
    main(args)