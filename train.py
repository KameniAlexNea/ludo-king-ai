from __future__ import annotations

import copy
import os

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from torch.utils.data import TensorDataset

from ludo_rl.callbacks.annealing import AnnealingCallback
from ludo_rl.callbacks.curriculum import ProgressCallback
from ludo_rl.callbacks.eval_baselines import SimpleBaselineEvalCallback
from ludo_rl.config import EnvConfig, TrainConfig
from ludo_rl.ludo_env.ludo_env import LudoRLEnv
from ludo_rl.ludo_env.ludo_env_selfplay import LudoRLEnvSelfPlay
from ludo_rl.trains.imitation import collect_imitation_samples, imitation_train
from ludo_rl.trains.lr_utils import apply_linear_lr
from ludo_rl.trains.training_args import TrainingArgs, parse_args
from ludo_rl.utils.move_utils import MoveUtils
from loguru import logger


def make_env(rank: int, seed: int, base_cfg: EnvConfig, env_type: str = "classic"):
    def _init():
        cfg = copy.deepcopy(base_cfg)
        cfg.seed = seed + rank
        if env_type == "selfplay":
            env = LudoRLEnvSelfPlay(cfg)
        else:
            env = LudoRLEnv(cfg)
        return ActionMasker(env, MoveUtils.get_action_mask_for_env)

    return _init


def _maybe_log_anneal(
    step: int, freq: int, model, lr_val: float, train_cfg: TrainConfig
):
    if freq <= 0:
        return
    if step % freq == 0:
        try:
            ent = getattr(model, "ent_coef", None)
            logger.info(
                f"[Anneal] step={step} lr={lr_val:.6g} ent={ent} capture_scale={train_cfg.capture_scale_initial}->{train_cfg.capture_scale_final}"
            )
        except Exception:
            pass


def main():
    args: TrainingArgs = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    env_cfg = EnvConfig(max_turns=args.max_turns)

    if args.n_envs == 1:
        venv = DummyVecEnv([make_env(0, 42, env_cfg, args.env_type)])
    else:
        venv = SubprocVecEnv(
            [
                make_env(i, 42 + i * 100, env_cfg, args.env_type)
                for i in range(args.n_envs)
            ]
        )

    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_reward=False)

    # Separate eval env with same wrappers (always classic for evaluation vs baselines)
    eval_env = DummyVecEnv([make_env(999, 1337, env_cfg, "classic")])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)
    try:
        eval_env.obs_rms = venv.obs_rms
    except Exception:
        pass

    model = MaskablePPO(
        "MlpPolicy",
        venv,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        tensorboard_log=args.logdir,
        verbose=1,
        device="auto",
    )

    # When using selfplay, inject the live model into envs so they can snapshot policy at reset
    if args.env_type == "selfplay":
        try:
            venv.env_method("set_model", model)
        except Exception:
            # Some VecEnv types may require accessing the underlying attribute
            pass

    progress_cb = ProgressCallback(total_timesteps=args.total_steps, update_freq=10_000)
    eval_cb = SimpleBaselineEvalCallback(
        baselines=[s.strip() for s in args.eval_baselines.split(",") if s.strip()],
        n_games=args.eval_games,
        eval_freq=args.eval_freq,
        env_cfg=env_cfg,
        verbose=1,
    )
    callbacks = [progress_cb, eval_cb]

    # Annealing callback (entropy + capture scale + learning rate). We'll wrap learning rate logic here by updating optimizer lr.
    train_cfg = TrainConfig(
        total_steps=args.total_steps,
        n_envs=args.n_envs,
        eval_freq=args.eval_freq,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        entropy_coef_initial=args.entropy_coef_initial,
        entropy_coef_final=args.entropy_coef_final,
        entropy_anneal_steps=args.entropy_anneal_steps,
        capture_scale_initial=args.capture_scale_initial,
        capture_scale_final=args.capture_scale_final,
        capture_scale_anneal_steps=args.capture_scale_anneal_steps,
    )
    anneal_cb = AnnealingCallback(train_cfg)
    callbacks.append(anneal_cb)

    # Optional imitation kickstart
    if args.imitation_enabled:
        logger.info("[Imitation] Collecting scripted policy samples...")
        strat_list = [
            s.strip() for s in args.imitation_strategies.split(",") if s.strip()
        ]
        # Single-seat samples
        base_env_for_imitation = LudoRLEnv(env_cfg)
        obs_s, act_s, mask_s = collect_imitation_samples(
            base_env_for_imitation,
            strat_list,
            steps_budget=args.imitation_steps,
            multi_seat=False,
        )
        # Multi-seat samples
        obs_m, act_m, mask_m = collect_imitation_samples(
            base_env_for_imitation,
            strat_list,
            steps_budget=args.imitation_steps,
            multi_seat=True,
        )
        obs_all = np.concatenate([obs_s, obs_m], axis=0)
        act_all = np.concatenate([act_s, act_m], axis=0)
        mask_all = np.concatenate([mask_s, mask_m], axis=0)
        dataset = TensorDataset(
            torch.from_numpy(obs_all),
            torch.from_numpy(act_all),
            torch.from_numpy(mask_all),
        )
        # Temporary entropy boost
        original_ent = (
            model.ent_coef if isinstance(model.ent_coef, float) else args.ent_coef
        )
        boosted_ent = original_ent + args.imitation_entropy_boost
        if isinstance(model.ent_coef, float):
            model.ent_coef = boosted_ent
        logger.info(
            f"[Imitation] Training on {len(dataset)} samples (single+multi-seat) for {args.imitation_epochs} epochs"
        )
        imitation_train(
            model,
            dataset,
            epochs=args.imitation_epochs,
            batch_size=args.imitation_batch_size,
        )
        # Restore entropy coef (annealing callback will handle future schedule)
        if isinstance(model.ent_coef, float):
            model.ent_coef = original_ent
        logger.info("[Imitation] Completed pretraining phase.")
        # After imitation, run a quick evaluation callback manually (one pass) to log baseline performance under TB
        try:
            eval_cb._run_eval()  # type: ignore
        except Exception:
            pass
        # Save post-imitation snapshot for curve comparison
        try:
            imitation_path = os.path.join(
                args.model_dir, "maskable_ppo_after_imitation"
            )
            model.save(imitation_path)
            logger.info(f"[Imitation] Saved post-imitation model to {imitation_path}.zip")
        except Exception as e:
            logger.info(f"[Imitation] Warning: could not save post-imitation model: {e}")

    # Add checkpointing if enabled
    if args.checkpoint_freq and args.checkpoint_freq > 0:
        ckpt_cb = CheckpointCallback(
            save_freq=args.checkpoint_freq // args.n_envs,
            save_path=args.model_dir,
            name_prefix=args.checkpoint_prefix,
            save_replay_buffer=True,
            save_vecnormalize=True,
            verbose=1,
        )
        callbacks.append(ckpt_cb)

    # Wrap learn with manual LR annealing by hooking into progress_cb logic via custom loop if needed.
    # Simpler: monkey patch progress callback to also adjust LR based on num_timesteps.
    if args.lr_anneal_enabled:
        initial_lr = args.learning_rate
        final_lr = args.lr_final
        original_on_step = progress_cb._on_step

        def patched_on_step():
            frac = (
                model.num_timesteps / float(args.total_steps)
                if args.total_steps > 0
                else 1.0
            )
            lr_val = apply_linear_lr(model, initial_lr, final_lr, frac)
            _maybe_log_anneal(
                model.num_timesteps, args.anneal_log_freq, model, lr_val, train_cfg
            )
            return original_on_step()

        progress_cb._on_step = patched_on_step  # type: ignore

    model.learn(total_timesteps=args.total_steps, callback=callbacks)
    model.save(os.path.join(args.model_dir, "maskable_ppo_ludo_rl_final"))


if __name__ == "__main__":
    main()
