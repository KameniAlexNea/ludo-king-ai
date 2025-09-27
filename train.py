

import copy
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
from ludo_rl.callbacks.hybrid_switch import HybridSwitchCallback
from ludo_rl.config import EnvConfig, TrainConfig
from ludo_rl.ludo_env.ludo_env import LudoRLEnv
from ludo_rl.ludo_env.ludo_env_selfplay import LudoRLEnvSelfPlay
from ludo_rl.ludo_env.ludo_env_hybrid import LudoRLEnvHybrid
from ludo_rl.trains.imitation import collect_imitation_samples, imitation_train
from ludo_rl.trains.training_args import parse_args
from ludo_rl.utils.move_utils import MoveUtils
from loguru import logger


def make_env(rank: int, seed: int, base_cfg: EnvConfig, env_type: str = "classic"):
    def _init():
        cfg = copy.deepcopy(base_cfg)
        cfg.seed = seed + rank
        if env_type == "selfplay":
            env = LudoRLEnvSelfPlay(cfg)
        elif env_type == "hybrid":
            env = LudoRLEnvHybrid(cfg)
        else:
            env = LudoRLEnv(cfg)
        # Return raw env here. The caller will wrap with ActionMasker only for
        # single-process environments (DummyVecEnv). When using SubprocVecEnv
        # we rely on the env.action_masks() API implemented on the envs so
        # MaskablePPO can access masks across subprocess boundaries.
        return env

    return _init


def main():
    args: TrainConfig = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    env_cfg = EnvConfig(max_turns=args.max_turns)
    
    # For selfplay and hybrid, force 4 players since they require specific opponent setups
    if args.env_type in ["selfplay", "hybrid"]:
        env_cfg.fixed_num_players = 4

    if args.n_envs == 1:
        venv = DummyVecEnv([lambda: ActionMasker(make_env(0, 42, env_cfg, args.env_type)(), MoveUtils.get_action_mask_for_env)])
    else:
        venv = SubprocVecEnv(
            [
                make_env(i, 42 + i * 100, env_cfg, args.env_type)
                for i in range(args.n_envs)
            ]
        )

    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_reward=True)

    # Separate eval env with same wrappers (always classic for evaluation vs baselines)
    # For evaluation we prefer single-process env for deterministic mask wrapping
    eval_raw = make_env(999, 1337, env_cfg, "classic")()
    eval_env = DummyVecEnv([lambda: ActionMasker(eval_raw, MoveUtils.get_action_mask_for_env)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True)

    # Set up learning rate (use callable for annealing)
    if args.lr_anneal_enabled:
        def lr_schedule(progress_remaining: float) -> float:
            return args.lr_final + progress_remaining * (args.learning_rate - args.lr_final)
        learning_rate = lr_schedule
    else:
        learning_rate = args.learning_rate

    # Optionally use a custom feature extractor when discrete observations are enabled
    policy_kwargs = {}
    try:
        if getattr(env_cfg, "obs", None) and getattr(env_cfg.obs, "discrete", False):
            from ludo_rl.features.multidiscrete_extractor import MultiDiscreteFeatureExtractor

            policy_kwargs = {
                "features_extractor_class": MultiDiscreteFeatureExtractor,
                "features_extractor_kwargs": {"embed_dim": args.embed_dim},
            }
    except Exception:
        # If feature extractor import fails, fall back to default
        policy_kwargs = {}

    model = MaskablePPO(
        "MlpPolicy",
        venv,
        learning_rate=learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        tensorboard_log=args.logdir,
        verbose=1,
        device="auto",
        gamma=0.995,
        policy_kwargs=policy_kwargs,
    )

    # When using selfplay or hybrid, inject the live model into envs so they can snapshot policy at reset
    if args.env_type in ["selfplay", "hybrid"]:
        try:
            venv.env_method("set_model", model)
            venv.env_method("set_obs_normalizer", venv)
        except Exception as e:
            raise RuntimeError(f"Failed to inject model and obs_normalizer into environments for {args.env_type} training: {e}") from e

    progress_cb = ProgressCallback(total_timesteps=args.total_steps, update_freq=10_000)
    eval_cb = SimpleBaselineEvalCallback(
        baselines=[s.strip() for s in args.eval_baselines.split(",") if s.strip()],
        n_games=args.eval_games,
        eval_freq=args.eval_freq,
        env_cfg=env_cfg,
        verbose=1,
        eval_env=eval_env,
        best_model_save_path=args.model_dir,
    )
    callbacks = [progress_cb, eval_cb]

    # Annealing callback (entropy + capture scale + learning rate). We'll wrap learning rate logic here by updating optimizer lr.
    if args.use_entropy_annealing:
        logger.info("[Annealing] Using entropy annealing during training.")
        anneal_cb = AnnealingCallback(args)
        callbacks.append(anneal_cb)

    # Hybrid switch callback if using hybrid env
    if args.env_type == "hybrid":
        switch_step = int(args.total_steps * args.hybrid_switch_rate)  # Switch halfway through training
        hybrid_cb = HybridSwitchCallback(switch_step, verbose=1)
        callbacks.append(hybrid_cb)

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
        # Normalize observations to match training data distribution
        obs_s = np.array([eval_env.normalize_obs(obs) for obs in obs_s])
        # Multi-seat samples
        obs_m, act_m, mask_m = collect_imitation_samples(
            base_env_for_imitation,
            strat_list,
            steps_budget=args.imitation_steps,
            multi_seat=True,
        )
        # Normalize observations to match training data distribution
        obs_m = np.array([eval_env.normalize_obs(obs) for obs in obs_m])
        obs_all = np.concatenate([obs_s, obs_m], axis=0)
        act_all = np.concatenate([act_s, act_m], axis=0)
        mask_all = np.concatenate([mask_s, mask_m], axis=0)
        dataset = TensorDataset(
            torch.from_numpy(obs_all),
            torch.from_numpy(act_all),
            torch.from_numpy(mask_all),
        )
        # Temporary entropy boost
        original_ent = float(model.ent_coef)
        boosted_ent = original_ent + args.imitation_entropy_boost
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
        model.ent_coef = original_ent
        logger.info("[Imitation] Completed pretraining phase.")
        # After imitation, run a quick evaluation callback manually (one pass) to log baseline performance under TB
        try:
            eval_cb.model = model  # Set the model on the callback
            eval_cb.on_step()  # type: ignore
        except Exception as e:
            logger.warning(f"[Imitation] Failed to run post-imitation evaluation: {e}")
            # Non-critical, continue training
        # Save post-imitation snapshot for curve comparison
        imitation_path = os.path.join(
            args.model_dir, "maskable_ppo_after_imitation"
        )
        try:
            model.save(imitation_path)
            logger.info(f"[Imitation] Saved post-imitation model to {imitation_path}.zip")
        except Exception as e:
            raise RuntimeError(f"[Imitation] Failed to save post-imitation model to {imitation_path}: {e}") from e

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

    model.learn(total_timesteps=args.total_steps, callback=callbacks)
    
    final_model_path = os.path.join(args.model_dir, "maskable_ppo_ludo_rl_final")
    try:
        model.save(final_model_path)
        logger.info(f"Training completed. Final model saved to {final_model_path}.zip")
    except Exception as e:
        raise RuntimeError(f"Failed to save final model to {final_model_path}: {e}") from e


if __name__ == "__main__":
    main()
