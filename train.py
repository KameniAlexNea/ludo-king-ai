# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import copy
import math
import os
from typing import Optional

import gymnasium as gym
import torch
from dotenv import load_dotenv
from loguru import logger
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from ludo_rl.callbacks.annealing import AnnealingCallback
from ludo_rl.callbacks.curriculum import ProgressCallback
from ludo_rl.callbacks.eval_baselines import SimpleBaselineEvalCallback
from ludo_rl.callbacks.hybrid_switch import HybridSwitchCallback
from ludo_rl.config import EnvConfig, TrainConfig
from ludo_rl.ludo_env.ludo_env import LudoRLEnv
from ludo_rl.ludo_env.ludo_env_hybrid import LudoRLEnvHybrid
from ludo_rl.ludo_env.ludo_env_selfplay import LudoRLEnvSelfPlay
from ludo_rl.trains.training_args import parse_args
from ludo_rl.utils.move_utils import MoveUtils

load_dotenv()


def make_env(
    rank: int,
    base_cfg: EnvConfig,
    seed: Optional[int] = None,
    env_type: str = "classic",
):
    def _init():
        cfg = copy.deepcopy(base_cfg)
        if seed is not None:
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

    if seed is not None:
        set_random_seed(seed)
    return _init


def main():
    args: TrainConfig = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    env_cfg = EnvConfig(max_turns=args.max_turns)
    logger.info(f"Config used: {env_cfg}")

    # For selfplay and hybrid, force 4 players since they require specific opponent setups
    if args.env_type in ["selfplay", "hybrid"]:
        env_cfg.fixed_num_players = 4

    if args.n_envs == 1:
        venv = DummyVecEnv(
            [
                lambda: ActionMasker(
                    make_env(0, env_cfg, None, args.env_type)(),
                    MoveUtils.get_action_mask_for_env,
                )
            ]
        )
    else:
        venv = SubprocVecEnv(
            [make_env(i, env_cfg, None, args.env_type) for i in range(args.n_envs)]
        )

    venv = VecMonitor(venv)
    # Do not normalize observations when using MultiDiscrete (discrete) encoding,
    # or when using Dict spaces (VecNormalize doesn't support Dict).
    norm_obs_flag = not getattr(env_cfg.obs, "discrete", False) and not isinstance(
        venv.observation_space, gym.spaces.Dict
    )
    logger.info(f"Observation normalization enabled: {norm_obs_flag}")
    if norm_obs_flag:
        venv = VecNormalize(
            venv,
            training=True,
            norm_obs=True,
            norm_reward=False,
            clip_obs=1.0,
            clip_reward=1000.0,
        )

    # Separate eval env with same wrappers (always classic for evaluation vs baselines)
    # For evaluation we prefer single-process env for deterministic mask wrapping
    eval_raw = make_env(999, env_cfg, 42, "classic")()
    eval_env = DummyVecEnv(
        [lambda: ActionMasker(eval_raw, MoveUtils.get_action_mask_for_env)]
    )
    eval_env = VecMonitor(eval_env)
    if norm_obs_flag:
        eval_env = VecNormalize(
            eval_env,
            training=False,
            norm_obs=True,
            norm_reward=False,
            clip_obs=1.0,
            clip_reward=1000.0,
        )
    # Share normalization statistics with the training env so evaluation matches training distribution
    if norm_obs_flag:
        try:
            eval_env.obs_rms = venv.obs_rms
            eval_env.ret_rms = venv.ret_rms
        except Exception:
            pass

    # Set up learning rate (use callable for annealing)
    if args.lr_anneal_enabled:
        lr_change = 300_000 / args.total_steps

        def lr_schedule(progress_remaining: float) -> float:
            lr_min = args.lr_final
            lr_max = args.learning_rate
            progress = 1 - progress_remaining
            if progress < lr_change:
                return 1e-6 + (lr_max - 1e-6) * (progress / lr_change)
            else:
                adjusted_progress = (progress - lr_change) / (1 - lr_change)
                return lr_max + 0.5 * (lr_max - lr_min) * (
                    1 + math.cos(math.pi * adjusted_progress)
                )

        learning_rate = lr_schedule
    else:
        learning_rate = args.learning_rate

    # Optionally use a custom feature extractor when discrete observations are enabled
    policy_kwargs = {
        "net_arch": [512, 256, 128],
        "activation_fn": torch.nn.LeakyReLU,
    }
    try:
        if env_cfg.obs.discrete:
            from ludo_rl.features.multidiscrete_extractor import (
                MultiDiscreteFeatureExtractor,
            )

            policy_kwargs.update(
                {
                    "features_extractor_class": MultiDiscreteFeatureExtractor,
                    "features_extractor_kwargs": {"embed_dim": args.embed_dim},
                }
            )
            logger.info(
                "Using MultiDiscreteFeatureExtractor for discrete observations."
            )
    except ImportError:
        # If feature extractor import fails, fall back to default
        logger.warning("Failed to import MultiDiscreteFeatureExtractor, using default.")
        policy_kwargs = {}

    if args.load_model:
        if not os.path.isfile(args.load_model):
            raise FileNotFoundError(
                f"Specified model to load not found: {args.load_model}"
            )
        try:
            logger.info(f"Loading model from {args.load_model}")
            model = MaskablePPO.load(
                args.load_model,
                env=venv,
                custom_objects={
                    "learning_rate": learning_rate,
                    "n_steps": args.n_steps,
                },
                device="auto",
            )
            # Update any changed hyperparameters
            model.ent_coef = args.ent_coef
            model.vf_coef = args.vf_coef
            model.batch_size = args.batch_size
            model.policy_kwargs = policy_kwargs
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from {args.load_model}: {e}"
            ) from e
    else:
        model = MaskablePPO(
            "MultiInputPolicy",
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
            raise RuntimeError(
                f"Failed to inject model and obs_normalizer into environments for {args.env_type} training: {e}"
            ) from e

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
        switch_step = int(
            args.total_steps * args.hybrid_switch_rate
        )  # Switch halfway through training
        hybrid_cb = HybridSwitchCallback(switch_step, verbose=1)
        callbacks.append(hybrid_cb)

    # Add checkpointing if enabled
    if args.save_freq and args.save_freq > 0:
        ckpt_cb = CheckpointCallback(
            save_freq=args.save_freq // args.n_envs,
            save_path=args.model_dir,
            name_prefix=args.checkpoint_prefix,
            save_replay_buffer=True,
            save_vecnormalize=True,
            verbose=1,
        )
        callbacks.append(ckpt_cb)

    final_model_path = os.path.join(args.model_dir, "maskable_ppo_ludo_rl_start")
    try:
        model.save(final_model_path)
        logger.info(f"Training started. Initial model saved to {final_model_path}.zip")
    except Exception as e:
        raise RuntimeError(
            f"Failed to save initial model to {final_model_path}: {e}"
        ) from e

    model.learn(total_timesteps=args.total_steps, callback=callbacks)

    final_model_path = os.path.join(args.model_dir, "maskable_ppo_ludo_rl_final")
    try:
        model.save(final_model_path)
        logger.info(f"Training completed. Final model saved to {final_model_path}.zip")
    except Exception as e:
        raise RuntimeError(
            f"Failed to save final model to {final_model_path}: {e}"
        ) from e


if __name__ == "__main__":
    main()
