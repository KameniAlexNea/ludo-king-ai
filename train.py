import os
import time
from dataclasses import asdict

import torch
from loguru import logger
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from wandb.integration.sb3 import WandbCallback

import wandb
from ludo_rl.extractor import LudoCnnExtractor, LudoTransformerExtractor
from ludo_rl.ludo_env import LudoEnv
from ludo_rl.ludo_king.config import config, net_config
from ludo_rl.ludo_king.reward import reward_config
from tools.arguments import TrainingSetup, parse_train_args
from tools.scheduler import CoefScheduler, lr_schedule


class ProfilerStepCallback(BaseCallback):
    """Steps the PyTorch profiler once per environment step."""

    def __init__(self, profiler) -> None:
        super().__init__()
        self.profiler = profiler

    def _on_step(self) -> bool:
        self.profiler.step()
        return True


if __name__ == "__main__":
    # --- Setup ---
    # Create directories for logs and models
    args = parse_train_args()
    print(args)

    # Unique timestamp for this training run
    run_id = f"ppo_ludo_{int(time.time())}"
    model_save_path = os.path.join(args.model_dir, run_id)
    log_path = os.path.join(args.log_dir, run_id)

    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    logger.debug("--- Initializing Environment ---")

    # --- Create Training Environment ---
    # We use a lambda to create the environment
    # Vectorize the environment
    if args.num_envs == 1:
        train_env = DummyVecEnv([LudoEnv])
    else:
        train_env = SubprocVecEnv([LudoEnv for _ in range(args.num_envs)])
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True)

    logger.debug("--- Setting up Callbacks ---")

    # --- Callbacks ---
    # Save a checkpoint every xxx steps
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq // args.num_envs),
        save_path=model_save_path,
        name_prefix="ludo_model",
        save_vecnormalize=True,
    )

    entropy_callback = CoefScheduler(
        total_timesteps=args.total_timesteps,
        att="ent_coef",
        schedule=lr_schedule(lr_min=args.ent_coef * 0.3, lr_max=args.ent_coef),
    )

    callbacks = [entropy_callback]
    wandb.init(
        project="ludo-king-ppo",
        name=run_id,
        config=asdict(
            TrainingSetup(
                config=config,
                network_config=net_config,
                reward_config=reward_config,
                train_config=args,
            )
        ),
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    if not args.profile:
        callbacks.append(checkpoint_callback)
        callbacks.append(WandbCallback())

    # --- Policy Kwargs ---
    # Define the custom feature extractor
    policy_kwargs = dict(
        activation_fn=torch.nn.GELU,  # GELU for smooth, non-saturating gradients (best for transformers)
        features_extractor_class=(
            LudoTransformerExtractor if args.use_transformer else LudoCnnExtractor
        ),
        features_extractor_kwargs=dict(
            features_dim=net_config.embed_dim
        ),  # Output features
        net_arch=dict(pi=net_config.pi, vf=net_config.vf),  # Actor/Critic network sizes
        share_features_extractor=True,
    )

    logger.debug("--- Initializing PPO Model ---")

    # --- Initialize Model ---
    # We MUST use MaskablePPO from sb3_contrib
    init_kwargs = dict(
        verbose=1,
        tensorboard_log=log_path,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=lr_schedule(lr_min=0.15, lr_max=args.clip_range),
        ent_coef=args.ent_coef,
        device=args.device,
        learning_rate=lr_schedule(
            lr_min=args.learning_rate * 0.3, lr_max=args.learning_rate
        ),
        target_kl=args.target_kl,
        vf_coef=args.vf_coef,
    )
    if args.resume:
        logger.info(f"--- Resuming training from {args.resume} ---")
        model = MaskablePPO.load(
            args.resume,
            env=train_env,
            **init_kwargs,
        )
    else:
        model = MaskablePPO(
            "MultiInputPolicy",  # Use MlpPolicy as our extractor outputs a flat vector
            train_env,
            policy_kwargs=policy_kwargs,
            **init_kwargs,
        )

    print("--- Model Summary ---")
    print(model.policy)

    final_model_path = os.path.join(model_save_path, "init_model")
    logger.info(f"--- Training Started. Saving initial model to {final_model_path} ---")
    model.save(final_model_path)

    logger.info(f"--- Starting Training ({run_id}) ---")

    profiling_summary = None
    profiler_log_dir = None
    trace_path = None
    if args.profile:
        logger.info("--- Profiling Enabled: results will be stored alongside logs ---")
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available() and "cuda" in args.device:
            activities.append(ProfilerActivity.CUDA)

        profiler_log_dir = os.path.join(log_path, "profiler")
        os.makedirs(profiler_log_dir, exist_ok=True)

        profiler_schedule = schedule(wait=0, warmup=1, active=1, repeat=1)

        with profile(
            activities=activities,
            schedule=profiler_schedule,
            on_trace_ready=tensorboard_trace_handler(profiler_log_dir),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            profiling_callbacks = CallbackList(callbacks + [ProfilerStepCallback(prof)])
            model.learn(
                total_timesteps=args.total_timesteps,
                callback=profiling_callbacks,
            )
            profiling_summary = prof.key_averages().table(
                sort_by="cpu_time_total", row_limit=20
            )
        trace_path = os.path.join(profiler_log_dir, "trace.json")
        try:
            prof.export_chrome_trace(trace_path)
        except RuntimeError:
            logger.warning("Profiler trace already saved; skipping explicit export.")
        if profiling_summary is not None and profiler_log_dir is not None:
            logger.info("--- Profiling Summary (top 20 ops by CPU time) ---")
            logger.info("\n{}", profiling_summary)
            if trace_path is not None:
                logger.info("Profiler trace exported to {}", trace_path)
    else:
        # --- Train the Model ---
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=CallbackList(callbacks),
        )

    # --- Save the Final Model ---
    final_model_path = os.path.join(model_save_path, "final_model")
    logger.info(f"--- Training Complete. Saving final model to {final_model_path} ---")
    model.save(final_model_path)
