import math
import os
import time
from typing import Callable

import torch
from loguru import logger

# We must use MaskablePPO from sb3_contrib to handle action masking
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from tools.arguments import parse_train_args
from ludo_rl.extractor import LudoCnnExtractor
from ludo_rl.ludo.config import net_config
from ludo_rl.ludo_env import LudoEnv


def lr_schedule(
    lr_min: float = 1e-5, lr_max: float = 3e-4, warmup_steps: float = 0.03
) -> Callable[[float], float]:
    lr_min, lr_max = min(lr_min, lr_max), max(lr_min, lr_max)

    def schedule(progress_remaining: float) -> float:
        progress = 1 - progress_remaining
        if progress < warmup_steps:
            return lr_min + (lr_max - lr_min) * (progress / warmup_steps)
        else:
            adjusted_progress = (progress - warmup_steps) / (1 - warmup_steps)
            return lr_min + 0.5 * (lr_max - lr_min) * (
                1 + math.cos(math.pi * adjusted_progress)
            )

    return schedule


# --- Main Training Script ---

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

    logger.debug("--- Setting up Callbacks ---")

    # --- Callbacks ---
    # Save a checkpoint every xxx steps
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq // args.num_envs),
        save_path=model_save_path,
        name_prefix="ludo_model",
        save_vecnormalize=True,
    )

    callback_list = CallbackList([checkpoint_callback])

    # --- Policy Kwargs ---
    # Define the custom feature extractor
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        features_extractor_class=LudoCnnExtractor,
        features_extractor_kwargs=dict(
            features_dim=net_config.embed_dim
        ),  # Output features
        net_arch=dict(pi=net_config.pi, vf=net_config.vf),  # Actor/Critic network sizes
        share_features_extractor=True,
    )

    logger.debug("--- Initializing PPO Model ---")

    # --- Initialize Model ---
    # We MUST use MaskablePPO from sb3_contrib
    if args.resume is not None:
        logger.info(f"--- Resuming training from {args.resume} ---")
        model = MaskablePPO.load(
            args.resume,
            env=train_env,
            device=args.device,
            learning_rate=lr_schedule(lr_max=args.learning_rate),
        )
    else:
        model = MaskablePPO(
            "MultiInputPolicy",  # Use MlpPolicy as our extractor outputs a flat vector
            train_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=log_path,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            device=args.device,
            learning_rate=lr_schedule(lr_max=args.learning_rate),
        )

    final_model_path = os.path.join(model_save_path, "init_model")
    logger.info(f"--- Training Started. Saving initial model to {final_model_path} ---")
    model.save(final_model_path)

    logger.info(f"--- Starting Training ({run_id}) ---")

    # --- Train the Model ---
    model.learn(total_timesteps=args.total_timesteps, callback=callback_list)

    # --- Save the Final Model ---
    final_model_path = os.path.join(model_save_path, "final_model")
    logger.info(f"--- Training Complete. Saving final model to {final_model_path} ---")
    model.save(final_model_path)

    # --- Load and Test the Trained Model ---
    logger.info("\n--- Testing Trained Model ---")
    del model  # Remove model from memory

    model = MaskablePPO.load(final_model_path)

    test_env = DummyVecEnv([lambda: LudoEnv(render_mode="human")])
    # test_env = MaskableListActions(test_env) # Don't forget to wrap!

    obs, info = test_env.reset()
    for _ in range(500):
        # We need to provide the action mask for deterministic prediction
        action_masks = test_env.env_method("action_masks")

        action, _states = model.predict(
            obs, action_masks=action_masks[0], deterministic=True
        )

        obs, reward, terminated, truncated, info = test_env.step(action)

        if terminated or truncated:
            logger.info("Test episode finished. Resetting.")
            obs, info = test_env.reset()

    test_env.close()
    logger.info("--- Test Complete ---")
