import os
import time

import torch
from loguru import logger
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from ludo_rl.extractor import LudoCnnExtractor, LudoTransformerExtractor
from ludo_rl.ludo.config import net_config
from ludo_rl.ludo_env import LudoEnv
from tools.arguments import parse_train_args
from tools.scheduler import CoefScheduler, lr_schedule

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

    entropy_callback = CoefScheduler(
        total_timesteps=args.total_timesteps,
        att="ent_coef",
        schedule=lr_schedule(lr_min=0.005, lr_max=args.ent_coef),
    )

    callback_list = CallbackList([checkpoint_callback, entropy_callback])

    # --- Policy Kwargs ---
    # Define the custom feature extractor
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        features_extractor_class=(
            LudoTransformerExtractor if net_config.use_transformer else LudoCnnExtractor
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
            clip_range=lr_schedule(lr_min=0.15, lr_max=args.clip_range),
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
