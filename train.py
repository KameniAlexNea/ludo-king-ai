import os
import time

import torch
from loguru import logger
# We must use MaskablePPO from sb3_contrib to handle action masking
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import \
    MaskableEvalCallback as EvalCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecMonitor)

from ludo_rl.extractor import LudoCnnExtractor
from ludo_rl.ludo_env import LudoEnv
from src.config import net_config

# --- Main Training Script ---

if __name__ == "__main__":
    # --- Setup ---
    # Create directories for logs and models
    BASE_DIR = "training"
    LOG_DIR = f"{BASE_DIR}/ludo_logs"
    MODEL_DIR = f"{BASE_DIR}/ludo_models"

    # Unique timestamp for this training run
    run_id = f"ppo_ludo_{int(time.time())}"
    model_save_path = os.path.join(MODEL_DIR, run_id)
    log_path = os.path.join(LOG_DIR, run_id)

    os.makedirs(model_save_path)
    os.makedirs(log_path)

    logger.debug("--- Initializing Environment ---")

    # --- Create Training Environment ---
    # We use a lambda to create the environment
    # Vectorize the environment
    train_env = SubprocVecEnv([LudoEnv for _ in range(os.cpu_count() // 2)])
    train_env = VecMonitor(train_env)

    # --- Create Evaluation Environment ---
    eval_env = LudoEnv()  # Set to human for render
    eval_env = DummyVecEnv([LudoEnv])
    eval_env = VecMonitor(eval_env)

    logger.debug("--- Setting up Callbacks ---")

    # --- Callbacks ---
    # Save a checkpoint every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000, save_path=model_save_path, name_prefix="ludo_model"
    )

    # Evaluate the model every 20,000 steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_save_path, "best_model"),
        log_path=log_path,
        eval_freq=20_000,
        deterministic=True,
        render=False,  # Set to True if you want to watch the eval
    )

    callback_list = CallbackList([checkpoint_callback, eval_callback])

    # --- Policy Kwargs ---
    # Define the custom feature extractor
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        features_extractor_class=LudoCnnExtractor,
        features_extractor_kwargs=dict(
            features_dim=net_config.embed_dim
        ),  # Output features
        net_arch=dict(pi=net_config.pi, vf=net_config.vf),  # Actor/Critic network sizes
        share_features_extractor=True
    )

    logger.debug("--- Initializing PPO Model ---")

    # --- Initialize Model ---
    # We MUST use MaskablePPO from sb3_contrib
    model = MaskablePPO(
        "MultiInputPolicy",  # Use MlpPolicy as our extractor outputs a flat vector
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_path,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.1,
        device="cpu",  # "cuda" if available, else "cpu"
    )

    final_model_path = os.path.join(model_save_path, "init_model")
    logger.info(f"--- Training Started. Saving initial model to {final_model_path} ---")
    model.save(final_model_path)

    logger.info(f"--- Starting Training ({run_id}) ---")

    # --- Train the Model ---
    model.learn(total_timesteps=1_000_000, callback=callback_list)

    # --- Save the Final Model ---
    final_model_path = os.path.join(model_save_path, "final_model")
    logger.info(f"--- Training Complete. Saving final model to {final_model_path} ---")
    model.save(final_model_path)

    # --- Load and Test the Trained Model ---
    logger.info("\n--- Testing Trained Model ---")
    del model  # Remove model from memory

    model = MaskablePPO.load(final_model_path)

    test_env = LudoEnv(render_mode="human")
    test_env = DummyVecEnv([lambda: test_env])
    # test_env = MaskableListActions(test_env) # Don't forget to wrap!

    obs, info = test_env.reset()
    for _ in range(500):
        # We need to provide the action mask for deterministic prediction
        action_masks = test_env.env_method("get_action_masks")

        action, _states = model.predict(
            obs, action_masks=action_masks[0], deterministic=True
        )

        obs, reward, terminated, truncated, info = test_env.step(action)

        if terminated or truncated:
            logger.info("Test episode finished. Resetting.")
            obs, info = test_env.reset()

    test_env.close()
    logger.info("--- Test Complete ---")
