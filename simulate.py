import time

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ludo_rl.ludo.config import config
from ludo_rl.ludo_env import LudoEnv

print("--- Initializing Test Environment ---")
print(f"Max game turns set to: {config.MAX_TURNS}")


def create_env():
    # Set render_mode="human" to see the print() statements
    env = LudoEnv(render_mode="human")
    return env


# Vectorize the environment
env = DummyVecEnv([create_env])
model = MaskablePPO.load(
    "training/ludo_models/ppo_ludo_1762131743/best_model/best_model.zip"
)

print("\n--- Starting Random Game Simulation ---")
start_time = time.time()

# Reset the environment
obs = env.reset()
episode_reward = 0
step_count = 0

# Main game loop
while True:
    step_count += 1

    # Get the action mask for the first (and only) environment
    action_masks = env.env_method("action_masks")[0]

    # Get the indices of all valid actions
    valid_actions = np.where(action_masks)[0]

    if len(valid_actions) == 0:
        # This error should be impossible now, as ludo_env.py
        # handles skipped turns internally.
        print(f"CRITICAL ERROR: Step {step_count} - No valid actions available.")
        print("This should not happen. Check ludo_env.py's reset() and step() loops.")
        break

    # Choose a random action from the valid set
    action, _ = model.predict(obs, action_masks=action_masks)

    # Take the step
    obs, rewards, dones, infos = env.step([action.item()])

    # Extract results for the first env
    reward = rewards[0]
    terminated = dones[0]  # 'dones' is True if the env terminated (agent won)

    # --- THIS IS THE TRUNCATED FIX ---
    # Your custom truncation logic is in info["truncated"]
    # NOT info["TimeLimit.truncated"]
    truncated = infos[0].get("truncated", False)
    # ---

    episode_reward += reward

    print(f"Step {step_count}, Action: {action.item()}, Reward: {reward:.2f}")

    # Check if the episode is over
    if terminated or truncated:
        print("\n--- SIMULATION COMPLETE ---")

        end_time = time.time()
        print(f"Total Steps: {step_count}")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"Simulation Time: {end_time - start_time:.2f} seconds")
        print("Info related to final rank:", infos[0].get("final_rank", "N/A"))
        break

# Close the environment
env.close()
