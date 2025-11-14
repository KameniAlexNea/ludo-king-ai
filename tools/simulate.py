import argparse
import os
import random
import time
from typing import Optional

import numpy as np
from loguru import logger
from sb3_contrib import MaskablePPO

from ludo_rl.ludo_env import LudoEnv
from ludo_rl.ludo_king import config as king_config


def seed_environ(seed_value: Optional[int] = None):
    random.seed(seed_value)
    np.random.seed(seed_value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a trained Ludo agent using LudoEnv"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the MaskablePPO model zip file",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=int(os.getenv("NUM_PLAYERS", 4)),
        help="Number of players in the game",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy at inference",
    )
    return parser.parse_args()


def _dice_from_obs(obs: dict) -> int:
    # obs["dice_roll"] is in 0..5; convert back to 1..6 for display
    try:
        return int(obs["dice_roll"][0]) + 1
    except Exception as e:
        logger.warning(f"Failed to extract dice from observation, defaulting to 1: {e}")
        return 1


def main() -> None:
    args = parse_args()
    seed_environ(42)

    print("--- Initializing LudoEnv ---")
    # Apply requested number of players to runtime config
    try:
        king_config.NUM_PLAYERS = int(args.num_players)
    except Exception as e:
        logger.warning(
            f"Failed to set NUM_PLAYERS to {args.num_players}, using default: {e}"
        )

    env = LudoEnv()
    # Show configured opponents from env
    try:
        print("Opponents:")
        # Agent is seat 0; env chooses opponents internally
        print(f"  Configured list: {env.opponents}")
    except Exception as e:
        logger.warning(f"Failed to display opponent configuration: {e}")

    model = MaskablePPO.load(args.model_path)
    model.policy.set_training_mode(False)
    print(model.policy)

    print("\n--- Starting Simulation (LudoEnv) ---")
    start_time = time.time()

    # Reset env
    obs, info = env.reset()
    step_count = 0
    terminated = False
    truncated = False

    while not terminated and not truncated and env.current_turn < king_config.MAX_TURNS:
        mask = env.action_masks()
        if mask is None:
            mask = np.ones(king_config.PIECES_PER_PLAYER, dtype=bool)

        # Predict action using mask
        if np.any(mask):
            action, _ = model.predict(
                obs, action_masks=mask[None, ...], deterministic=args.deterministic
            )
            action = int(np.asarray(action).item())
        else:
            action = 0

        step_count += 1
        dice = _dice_from_obs(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"Step {step_count}, Agent piece {action}, Dice: {dice}, Reward: {reward:.2f}"
        )

    print("\n--- SIMULATION COMPLETE ---")
    end_time = time.time()
    print(f"Total Turns: {env.current_turn}")
    print(f"Total Steps: {step_count}")
    print(f"Simulation Time: {end_time - start_time:.2f} seconds")
    # final rank if available
    final_rank = info.get("final_rank") if isinstance(info, dict) else None
    if final_rank is not None:
        print(f"Final Rank: {final_rank}")


if __name__ == "__main__":
    main()
