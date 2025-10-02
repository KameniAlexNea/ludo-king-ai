"""Small smoke experiment: discrete observations + conservative PPO hyperparams.

This script runs a short MaskablePPO training (20k steps) on a single-process
environment with discrete observations and VecNormalize(norm_reward=True).
Use it to quickly verify whether the discrete pipeline and feature extractor
work and to collect basic training metrics.
"""
import os
import tempfile
from datetime import datetime

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from ludo_rl.config import EnvConfig, TrainConfig
from ludo_rl.ludo_env.ludo_env import LudoRLEnv
from ludo_rl.utils.move_utils import MoveUtils


def make_env(cfg: EnvConfig):
    def _init():
        env = LudoRLEnv(cfg)
        return ActionMasker(env, MoveUtils.get_action_mask_for_env)

    return _init


def main():
    # Conservative, deterministic smoke config
    env_cfg = EnvConfig()
    env_cfg.obs.discrete = True
    env_cfg.randomize_agent = False
    env_cfg.fixed_num_players = 2

    # Single-process deterministic env for quick debugging
    venv = DummyVecEnv([make_env(env_cfg)])
    venv = VecMonitor(venv)
    # Use the discrete extractor if present (train.py uses the same logic)
    policy_kwargs = {}
    try:
        if getattr(env_cfg, "obs", None) and getattr(env_cfg.obs, "discrete", False):
            from ludo_rl.features.multidiscrete_extractor import MultiDiscreteFeatureExtractor

            policy_kwargs = {
                "features_extractor_class": MultiDiscreteFeatureExtractor,
                "features_extractor_kwargs": {"embed_dim": 32},
            }
    except Exception as e:
        print("Failed to import MultiDiscreteFeatureExtractor:", e)

    # Conservative hyperparameters
    model = MaskablePPO(
        "MlpPolicy",
        venv,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=256,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1,
        device="auto",
        policy_kwargs=policy_kwargs,
    )

    # Run a short training for smoke testing
    total_steps = 20_000
    print(f"Starting smoke run: {total_steps} steps at {datetime.now()}")
    model.learn(total_timesteps=total_steps)

    # Save model + vecnormalize for inspection
    out_dir = os.path.join("./training", "smoke_test")
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "maskable_ppo_smoke_discrete")
    model.save(model_path)
    try:
        # save VecNormalize statistics
        venv.save(os.path.join(out_dir, "vecnormalize_smoke.pkl"))
    except Exception:
        pass

    print(f"Smoke run finished. Model saved to {model_path}.zip")


if __name__ == "__main__":
    main()
