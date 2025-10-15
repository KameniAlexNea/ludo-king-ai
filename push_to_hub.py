import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import copy
import math
from typing import Optional

import gymnasium as gym
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

from huggingface_sb3 import package_to_hub

import gymnasium as gym
from ludo_rl.config import EnvConfig
from ludo_rl.ludo_env.ludo_env_base import LudoRLEnvBase

class LudoRLEnvBaseRegistered(LudoRLEnvBase):
    def __init__(self):
        cfg = EnvConfig(max_turns=500)
        super().__init__(cfg)

env_id = "LudoRLEnvBase-v0"
model_path = "saved_states/models/14_10_2025/best_model.zip"
gym.register(
    id=env_id,
    entry_point=LudoRLEnvBaseRegistered,
)

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
        env.render_mode = "rgb_array"
        # gym.make()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.RecordVideo(env, "training/videos", episode_trigger=lambda t: t % 1 == 0)
        return env

    if seed is not None:
        set_random_seed(seed)
    return _init

env_cfg = EnvConfig(max_turns=500)

eval_env = make_env(999, env_cfg, 42, "classic")()
eval_env = DummyVecEnv([lambda: eval_env])
eval_env = VecMonitor(eval_env)
n_eval_episodes=1000

model = MaskablePPO.load(
	model_path,
	device="auto",
)

import json

# Monkey patch json.dump to automatically convert numpy types
original_dump = json.dump

def custom_dump(obj, fp, **kwargs):
    def convert(value):
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [convert(item) for item in value]
        elif hasattr(value, 'item'):  # numpy types
            return value.item()
        else:
            return value
    
    converted_obj = convert(obj)
    original_dump(converted_obj, fp, **kwargs)

json.dump = custom_dump

package_to_hub(
    model=model,
    model_name="ppo-ludo-king-ai",
    model_architecture="MaskablePPO",
    # model_architecture="PPO",
    env_id=env_id,
    eval_env=eval_env,
    repo_id="alexneakameni/ppo-ludo-king-ai",
    commit_message="PPO Ludo King AI",
	n_eval_episodes=n_eval_episodes,
    logs="saved_states/",
)