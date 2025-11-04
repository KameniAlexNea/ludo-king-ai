from __future__ import annotations

import argparse
import os
from dataclasses import dataclass


@dataclass
class TrainConfig:
    total_timesteps: int
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    num_envs: int
    log_dir: str
    model_dir: str
    device: str
    checkpoint_freq: int
    learning_rate: float


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ludo MaskablePPO training options")
    parser.add_argument("--total-timesteps", type=int, default=50_000_000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.1)
    parser.add_argument("--num-envs", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="training/ludo_logs")
    parser.add_argument("--model-dir", type=str, default="training/ludo_models")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-freq", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    return parser


def parse_train_args(args: list[str] | None = None) -> TrainConfig:
    parser = build_train_parser()
    namespace = parser.parse_args(args=args)

    num_envs = namespace.num_envs
    if num_envs <= 0:
        cpu_count = os.cpu_count() or 1
        num_envs = max(1, cpu_count // 2)

    return TrainConfig(
        total_timesteps=namespace.total_timesteps,
        n_steps=namespace.n_steps,
        batch_size=namespace.batch_size,
        n_epochs=namespace.n_epochs,
        gamma=namespace.gamma,
        gae_lambda=namespace.gae_lambda,
        clip_range=namespace.clip_range,
        ent_coef=namespace.ent_coef,
        num_envs=num_envs,
        log_dir=namespace.log_dir,
        model_dir=namespace.model_dir,
        device=namespace.device,
        checkpoint_freq=namespace.checkpoint_freq,
        learning_rate=namespace.learning_rate
    )
