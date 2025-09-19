import argparse
from dataclasses import dataclass

from ludo_rl.config import TrainConfig


@dataclass
class TrainingArgs:
    total_steps: int = TrainConfig.total_steps
    n_envs: int = TrainConfig.n_envs
    logdir: str = TrainConfig.logdir
    model_dir: str = TrainConfig.model_dir
    eval_freq: int = TrainConfig.eval_freq
    eval_games: int = TrainConfig.eval_games
    checkpoint_freq: int = 100_000
    checkpoint_prefix: str = "ppo_ludo"
    eval_baselines: str = TrainConfig.eval_baselines
    learning_rate: float = TrainConfig.learning_rate
    n_steps: int = TrainConfig.n_steps
    batch_size: int = TrainConfig.batch_size
    ent_coef: float = TrainConfig.ent_coef
    max_turns: int = TrainConfig.max_turns
    # Imitation / kickstart
    imitation_enabled: bool = False
    imitation_strategies: str = TrainConfig.imitation_strategies
    imitation_steps: int = TrainConfig.imitation_steps
    imitation_batch_size: int = TrainConfig.imitation_batch_size
    imitation_epochs: int = TrainConfig.imitation_epochs
    imitation_entropy_boost: float = TrainConfig.imitation_entropy_boost
    # Annealing overrides
    entropy_coef_initial: float = TrainConfig.entropy_coef_initial
    entropy_coef_final: float = TrainConfig.entropy_coef_final
    entropy_anneal_steps: int = TrainConfig.entropy_anneal_steps
    capture_scale_initial: float = TrainConfig.capture_scale_initial
    capture_scale_final: float = TrainConfig.capture_scale_final
    capture_scale_anneal_steps: int = TrainConfig.capture_scale_anneal_steps
    # Learning rate annealing
    lr_final: float = TrainConfig.learning_rate * 0.25
    lr_anneal_enabled: bool = False
    anneal_log_freq: int = 50_000
    env_type: str = "classic"


def parse_args() -> TrainingArgs:
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=TrainConfig.total_steps)
    p.add_argument("--n-envs", type=int, default=TrainConfig.n_envs)
    p.add_argument("--logdir", type=str, default=TrainConfig.logdir)
    p.add_argument("--model-dir", type=str, default=TrainConfig.model_dir)
    p.add_argument("--eval-freq", type=int, default=TrainConfig.eval_freq)
    p.add_argument("--eval-games", type=int, default=TrainConfig.eval_games)
    p.add_argument(
        "--checkpoint-freq",
        type=int,
        default=100_000,
        help="Checkpoint every N steps; 0 disables",
    )
    p.add_argument(
        "--checkpoint-prefix",
        type=str,
        default="ppo_ludo",
        help="Checkpoint file prefix",
    )
    p.add_argument(
        "--eval-baselines",
        type=str,
        default=TrainConfig.eval_baselines,
        help="Comma-separated list of opponent strategy names",
    )
    p.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    p.add_argument("--n-steps", type=int, default=TrainConfig.n_steps)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--ent-coef", type=float, default=TrainConfig.ent_coef)
    p.add_argument("--max-turns", type=int, default=TrainConfig.max_turns)
    # Imitation / kickstart
    p.add_argument("--imitation-enabled", action="store_true", default=False)
    p.add_argument(
        "--imitation-strategies",
        type=str,
        default=TrainConfig.imitation_strategies,
        help="Comma-separated scripted strategies to imitate",
    )
    p.add_argument("--imitation-steps", type=int, default=TrainConfig.imitation_steps)
    p.add_argument(
        "--imitation-batch-size", type=int, default=TrainConfig.imitation_batch_size
    )
    p.add_argument("--imitation-epochs", type=int, default=TrainConfig.imitation_epochs)
    p.add_argument(
        "--imitation-entropy-boost",
        type=float,
        default=TrainConfig.imitation_entropy_boost,
        help="Temporary entropy bonus added to ent_coef during imitation phase",
    )
    # Annealing overrides
    p.add_argument(
        "--entropy-coef-initial", type=float, default=TrainConfig.entropy_coef_initial
    )
    p.add_argument(
        "--entropy-coef-final", type=float, default=TrainConfig.entropy_coef_final
    )
    p.add_argument(
        "--entropy-anneal-steps", type=int, default=TrainConfig.entropy_anneal_steps
    )
    p.add_argument(
        "--capture-scale-initial",
        type=float,
        default=TrainConfig.capture_scale_initial,
    )
    p.add_argument(
        "--capture-scale-final", type=float, default=TrainConfig.capture_scale_final
    )
    p.add_argument(
        "--capture-scale-anneal-steps",
        type=int,
        default=TrainConfig.capture_scale_anneal_steps,
    )
    # Learning rate annealing
    p.add_argument(
        "--lr-final",
        type=float,
        default=TrainConfig.learning_rate * 0.25,
        help="Final learning rate for linear anneal (initial is --learning-rate)",
    )
    p.add_argument(
        "--lr-anneal-enabled",
        action="store_true",
        default=False,
        help="Enable linear LR annealing",
    )
    p.add_argument(
        "--anneal-log-freq",
        type=int,
        default=50_000,
        help="Log annealed values (entropy, capture scale, lr) every N env steps",
    )
    p.add_argument(
        "--env-type",
        type=str,
        default="classic",
        choices=["classic", "selfplay"],
        help="Environment type",
    )
    args = p.parse_args()
    return TrainingArgs(**vars(args))
