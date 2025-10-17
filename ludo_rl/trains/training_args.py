import argparse
from dataclasses import asdict

from ludo_rl.config import TrainConfig


def parse_args() -> TrainConfig:
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
        default=TrainConfig.checkpoint_freq,
        help="Checkpoint every N steps; 0 disables",
    )
    p.add_argument(
        "--checkpoint-prefix",
        type=str,
        default=TrainConfig.checkpoint_prefix,
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
    p.add_argument("--vf-coef", type=float, default=TrainConfig.vf_coef)
    p.add_argument("--gamma", type=float, default=TrainConfig.gamma)
    p.add_argument("--gae-lambda", type=float, default=TrainConfig.gae_lambda)
    p.add_argument("--max-turns", type=int, default=TrainConfig.max_turns)
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
        "--use-entropy-annealing",
        action="store_true",
        default=TrainConfig.use_entropy_annealing,
        help="Use entropy annealing if True",
    )
    p.add_argument(
        "--lr-final",
        type=float,
        default=TrainConfig.lr_final,
        help="Final learning rate for linear anneal (initial is --learning-rate)",
    )
    p.add_argument(
        "--lr-anneal-enabled",
        action="store_true",
        default=TrainConfig.lr_anneal_enabled,
        help="Enable linear LR annealing",
    )
    p.add_argument(
        "--anneal-log-freq",
        type=int,
        default=TrainConfig.anneal_log_freq,
        help="Log annealed values (entropy, capture scale, lr) every N env steps",
    )
    p.add_argument(
        "--env-type",
        type=str,
        default=TrainConfig.env_type,
        choices=["classic", "selfplay", "hybrid"],
        help="Environment type",
    )
    p.add_argument(
        "--hybrid-switch-rate",
        type=float,
        default=TrainConfig.hybrid_switch_rate,
        help="Switch rate for hybrid environment",
    )
    p.add_argument(
        "--embed_dim",
        type=int,
        default=TrainConfig.embed_dim,
        help="Embedding dimension for discrete observation extractor",
    )
    p.add_argument(
        "--load_model",
        type=str,
        default=TrainConfig.load_model,
        help="Path to model to load",
    )
    args = p.parse_args()
    args = TrainConfig(**vars(args))
    print(asdict(args))
    return args
