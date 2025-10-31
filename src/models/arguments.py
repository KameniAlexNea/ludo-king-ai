import argparse

from models.configs.config import EnvConfig, TrainConfig


def parse_args() -> tuple[TrainConfig, EnvConfig]:
    defaults = TrainConfig()
    env_defaults = EnvConfig()

    parser = argparse.ArgumentParser(description="Train a PPO agent on classic Ludo.")
    parser.add_argument("--total-steps", type=int, default=defaults.total_steps)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--n-steps", type=int, default=defaults.n_steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--ent-coef", type=float, default=defaults.ent_coef)
    parser.add_argument("--vf-coef", type=float, default=defaults.vf_coef)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--gae-lambda", type=float, default=defaults.gae_lambda)
    parser.add_argument("--logdir", type=str, default=defaults.logdir)
    parser.add_argument("--model-dir", type=str, default=defaults.model_dir)
    parser.add_argument("--seed", type=int, default=defaults.seed, nargs="?")
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--save-steps", type=int, default=defaults.save_steps)
    parser.add_argument(
        "--pi-net-arch",
        type=int,
        nargs="+",
        default=list(defaults.pi_net_arch),
        help="Policy branch architecture, e.g. --pi-net-arch 256 128.",
    )
    parser.add_argument(
        "--vf-net-arch",
        type=int,
        nargs="+",
        default=list(defaults.vf_net_arch),
        help="Value branch architecture, e.g. --vf-net-arch 256 128.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=defaults.n_envs,
        help=(
            "Number of parallel environments (>=1). Eight CPU workers are a good default."
        ),
    )
    parser.add_argument("--max-turns", type=int, default=env_defaults.max_turns)
    parser.add_argument(
        "--fixed-agent-color",
        type=str,
        default=None,
        help="Optional fixed color for the learning agent (e.g. BLUE).",
    )
    parser.add_argument(
        "--opponent-strategy",
        type=str,
        default=env_defaults.opponent_strategy,
        help="Strategy name understood by ludo_engine.StrategyFactory.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=defaults.eval_freq,
        help="Run evaluations every N timesteps.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=defaults.eval_episodes,
        help="Number of evaluation games per opponent when evaluations run.",
    )
    parser.add_argument(
        "--eval-opponents",
        type=str,
        default=",".join(defaults.eval_opponents),
        help="Comma separated list of opponent strategies for evaluation runs.",
    )
    parser.add_argument(
        "--eval-stochastic",
        dest="eval_deterministic",
        action="store_false",
        default=defaults.eval_deterministic,
        help="Use stochastic actions during evaluation instead of deterministic default.",
    )

    args = parser.parse_args()

    eval_opponents = tuple(
        opponent.strip()
        for opponent in args.eval_opponents.split(",")
        if opponent.strip()
    )
    if not eval_opponents:
        eval_opponents = defaults.eval_opponents
    if args.eval_freq <= 0:
        parser.error("--eval-freq must be positive")

    if args.save_steps <= 0:
        parser.error("--save-steps must be positive")

    train_cfg = TrainConfig(
        total_steps=args.total_steps,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        logdir=args.logdir,
        model_dir=args.model_dir,
        seed=args.seed,
        device=args.device,
        save_steps=args.save_steps,
        n_envs=max(1, args.n_envs),
        eval_freq=args.eval_freq,
        eval_episodes=max(1, args.eval_episodes),
        eval_opponents=eval_opponents,
        eval_deterministic=args.eval_deterministic,
        pi_net_arch=tuple(args.pi_net_arch),
        vf_net_arch=tuple(args.vf_net_arch),
    )

    env_cfg = EnvConfig(
        max_turns=args.max_turns,
        seed=args.seed,
        randomize_agent=args.fixed_agent_color is None,
        opponent_strategy=args.opponent_strategy,
    )
    if args.fixed_agent_color:
        env_cfg.randomize_agent = False
        env_cfg.fixed_agent_color = args.fixed_agent_color
        env_cfg.opponent_strategy = args.opponent_strategy

    return train_cfg, env_cfg
