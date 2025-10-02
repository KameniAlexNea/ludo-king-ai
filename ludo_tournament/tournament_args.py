import argparse
import os


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PPO vs Strategies Tournament System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        default=int(os.getenv("MAX_TURNS_PER_GAME", 1000)),
        help="Maximum turns per game before declaring draw",
    )

    parser.add_argument(
        "--games-per-matchup",
        type=int,
        default=int(os.getenv("GAMES_PER_MATCHUP", 10)),
        help="Number of games to play per strategy combination",
    )

    parser.add_argument("--seed", type=int, default=42, help="Tournament random seed")

    parser.add_argument(
        "--models-dir",
        type=str,
        default="./training/models",
        help="Directory containing PPO model files",
    )

    parser.add_argument(
        "--strategies",
        type=str,
        nargs="*",
        help="Specific strategies to include (default: all available)",
    )

    parser.add_argument("--quiet", action="store_true", help="Reduce verbose output")

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save game states and results (optional)",
    )

    parser.add_argument(
        "--model-preference",
        type=str,
        choices=["best", "final", "steps"],
        default="final",
        help="Preference for selecting PPO model: 'best' (prefer BEST model), 'final' (prefer FINAL model), 'steps' (prefer highest step count)",
    )

    parser.add_argument(
        "--env",
        type=str,
        choices=["classic", "single-seat"],
        default="classic",
        help="Environment kind: 'classic' or 'single-seat'",
    )
    return parser.parse_args()
