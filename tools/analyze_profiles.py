"""Analyze and compare multiple agent profiles."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze and compare agent behavioral profiles"
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        required=True,
        help="Directory containing profile JSON files",
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs="+",
        help="Paths to specific profile files to compare",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for comparison report (optional)",
    )
    return parser.parse_args()


def load_profiles(profile_dir: Path) -> List[Dict]:
    """Load all profile JSON files from directory."""
    profiles = []
    for file in sorted(profile_dir.glob("profile_*.json")):
        with open(file, "r") as f:
            profiles.append(json.load(f))
    return profiles


def analyze_style_evolution(profiles: List[Dict]) -> Dict:
    """Analyze how playing style evolves across games."""
    evolution = defaultdict(list)

    for i, profile in enumerate(profiles):
        for segment in profile["profile_segments"]:
            step_range = segment["step_range"]
            mid_point = (step_range[0] + step_range[1]) / 2
            evolution[segment["style"]].append(
                {
                    "game": i,
                    "step": mid_point,
                    "confidence": segment["confidence"],
                    "characteristics": segment["characteristics"],
                }
            )

    return dict(evolution)


def compute_aggregate_metrics(profiles: List[Dict]) -> Dict:
    """Compute aggregate metrics across all profiles."""
    metrics = {
        "total_games": len(profiles),
        "wins": 0,
        "avg_captures": 0,
        "avg_got_captured": 0,
        "avg_blockades": 0,
        "avg_finished": 0,
        "style_frequency": defaultdict(int),
        "avg_characteristics": defaultdict(list),
    }

    for profile in profiles:
        summary = profile["overall_summary"]
        metrics["wins"] += int(summary["win_achieved"])
        metrics["avg_captures"] += summary["total_captures"]
        metrics["avg_got_captured"] += summary["total_got_captured"]
        metrics["avg_blockades"] += summary["total_blockades"]
        metrics["avg_finished"] += summary["total_finished"]

        # Count dominant styles
        metrics["style_frequency"][summary["dominant_style"]] += 1

        # Collect characteristics from all segments
        for segment in profile["profile_segments"]:
            chars = segment["characteristics"]
            for key, value in chars.items():
                metrics["avg_characteristics"][key].append(value)

    # Compute averages
    n = len(profiles)
    metrics["avg_captures"] /= n
    metrics["avg_got_captured"] /= n
    metrics["avg_blockades"] /= n
    metrics["avg_finished"] /= n

    # Average characteristics
    for key, values in metrics["avg_characteristics"].items():
        metrics["avg_characteristics"][key] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "std": (
                sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values)
            )
            ** 0.5,
        }

    metrics["style_frequency"] = dict(metrics["style_frequency"])
    metrics["avg_characteristics"] = dict(metrics["avg_characteristics"])

    return metrics


def print_comparison(profiles: List[Dict]) -> None:
    """Print comparison of multiple profiles."""
    print("\n" + "=" * 80)
    print("PROFILE COMPARISON")
    print("=" * 80)

    for i, profile in enumerate(profiles):
        summary = profile["overall_summary"]
        print(f"\nProfile {i + 1}: {profile['game_id']}")
        print(f"  Player: {profile['player_color']} ({profile['player_strategy']})")
        print(f"  Dominant style: {summary['dominant_style']}")
        print(f"  Win: {summary['win_achieved']}")
        print(
            f"  Captures: {summary['total_captures']}, Got captured: {summary['total_got_captured']}"
        )
        print(
            f"  Blockades: {summary['total_blockades']}, Finished: {summary['total_finished']}"
        )
        print(f"  Segments: {len(profile['profile_segments'])}")

        if summary.get("key_transitions"):
            print(f"  Transitions: {len(summary['key_transitions'])}")


def print_aggregate_analysis(metrics: Dict) -> None:
    """Print aggregate analysis results."""
    print("\n" + "=" * 80)
    print("AGGREGATE ANALYSIS")
    print("=" * 80)

    print(f"\nTotal games analyzed: {metrics['total_games']}")
    print(
        f"Win rate: {metrics['wins']}/{metrics['total_games']} ({100 * metrics['wins'] / metrics['total_games']:.1f}%)"
    )

    print("\nAverage statistics per game:")
    print(f"  Captures made:    {metrics['avg_captures']:.1f}")
    print(f"  Got captured:     {metrics['avg_got_captured']:.1f}")
    print(f"  Blockades formed: {metrics['avg_blockades']:.1f}")
    print(f"  Pieces finished:  {metrics['avg_finished']:.1f}")

    print("\nDominant style distribution:")
    for style, count in sorted(metrics["style_frequency"].items(), key=lambda x: -x[1]):
        pct = 100 * count / metrics["total_games"]
        print(f"  {style:15s}: {count:>3} games ({pct:>5.1f}%)")

    print("\nAverage characteristics across all segments:")
    for char, stats in sorted(metrics["avg_characteristics"].items()):
        print(
            f"  {char:20s}: {stats['mean']:.3f} (±{stats['std']:.3f}) [{stats['min']:.3f}, {stats['max']:.3f}]"
        )


def main() -> None:
    args = parse_args()

    if args.compare:
        # Compare specific profiles
        profiles = []
        for path in args.compare:
            with open(path, "r") as f:
                profiles.append(json.load(f))
        print_comparison(profiles)
    else:
        # Analyze directory of profiles
        profile_dir = Path(args.profile_dir)
        if not profile_dir.exists():
            print(f"Error: Directory '{profile_dir}' does not exist")
            return

        profiles = load_profiles(profile_dir)
        if not profiles:
            print(f"Error: No profile files found in '{profile_dir}'")
            return

        print(f"Loaded {len(profiles)} profiles from {profile_dir}")

        # Compute metrics
        metrics = compute_aggregate_metrics(profiles)

        # Print analysis
        print_aggregate_analysis(metrics)

        # Analyze style evolution
        evolution = analyze_style_evolution(profiles)
        print("\n" + "=" * 80)
        print("STYLE EVOLUTION")
        print("=" * 80)
        for style, occurrences in sorted(evolution.items()):
            print(f"\n{style.upper()} ({len(occurrences)} occurrences):")
            games = set(occ["game"] for occ in occurrences)
            print(f"  Appears in {len(games)} games")
            avg_conf = sum(occ["confidence"] for occ in occurrences) / len(occurrences)
            print(f"  Average confidence: {avg_conf:.2f}")

        # Save report if requested
        if args.output:
            report = {
                "aggregate_metrics": metrics,
                "style_evolution": evolution,
                "profiles": profiles,
            }
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\n✓ Report saved to {args.output}")


if __name__ == "__main__":
    main()
