"""
Profiling script for LudoEnv to identify bottlenecks.

This script runs the environment with profiling enabled to identify
performance bottlenecks in the RL training loop (env.step, env.reset, etc).
"""

from __future__ import annotations

import cProfile
import pstats
import random
import time
from io import StringIO
from pathlib import Path

import numpy as np

from ludo_rl.ludo_env import LudoEnv


def seed_environ(seed_value: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)


def run_single_episode(env: LudoEnv, rng: random.Random) -> dict:
    """Run a single episode and return timing statistics."""
    # Timing stats
    time_reset = 0.0
    time_step = 0.0
    time_action_mask = 0.0
    time_random_action = 0.0
    step_count = 0

    # Reset
    t0 = time.perf_counter()
    obs, info = env.reset(seed=rng.randint(0, 1_000_000))
    time_reset = time.perf_counter() - t0

    terminated = False
    truncated = False

    while not terminated and not truncated:
        # Time action_masks
        t0 = time.perf_counter()
        mask = env.action_masks()
        time_action_mask += time.perf_counter() - t0

        # Time random action selection
        t0 = time.perf_counter()
        if mask is not None and np.any(mask):
            valid_actions = np.where(mask)[0]
            action = int(rng.choice(valid_actions))
        else:
            action = 0
        time_random_action += time.perf_counter() - t0

        # Time step
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        time_step += time.perf_counter() - t0

        step_count += 1

    return {
        "steps": step_count,
        "time_reset": time_reset,
        "time_step": time_step,
        "time_action_mask": time_action_mask,
        "time_random_action": time_random_action,
        "final_rank": info.get("final_rank") if isinstance(info, dict) else None,
    }


def run_profiled_episodes(num_episodes: int = 100, seed: int = 42):
    """Run multiple episodes with timing statistics."""
    seed_environ(seed)
    rng = random.Random(seed)

    print(f"Running {num_episodes} episodes with LudoEnv")
    print("=" * 80)

    # Create environment once
    env = LudoEnv()
    print(f"Opponents: {env.opponents}")
    print("=" * 80)

    total_stats = {
        "steps": 0,
        "time_reset": 0.0,
        "time_step": 0.0,
        "time_action_mask": 0.0,
        "time_random_action": 0.0,
        "wins": 0,
    }

    start_time = time.perf_counter()

    for i in range(num_episodes):
        stats = run_single_episode(env, rng)
        total_stats["steps"] += stats["steps"]
        total_stats["time_reset"] += stats["time_reset"]
        total_stats["time_step"] += stats["time_step"]
        total_stats["time_action_mask"] += stats["time_action_mask"]
        total_stats["time_random_action"] += stats["time_random_action"]
        if stats["final_rank"] == 1:
            total_stats["wins"] += 1

        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_episodes} episodes...")

    total_time = time.perf_counter() - start_time
    env.close()

    print("\n" + "=" * 80)
    print("TIMING ANALYSIS")
    print("=" * 80)
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Total episodes: {num_episodes}")
    print(f"Time per episode: {total_time / num_episodes:.4f}s")
    print(f"Total steps: {total_stats['steps']}")
    print(f"Steps per episode: {total_stats['steps'] / num_episodes:.1f}")
    print(f"Agent wins: {total_stats['wins']}/{num_episodes}")
    print()

    # Calculate percentages
    time_sum = (
        total_stats["time_reset"]
        + total_stats["time_step"]
        + total_stats["time_action_mask"]
        + total_stats["time_random_action"]
    )

    print("Time breakdown:")
    print(
        f"  env.reset:        {total_stats['time_reset']:8.3f}s "
        f"({100 * total_stats['time_reset'] / time_sum:5.1f}%) "
        f"- {total_stats['time_reset'] / num_episodes * 1000:.3f}ms per call"
    )
    print(
        f"  env.step:         {total_stats['time_step']:8.3f}s "
        f"({100 * total_stats['time_step'] / time_sum:5.1f}%) "
        f"- {total_stats['time_step'] / total_stats['steps'] * 1000:.3f}ms per call"
    )
    print(
        f"  action_masks:     {total_stats['time_action_mask']:8.3f}s "
        f"({100 * total_stats['time_action_mask'] / time_sum:5.1f}%) "
        f"- {total_stats['time_action_mask'] / total_stats['steps'] * 1000:.3f}ms per call"
    )
    print(
        f"  random_action:    {total_stats['time_random_action']:8.3f}s "
        f"({100 * total_stats['time_random_action'] / time_sum:5.1f}%) "
        f"- {total_stats['time_random_action'] / total_stats['steps'] * 1000:.3f}ms per call"
    )
    print()
    print(
        f"Measured overhead: {time_sum:.3f}s ({100 * time_sum / total_time:.1f}% of total)"
    )
    print(
        f"Unmeasured overhead: {total_time - time_sum:.3f}s "
        f"({100 * (total_time - time_sum) / total_time:.1f}% of total)"
    )

    return total_stats


def run_cprofile_episodes(num_episodes: int = 10):
    """Run episodes with cProfile for detailed function-level profiling."""
    print("\n" + "=" * 80)
    print("DETAILED PROFILING (cProfile)")
    print("=" * 80)

    seed_environ(42)
    rng = random.Random(42)
    env = LudoEnv()

    profiler = cProfile.Profile()
    profiler.enable()

    # Run multiple episodes for better stats
    for _ in range(num_episodes):
        run_single_episode(env, rng)

    profiler.disable()
    env.close()

    # Print stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(60)  # Top 60 functions
    print(s.getvalue())

    # Save to file
    output_dir = Path(__file__).parent.parent / "training/profiling_results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "ludo_env_profile.prof"
    profiler.dump_stats(str(output_file))
    print(f"\nProfile saved to: {output_file}")
    print(f"View with: python -m pstats {output_file}")


def run_step_breakdown(num_episodes: int = 20):
    """Detailed breakdown of what happens inside env.step()."""
    print("\n" + "=" * 80)
    print("STEP BREAKDOWN PROFILING")
    print("=" * 80)

    seed_environ(42)
    rng = random.Random(42)
    env = LudoEnv()

    # Monkey-patch to measure internal timings
    original_step = env.step

    step_timings = {
        "total_steps": 0,
        "agent_steps": 0,
        "opponent_steps": 0,
    }

    def instrumented_step(action):
        step_timings["total_steps"] += 1
        return original_step(action)

    env.step = instrumented_step

    # Run episodes
    for ep in range(num_episodes):
        obs, info = env.reset(seed=rng.randint(0, 1_000_000))
        terminated = False
        truncated = False

        while not terminated and not truncated:
            mask = env.action_masks()
            if mask is not None and np.any(mask):
                valid_actions = np.where(mask)[0]
                action = int(rng.choice(valid_actions))
            else:
                action = 0
            obs, reward, terminated, truncated, info = env.step(action)

    env.close()

    print(f"Episodes: {num_episodes}")
    print(f"Total env.step() calls: {step_timings['total_steps']}")
    print(f"Steps per episode: {step_timings['total_steps'] / num_episodes:.1f}")


def main():
    """Main entry point."""
    print("LudoEnv Profiling")
    print("=" * 80)
    print()

    # First run timing analysis
    run_profiled_episodes(num_episodes=100, seed=42)

    # Step breakdown
    run_step_breakdown(num_episodes=20)

    # Then run detailed profiling
    run_cprofile_episodes(num_episodes=10)


if __name__ == "__main__":
    main()
