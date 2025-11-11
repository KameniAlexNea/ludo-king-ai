import argparse
import json
import os
import random
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sb3_contrib import MaskablePPO

from ludo_rl.ludo_king import config as king_config
from ludo_rl.ludo_king.enums import Color
from ludo_rl.ludo_king.game import Game
from ludo_rl.ludo_king.player import Player
from ludo_rl.ludo_king.types import Move
from ludo_rl.strategy.features import build_move_options
from ludo_rl.strategy.registry import STRATEGY_REGISTRY
from ludo_rl.strategy.registry import available as available_strategies
from ludo_rl.strategy.rl_agent import RLStrategy

# -----------------------------
# CLI and utilities
# -----------------------------


def seed_everything(seed: Optional[int]) -> random.Random:
    rng = random.Random(seed)
    np.random.seed(seed if seed is not None else 0)
    return rng


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate agent and profile its play-style alignment over time"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the MaskablePPO model .zip",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=int(os.getenv("NUM_PLAYERS", 4)),
        help="Number of players (2 or 4)",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=1,
        help="Number of games to simulate",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=int(os.getenv("PROFILE_WINDOW", 30)),
        help="Sliding window (in agent moves) for profile smoothing",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.getenv("PROFILE_THRESHOLD", 0.6)),
        help="Minimum alignment ratio in window to assign a profile",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="training/profile_analysis.json",
        help="Where to save the JSON report",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--include-timeline",
        action="store_true",
        help="Include per-step profile timeline in JSON output",
    )
    return parser.parse_args()


# -----------------------------
# Profiling core
# -----------------------------


@dataclass
class StepRecord:
    step: int
    dice: int
    agent_piece: int
    aligned: Dict[str, bool]


@dataclass
class Segment:
    start: int
    end: int
    profile: str
    accuracy: float


def build_action_mask_and_choices(
    player: Player, legal_moves: Sequence[Move]
) -> Tuple[np.ndarray, List[dict | None]]:
    pieces_per_player = len(player.pieces)
    action_mask = np.zeros(pieces_per_player, dtype=np.bool_)
    move_choices: List[dict | None] = [None] * pieces_per_player
    for mv in legal_moves:
        pid = int(mv.piece_id)
        if 0 <= pid < pieces_per_player and move_choices[pid] is None:
            action_mask[pid] = True
            move_choices[pid] = {"piece": player.pieces[pid], "new_pos": mv.new_pos}
    return action_mask, move_choices


def choose_move_by_piece(legal_moves: Sequence[Move], piece_id: int) -> Optional[Move]:
    for mv in legal_moves:
        if int(mv.piece_id) == int(piece_id):
            return mv
    return None


def segment_profiles(
    records: List[StepRecord], window: int, threshold: float
) -> Tuple[List[str], List[Segment]]:
    if not records:
        return [], []

    # Collect strategy names from records
    names: List[str] = []
    for rec in records:
        for k in rec.aligned.keys():
            if k not in names:
                names.append(k)

    # Rolling window counts per strategy
    counters: Dict[str, int] = {n: 0 for n in names}
    queue: deque[Dict[str, bool]] = deque()
    selected: List[str] = []

    for idx, rec in enumerate(records):
        queue.append(rec.aligned)
        for n in names:
            if rec.aligned.get(n, False):
                counters[n] += 1
        # Trim window
        while len(queue) > window:
            old = queue.popleft()
            for n in names:
                if old.get(n, False):
                    counters[n] -= 1

        # Compute best strategy in current window
        win_len = len(queue)
        best_name = "mixed"
        best_ratio = 0.0
        for n in names:
            ratio = counters[n] / max(1, win_len)
            if ratio > best_ratio or (
                ratio == best_ratio and n == (selected[-1] if selected else n)
            ):
                best_ratio = ratio
                best_name = n
        if best_ratio < threshold:
            best_name = "mixed"
        selected.append(best_name)

    # Compress into segments
    segments: List[Segment] = []
    if selected:
        cur = selected[0]
        start_idx = 0
        for i in range(1, len(selected)):
            if selected[i] != cur:
                seg_records = [
                    r
                    for r, lab in zip(records[start_idx:i], selected[start_idx:i])
                    if lab == cur
                ]
                acc = 0.0
                if cur != "mixed" and seg_records:
                    total = len(seg_records)
                    acc = (
                        sum(1 for r in seg_records if r.aligned.get(cur, False)) / total
                    )
                segments.append(
                    Segment(
                        start=records[start_idx].step,
                        end=records[i - 1].step,
                        profile=cur,
                        accuracy=acc,
                    )
                )
                cur = selected[i]
                start_idx = i
        # last segment
        seg_records = [
            r for r, lab in zip(records[start_idx:], selected[start_idx:]) if lab == cur
        ]
        acc = 0.0
        if cur != "mixed" and seg_records:
            total = len(seg_records)
            acc = sum(1 for r in seg_records if r.aligned.get(cur, False)) / total
        segments.append(
            Segment(
                start=records[start_idx].step,
                end=records[-1].step,
                profile=cur,
                accuracy=acc,
            )
        )

    return selected, segments


def run_single_game(
    model: MaskablePPO,
    strategy_pool: Dict[str, object],
    num_players: int,
    rng: random.Random,
    window: int,
    threshold: float,
    include_timeline: bool,
) -> dict:
    # Build game
    if num_players == 2:
        color_ids = [int(Color.RED), int(Color.YELLOW)]
    else:
        color_ids = [
            int(Color.RED),
            int(Color.GREEN),
            int(Color.YELLOW),
            int(Color.BLUE),
        ][:num_players]
    players = [Player(color=c) for c in color_ids]
    game = Game(players=players)

    # Opponents lineup (simple slice of available strategies)
    available_names = [n for n in available_strategies().keys()]
    opponents = available_names[: max(0, len(players) - 1)]
    for idx, pl in enumerate(game.players):
        for pc in pl.pieces:
            pc.position = 0
        pl.has_finished = False
        if idx == 0:
            pl.strategy_name = "agent"  # type: ignore[attr-defined]
        else:
            name = opponents[idx - 1] if idx - 1 < len(opponents) else ""
            if name and name in STRATEGY_REGISTRY:
                cls = STRATEGY_REGISTRY[name]
                try:
                    pl.strategy = cls.create_instance(rng)
                except NotImplementedError:
                    pl.strategy = cls()
                pl.strategy_name = name  # type: ignore[attr-defined]
            else:
                pl.strategy = None
                pl.strategy_name = "random"  # type: ignore[attr-defined]

    # Profiling setup
    rl_agent = RLStrategy(model=model, deterministic=True)
    step_idx = 0  # counts only agent decisions
    turns = 0
    current = 0
    finish_order: List[int] = []
    step_records: List[StepRecord] = []

    start_time = time.time()

    while turns < king_config.MAX_TURNS and len(finish_order) < len(game.players):
        player = game.players[current]
        if player.check_won():
            if current not in finish_order:
                finish_order.append(current)
            current = (current + 1) % len(game.players)
            continue

        extra = True
        while extra:
            dice = game.roll_dice()
            legal = game.legal_moves(current, dice)
            if not legal:
                extra = False
                continue

            board_stack = game.board.build_tensor(int(player.color))

            if current == 0:
                # Build context once for all strategies
                action_mask, move_choices = build_action_mask_and_choices(player, legal)
                ctx = build_move_options(
                    board_stack, int(dice), action_mask, move_choices
                )

                # Agent decision
                agent_decision = rl_agent.select_move(ctx)
                if agent_decision is None:
                    mv = rng.choice(legal)
                    agent_piece_id = int(mv.piece_id)
                else:
                    agent_piece_id = int(agent_decision.piece_id)
                    mv = choose_move_by_piece(legal, agent_piece_id) or rng.choice(
                        legal
                    )

                # Compare with each heuristic strategy
                aligned: Dict[str, bool] = {}
                for name, strat in strategy_pool.items():
                    try:
                        move_opt = strat.select_move(ctx)
                        pid = None if move_opt is None else int(move_opt.piece_id)
                        aligned[name] = pid == agent_piece_id
                    except Exception:
                        aligned[name] = False

                step_idx += 1
                step_records.append(
                    StepRecord(
                        step=step_idx,
                        dice=dice,
                        agent_piece=agent_piece_id,
                        aligned=aligned,
                    )
                )
            else:
                # Opponent move using their attached strategy (or random fallback in Player.choose)
                decision = player.choose(board_stack, dice, legal)
                mv = decision if decision is not None else rng.choice(legal)

            result = game.apply_move(mv)
            extra = result.extra_turn and result.events.move_resolved

            if player.check_won() and current not in finish_order:
                finish_order.append(current)

        current = (current + 1) % len(game.players)
        turns += 1

    elapsed = time.time() - start_time

    # Aggregate profile segments
    timeline, segments = segment_profiles(
        step_records, window=window, threshold=threshold
    )

    # Per-strategy accuracy within this game
    totals: Dict[str, int] = defaultdict(int)
    hits: Dict[str, int] = defaultdict(int)
    for rec in step_records:
        for name, ok in rec.aligned.items():
            totals[name] += 1
            if ok:
                hits[name] += 1
    game_accuracy = {
        name: (hits[name] / totals[name]) if totals[name] > 0 else 0.0
        for name in totals
    }

    # Prepare JSON-serializable output
    return {
        "steps": step_idx,
        "turns": turns,
        "elapsed_sec": elapsed,
        "finish_order": finish_order,
        "strategy_accuracy": game_accuracy,
        "strategy_hits": hits,
        "strategy_totals": totals,
        "profile_segments": [asdict(s) for s in segments],
        "profile_timeline": timeline if include_timeline else [],
    }


def main() -> None:
    args = parse_args()
    rng = seed_everything(args.seed)

    # Load model and set to eval
    model = MaskablePPO.load(args.model_path)
    model.policy.set_training_mode(False)

    # Build pool of strategies to compare against (exclude human/llm/rl)
    pool_names = [n for n in available_strategies().keys()]
    strategy_pool: Dict[str, object] = {}
    for name in pool_names:
        cls = STRATEGY_REGISTRY.get(name)
        if cls is None:
            continue
        try:
            strategy_pool[name] = cls.create_instance(rng)
        except NotImplementedError:
            try:
                strategy_pool[name] = cls()
            except Exception:
                continue

    print(f"Profiling against strategies: {', '.join(strategy_pool.keys())}")

    results: List[dict] = []

    # Global accuracy accumulators
    global_totals: Dict[str, int] = defaultdict(int)
    global_hits: Dict[str, int] = defaultdict(int)

    for gi in range(args.num_games):
        game_res = run_single_game(
            model=model,
            strategy_pool=strategy_pool,
            num_players=args.num_players,
            rng=rng,
            window=args.window,
            threshold=args.threshold,
            include_timeline=args.include_timeline,
        )
        results.append(game_res)
        # Accumulate global accuracy using exact hits/totals
        for name, total in game_res.get("strategy_totals", {}).items():
            global_totals[name] += int(total)
        for name, hit in game_res.get("strategy_hits", {}).items():
            global_hits[name] += int(hit)
        print(
            f"Game {gi + 1}/{args.num_games}: steps={game_res['steps']} segments={len(game_res['profile_segments'])}"
        )

    global_accuracy = {
        name: (
            (global_hits[name] / global_totals[name])
            if global_totals[name] > 0
            else 0.0
        )
        for name in global_totals
    }

    output = {
        "model_path": args.model_path,
        "num_games": args.num_games,
        "num_players": args.num_players,
        "window": args.window,
        "threshold": args.threshold,
        "strategies": list(strategy_pool.keys()),
        "global_strategy_accuracy": global_accuracy,
        "games": results,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved profile analysis to {args.output_file}")


if __name__ == "__main__":
    main()
