"""Simulate a trained Ludo agent and generate behavioral profile."""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sb3_contrib import MaskablePPO

from ludo_profile import GameAnalyzer
from ludo_profile.trace_builder import build_trace_from_game
from ludo_rl.ludo_king import Board, Color, Game, Player
from ludo_rl.ludo_king import config as king_config
from ludo_rl.strategy.registry import STRATEGY_REGISTRY
from ludo_rl.strategy.registry import available as available_strategies


def seed_environ(seed_value: Optional[int] = None):
    random.seed(seed_value)
    np.random.seed(seed_value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a trained Ludo agent with behavioral profiling"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the MaskablePPO model zip file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training/profile_outputs",
        help="Directory to save profile results (default: profile_outputs)",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=1,
        help="Number of games to simulate (default: 1)",
    )
    parser.add_argument(
        "--segmentation",
        type=str,
        choices=["fixed", "phase", "adaptive"],
        default="phase",
        help="Segmentation strategy for profiling (default: phase)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="Window size for fixed segmentation (default: 20)",
    )
    parser.add_argument(
        "--save-traces",
        action="store_true",
        help="Save raw game traces to JSON files",
    )
    return parser.parse_args()


def attach_strategy(player: Player, strategy_name: str, rng: random.Random) -> None:
    cls = STRATEGY_REGISTRY[strategy_name]
    try:
        player.strategy = cls.create_instance(rng)
    except NotImplementedError:
        player.strategy = cls()
    player.strategy_name = strategy_name  # type: ignore[attr-defined]


def build_board_stack(board: Board, player_color: int) -> np.ndarray:
    return board.build_tensor(player_color)


def _extract_piece_index(move: object) -> int | None:
    for attr in ("piece_id", "token", "piece_index"):
        if hasattr(move, attr):
            try:
                return int(getattr(move, attr))
            except Exception:
                pass
    piece = getattr(move, "piece", None)
    if piece is not None:
        for attr in ("piece_id", "index", "id"):
            if hasattr(piece, attr):
                try:
                    return int(getattr(piece, attr))
                except Exception:
                    pass
    return None


def simulate_game(
    model: MaskablePPO,
    opponents: list[str],
    game_id: str,
    rng: random.Random,
    verbose: bool = True,
) -> tuple[Game, list[dict], int]:
    """
    Simulate a single game and capture move history.

    Returns
    -------
    tuple[Game, list[dict], int]
        (game instance, move history, winner index)
    """
    # Build a 4-player game: agent at RED, opponents at other colors
    color_ids = [int(Color.RED), int(Color.GREEN), int(Color.YELLOW), int(Color.BLUE)]
    players = [Player(color=c) for c in color_ids]
    game = Game(players=players)

    for idx, pl in enumerate(game.players):
        for pc in pl.pieces:
            pc.position = 0
        pl.has_finished = False
        if idx == 0:
            pl.strategy_name = "agent"  # type: ignore[attr-defined]
        else:
            attach_strategy(pl, opponents[idx - 1], rng)

    if verbose:
        print(f"\nGame {game_id}:")
        print("  Opponents:")
        for idx, pl in enumerate(game.players):
            if idx > 0:
                print(f"    Seat {idx}: {getattr(pl, 'strategy_name', '?')}")

    move_history = []
    step_count = 0
    turns = 0
    finish_order: list[int] = []
    current = 0

    while turns < king_config.MAX_TURNS and len(finish_order) < len(game.players):
        player = game.players[current]
        if player.check_won():
            if current not in finish_order:
                finish_order.append(current)
            current = (current + 1) % len(game.players)
            continue

        extra = True
        while extra:
            step_count += 1
            dice = game.roll_dice()
            legal = game.legal_moves(current, dice)
            if not legal:
                extra = False
                continue

            board_stack = build_board_stack(game.board, int(player.color))

            if current == 0:
                # Agent move via model
                action_mask = np.zeros(king_config.PIECES_PER_PLAYER, dtype=bool)
                moves_by_piece: dict[int, list[object]] = {}
                for mv in legal:
                    pid = _extract_piece_index(mv)
                    if pid is None:
                        continue
                    if 0 <= pid < king_config.PIECES_PER_PLAYER:
                        action_mask[pid] = True
                        moves_by_piece.setdefault(pid, []).append(mv)

                obs = {
                    "board": board_stack[None, ...],
                    "dice_roll": np.array([[dice - 1]], dtype=np.int64),
                }
                action, _ = model.predict(obs, action_masks=action_mask[None, ...])
                pid = int(action.item())
                mv = rng.choice(moves_by_piece.get(pid, legal))
            else:
                decision = player.choose(board_stack, dice, legal)
                mv = decision if decision is not None else rng.choice(legal)

            # Record move details
            old_pos = player.pieces[mv.piece_id].position
            result = game.apply_move(mv)

            # Convert events to dict format for trace builder
            events_dict = {
                "exited_home": result.events.exited_home,
                "finished": result.events.finished,
                "knockouts": [
                    {
                        "player": ko["player"],
                        "piece_id": ko["piece_id"],
                        "abs_pos": ko.get("abs_pos", 0),
                    }
                    for ko in result.events.knockouts
                ],
                "hit_blockade": result.events.hit_blockade,
                "blockades": [
                    {"player": blk["player"], "rel": blk.get("rel", 0)}
                    for blk in result.events.blockades
                ],
                "move_resolved": result.events.move_resolved,
            }

            move_history.append(
                {
                    "player": current,
                    "dice": dice,
                    "piece": mv.piece_id,
                    "old": old_pos,
                    "new": result.new_position,
                    "events": events_dict,
                    "extra": result.extra_turn,
                }
            )

            extra = result.extra_turn and result.events.move_resolved

            if player.check_won() and current not in finish_order:
                finish_order.append(current)

        current = (current + 1) % len(game.players)
        turns += 1

    winner = finish_order[0] if finish_order else None

    if verbose:
        print(f"  Total Turns: {turns}, Total Steps: {step_count}")
        print(
            f"  Winner: Player {winner} ({Color(game.players[winner].color).name if winner is not None else 'None'})"
        )
        print(f"  Finish order: {finish_order}")

    return game, move_history, winner


def print_profile_summary(profile, verbose: bool = True):
    """Print a concise profile summary."""
    if not verbose:
        return

    print(f"\n{'=' * 60}")
    print(
        f"PROFILE: Player {profile.player_index} - {profile.player_color} ({profile.player_strategy})"
    )
    print(f"{'=' * 60}")
    print(f"Dominant style: {profile.overall_summary.dominant_style}")
    print(f"Win achieved: {profile.overall_summary.win_achieved}")
    print("\nStyle distribution:")
    for style, proportion in sorted(
        profile.overall_summary.style_distribution.items(), key=lambda x: -x[1]
    ):
        print(f"  {style:15s}: {proportion:>6.1%}")

    print("\nOverall statistics:")
    print(f"  Captures made:    {profile.overall_summary.total_captures}")
    print(f"  Got captured:     {profile.overall_summary.total_got_captured}")
    print(f"  Blockades formed: {profile.overall_summary.total_blockades}")
    print(f"  Pieces finished:  {profile.overall_summary.total_finished}")

    print(f"\nSegments ({len(profile.profile_segments)}):")
    for i, segment in enumerate(profile.profile_segments):
        print(
            f"\n  [{i + 1}] Steps {segment.step_range[0]}-{segment.step_range[1]}: {segment.style.upper()} (conf: {segment.confidence:.2f})"
        )
        print(
            f"      Aggression: {segment.characteristics.aggression:.2f} | "
            f"Risk: {segment.characteristics.risk_taking:.2f} | "
            f"Exploration: {segment.characteristics.exploration:.2f}"
        )
        print(
            f"      Finishing: {segment.characteristics.finishing:.2f} | "
            f"Blockade: {segment.characteristics.blockade_usage:.2f} | "
            f"Defense: {segment.characteristics.defensiveness:.2f}"
        )
        if segment.behaviors:
            print(f"      → {', '.join(segment.behaviors[:3])}")

    if profile.overall_summary.key_transitions:
        print(f"\nKey transitions ({len(profile.overall_summary.key_transitions)}):")
        for trans in profile.overall_summary.key_transitions:
            print(
                f"  Step {trans.step:>4}: {trans.from_style:>12} → {trans.to_style:<12} ({trans.trigger})"
            )


def main() -> None:
    args = parse_args()
    seed_environ(42)

    print("=" * 60)
    print("LUDO AGENT SIMULATION WITH BEHAVIORAL PROFILING")
    print("=" * 60)
    print(f"Max game turns: {king_config.MAX_TURNS}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of games: {args.num_games}")
    print(f"Segmentation: {args.segmentation}")
    if args.segmentation == "fixed":
        print(f"Window size: {args.window_size}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load model
    print("\nLoading model...")
    model = MaskablePPO.load(args.model_path)
    model.policy.set_training_mode(False)
    print("✓ Model loaded")

    # Setup opponents
    rng = random.Random(12345)
    opp_list = [s for s in available_strategies() if s]
    opponents = opp_list[:3]
    print(f"Opponents: {', '.join(opponents)}")

    # Create analyzer
    analyzer = GameAnalyzer()

    # Simulate games and collect profiles
    all_profiles = []
    aggregate_stats = {
        "wins": 0,
        "total_captures": 0,
        "total_got_captured": 0,
        "total_blockades": 0,
        "style_counts": {},
    }

    start_time = time.time()

    for game_num in range(args.num_games):
        game_id = f"game_{game_num:03d}"

        # Simulate game
        game, move_history, winner = simulate_game(
            model, opponents, game_id, rng, verbose=(args.num_games <= 5)
        )

        # Build trace
        trace = build_trace_from_game(game, move_history)

        # Save raw trace if requested
        if args.save_traces:
            trace_dict = {
                "game_id": trace.game_id,
                "num_players": trace.num_players,
                "players": [
                    {"index": p.index, "color": p.color, "strategy": p.strategy}
                    for p in trace.players
                ],
                "moves": [
                    {
                        "step": m.step,
                        "player_index": m.player_index,
                        "dice_roll": m.dice_roll,
                        "piece_id": m.piece_id,
                        "old_position": m.old_position,
                        "new_position": m.new_position,
                        "events": {
                            "exited_home": m.events.exited_home,
                            "finished": m.events.finished,
                            "knockouts": [
                                {
                                    "player": ko.player,
                                    "piece_id": ko.piece_id,
                                    "abs_pos": ko.abs_pos,
                                }
                                for ko in m.events.knockouts
                            ],
                            "hit_blockade": m.events.hit_blockade,
                            "blockades": [
                                {"player": blk.player, "rel": blk.rel}
                                for blk in m.events.blockades
                            ],
                            "move_resolved": m.events.move_resolved,
                        },
                        "extra_turn": m.extra_turn,
                    }
                    for m in trace.moves
                ],
                "winner": trace.winner,
                "total_turns": trace.total_turns,
            }
            with open(output_dir / f"trace_{game_id}.json", "w") as f:
                json.dump(trace_dict, f, indent=2)

        # Analyze agent (player 0)
        profile = analyzer.analyze_game(
            trace,
            player_index=0,
            segmentation=args.segmentation,
            window_size=args.window_size,
        )

        # Save profile
        with open(output_dir / f"profile_{game_id}.json", "w") as f:
            json.dump(profile.to_dict(), f, indent=2)

        all_profiles.append(profile)

        # Update aggregate stats
        if profile.overall_summary.win_achieved:
            aggregate_stats["wins"] += 1
        aggregate_stats["total_captures"] += profile.overall_summary.total_captures
        aggregate_stats["total_got_captured"] += (
            profile.overall_summary.total_got_captured
        )
        aggregate_stats["total_blockades"] += profile.overall_summary.total_blockades

        dominant = profile.overall_summary.dominant_style
        aggregate_stats["style_counts"][dominant] = (
            aggregate_stats["style_counts"].get(dominant, 0) + 1
        )

        # Print profile for first few games
        if args.num_games <= 5:
            print_profile_summary(profile, verbose=True)

    end_time = time.time()

    # Print aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    print(f"Games simulated: {args.num_games}")
    print(f"Total simulation time: {end_time - start_time:.2f} seconds")
    print(
        f"Average time per game: {(end_time - start_time) / args.num_games:.2f} seconds"
    )
    print("\nAgent performance:")
    print(
        f"  Wins: {aggregate_stats['wins']}/{args.num_games} ({100 * aggregate_stats['wins'] / args.num_games:.1f}%)"
    )
    print(
        f"  Avg captures per game: {aggregate_stats['total_captures'] / args.num_games:.1f}"
    )
    print(
        f"  Avg got captured per game: {aggregate_stats['total_got_captured'] / args.num_games:.1f}"
    )
    print(
        f"  Avg blockades per game: {aggregate_stats['total_blockades'] / args.num_games:.1f}"
    )

    print("\nDominant style distribution:")
    for style, count in sorted(
        aggregate_stats["style_counts"].items(), key=lambda x: -x[1]
    ):
        print(
            f"  {style:15s}: {count:>3} games ({100 * count / args.num_games:>5.1f}%)"
        )

    # Save aggregate summary
    summary = {
        "num_games": args.num_games,
        "simulation_time": end_time - start_time,
        "model_path": args.model_path,
        "opponents": opponents,
        "segmentation": args.segmentation,
        "aggregate_stats": aggregate_stats,
        "profiles": [p.to_dict() for p in all_profiles],
    }

    with open(output_dir / "aggregate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ All results saved to {output_dir}/")
    print(f"  - {args.num_games} profile JSON files")
    if args.save_traces:
        print(f"  - {args.num_games} trace JSON files")
    print("  - 1 aggregate summary JSON file")


if __name__ == "__main__":
    main()
