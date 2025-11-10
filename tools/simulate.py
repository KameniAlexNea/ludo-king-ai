import argparse
import random
import time
from typing import Optional

import numpy as np
from sb3_contrib import MaskablePPO

from ludo_rl.ludo_king import Game, Player, Board, Color, config as king_config
from ludo_rl.strategy.registry import available as available_strategies, STRATEGY_REGISTRY


def seed_environ(seed_value: Optional[int] = None):
    random.seed(seed_value)
    np.random.seed(seed_value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a trained Ludo agent (ludo_king)")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the MaskablePPO model zip file",
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


def main() -> None:
    args = parse_args()
    seed_environ(42)

    print("--- Initializing Test Environment (ludo_king) ---")
    print(f"Max game turns set to: {king_config.MAX_TURNS}")

    # Build a 4-player game: agent at RED, opponents are default strategies
    color_ids = [int(Color.RED), int(Color.GREEN), int(Color.YELLOW), int(Color.BLUE)]
    players = [Player(color=c) for c in color_ids]
    game = Game(players=players)

    rng = random.Random(12345)
    opp_list = [s for s in available_strategies() if s]
    opponents = opp_list[:3]

    for idx, pl in enumerate(game.players):
        for pc in pl.pieces:
            pc.position = 0
        pl.has_finished = False
        if idx == 0:
            pl.strategy_name = "agent"  # type: ignore[attr-defined]
        else:
            attach_strategy(pl, opponents[idx - 1], rng)

    print("Opponents:")
    for idx, pl in enumerate(game.players):
        if idx == 0:
            continue
        print(f"  Seat {idx}: {getattr(pl, 'strategy_name', '?')}")

    model = MaskablePPO.load(args.model_path)
    model.policy.set_training_mode(False)
    print(model.policy)

    print("\n--- Starting Simulation ---")
    start_time = time.time()

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
                print(
                    f"Step {step_count}, Agent piece {pid}, Dice: {dice}, Legal: {len(legal)}"
                )
            else:
                decision = player.choose(board_stack, dice, legal)
                mv = decision if decision is not None else rng.choice(legal)

            result = game.apply_move(mv)
            extra = result.extra_turn and result.events.move_resolved

            if player.check_won() and current not in finish_order:
                finish_order.append(current)

        current = (current + 1) % len(game.players)
        turns += 1

    print("\n--- SIMULATION COMPLETE ---")
    end_time = time.time()
    print(f"Total Turns: {turns}")
    print(f"Total Steps: {step_count}")
    print(f"Simulation Time: {end_time - start_time:.2f} seconds")
    print("Finish order (seats):", finish_order)


if __name__ == "__main__":
    main()
