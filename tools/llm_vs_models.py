"""Run matches defined by a YAML configuration file using ludo_king engine."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from loguru import logger
from sb3_contrib import MaskablePPO
import numpy as np

from ludo_rl.ludo_king import Game, Player, Board, Color, config as king_config
from ludo_rl.strategy.llm_agent import (
    DEFAULT_SYSTEM_PROMPT,
    LLMStrategy,
    init_chat_model,
)
from ludo_rl.strategy.registry import STRATEGY_REGISTRY

POINTS_TABLE = (3, 2, 1, 0)


try:  # Optional dependency guard for YAML parsing
    import yaml
except ImportError as exc:  # pragma: no cover - runtime guard
    yaml = None  # type: ignore[assignment]
    _yaml_import_error = exc
else:
    _yaml_import_error = None


@dataclass(slots=True)
class Participant:
    label: str
    type: str  # 'static' | 'llm' | 'rl'
    strategy: object | None = None  # for 'static' and 'llm'
    model: MaskablePPO | None = None  # for 'rl'
    deterministic: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pit configured participants against each other using YAML settings."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML file describing match setup.",
    )
    return parser.parse_args()


def _attach_strategy(player: Player, strategy: object, label: str, rng: random.Random) -> None:
    # For registry strategies: either accept rng via create_instance or default constructor.
    # Here, strategy is already an instance.
    player.strategy = strategy
    player.strategy_name = label  # type: ignore[attr-defined]


def attach_players(game: Game, seats: Sequence[Participant], rng: random.Random) -> tuple[Dict[int, str], Dict[int, Participant]]:
    labels: Dict[int, str] = {}
    rl_assignments: Dict[int, Participant] = {}
    for idx, player in enumerate(game.players):
        # Reset
        for piece in player.pieces:
            piece.position = 0
        player.has_finished = False

        participant = seats[idx]
        labels[idx] = participant.label

        if participant.type in {"static", "llm"} and participant.strategy is not None:
            _attach_strategy(player, participant.strategy, participant.label, rng)
        elif participant.type == "rl":
            player.strategy_name = participant.label  # type: ignore[attr-defined]
            rl_assignments[idx] = participant
        else:
            raise SystemExit(f"Invalid participant configuration at seat {idx}: {participant}")
    return labels, rl_assignments


def determine_rankings(game: Game, finish_order: List[int]) -> List[int]:
    ordered = finish_order.copy()
    total_players = len(game.players)
    remaining = [idx for idx in range(total_players) if idx not in ordered]

    def player_progress(player_index: int) -> tuple[int, int]:
        player = game.players[player_index]
        finished = sum(1 for pc in player.pieces if pc.position == king_config.HOME_FINISH)
        progress = sum(pc.position for pc in player.pieces)
        return finished, progress

    remaining.sort(key=lambda idx: player_progress(idx), reverse=True)
    ordered.extend(remaining)
    return ordered[: total_players]


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


def _decide_with_model(
    participant: Participant,
    board_stack: Any,
    dice_roll: int,
    legal: list[object],
    rng: random.Random,
) -> object | None:
    assert participant.model is not None
    # Build action mask
    action_mask = np.zeros(king_config.PIECES_PER_PLAYER, dtype=bool)
    moves_by_piece: Dict[int, List[object]] = {}
    for mv in legal:
        pid = _extract_piece_index(mv)
        if pid is None:
            continue
        if 0 <= pid < king_config.PIECES_PER_PLAYER:
            action_mask[pid] = True
            moves_by_piece.setdefault(pid, []).append(mv)

    try:
        action, _ = participant.model.predict(
            {"board": board_stack[None, ...], "dice_roll": np.array([[dice_roll - 1]], dtype=np.int64)},
            action_masks=action_mask[None, ...],
            deterministic=participant.deterministic,
        )
        pid = int(action.item())
        candidates = moves_by_piece.get(pid)
        if candidates:
            return rng.choice(candidates)
    except Exception as exc:
        logger.warning(f"Model decision failed for {participant.label}: {exc}")
    return rng.choice(legal) if legal else None


def run_single_game(
    base_rng: random.Random,
    participants: Sequence[Participant],
    *,
    game_index: int,
) -> Dict[str, object]:
    # Seat colors: RED, GREEN, YELLOW, BLUE (default 4 players)
    num = king_config.NUM_PLAYERS
    if num == 2:
        color_ids = [int(Color.RED), int(Color.YELLOW)]
    else:
        color_ids = [int(Color.RED), int(Color.GREEN), int(Color.YELLOW), int(Color.BLUE)][:num]
    players = [Player(color=c) for c in color_ids]
    game = Game(players=players)

    round_seed = base_rng.randint(0, 1_000_000)
    random.seed(round_seed)

    logger.info(
        f"[Game {game_index:02d}] Starting with seed={round_seed} | participants="
        f"{', '.join(p.label for p in participants)}"
    )

    seat_labels, rl_assignments = attach_players(game, participants, base_rng)

    finish_order: List[int] = []
    turns_taken = 0
    current_index = 0
    total_players = len(game.players)

    while turns_taken < king_config.MAX_TURNS and len(finish_order) < total_players:
        label = seat_labels[current_index]
        player = game.players[current_index]

        if player.check_won():
            if current_index not in finish_order:
                finish_order.append(current_index)
            current_index = (current_index + 1) % total_players
            continue

        logger.info(f"[Game {game_index:02d}] Player {label} taking turn.")

        extra = True
        while extra:
            dice = game.roll_dice()
            legal = game.legal_moves(current_index, dice)
            if not legal:
                _log_outcome(game_index, label, None, dice, skipped=True, extra=False)
                extra = False
                continue

            board_stack = game.board.build_tensor(int(player.color))

            if current_index in rl_assignments:
                mv = _decide_with_model(rl_assignments[current_index], board_stack, dice, legal, base_rng)
            else:
                decision = player.choose(board_stack, dice, legal)
                mv = decision if decision is not None else base_rng.choice(legal)

            result = game.apply_move(mv)
            _log_outcome(game_index, label, mv, dice, skipped=False, extra=result.extra_turn)
            extra = result.extra_turn and result.events.move_resolved

            if player.check_won() and current_index not in finish_order:
                finish_order.append(current_index)

        current_index = (current_index + 1) % total_players
        turns_taken += 1

    rankings = determine_rankings(game, finish_order)
    ranking_labels = [seat_labels[idx] for idx in rankings]

    logger.info(
        f"[Game {game_index:02d}] Completed in {turns_taken} turns. Rankings: "
        f"{ranking_labels}"
    )

    return {"turns": turns_taken, "rankings": ranking_labels}


def _log_outcome(
    game_index: int,
    label: str,
    move: Any,
    dice: int,
    *,
    skipped: bool,
    extra: bool,
) -> None:
    if skipped:
        logger.info(f"[Game {game_index:02d}] Player {label} skipped turn (dice={dice}).")
        return
    pid = _extract_piece_index(move)
    extras = []
    if extra:
        extras.append("extra")
    extras_text = f" ({', '.join(extras)})" if extras else ""
    logger.info(
        f"[Game {game_index:02d}] Player {label} moved piece {pid if pid is not None else '?'} with dice {dice}{extras_text}."
    )


def _ensure_yaml_available() -> None:
    if yaml is None:  # pragma: no cover - runtime guard
        raise SystemExit(
            "PyYAML is required for this script. Install it via 'pip install PyYAML'."
        ) from _yaml_import_error


def _load_yaml_config(path: str) -> Dict[str, Any]:
    _ensure_yaml_available()
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise SystemExit("Configuration file must contain a mapping at the top level.")
    logger.info(f"Loaded configuration from {config_path.resolve()}")
    return data


def _iter_participant_entries(config_map: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    if "participants" in config_map:
        participants = config_map["participants"]
        if not isinstance(participants, list):
            raise SystemExit("'participants' must be a list of opponent definitions.")
        for entry in participants:
            if not isinstance(entry, dict):
                raise SystemExit("Each participant entry must be a mapping.")
            yield entry
        return

    keys = sorted(k for k in config_map.keys() if k.lower().startswith("opp"))
    for key in keys:
        value = config_map[key]
        if not isinstance(value, dict):
            raise SystemExit(f"'{key}' must be a mapping describing the participant.")
        entry = dict(value)
        entry.setdefault("label", entry.get("name") or key)
        yield entry


def _load_llm_participant(entry: Dict[str, Any]) -> Participant:
    if init_chat_model is None:
        raise SystemExit("LangChain chat models are unavailable. Install 'langchain'.")

    model_name = entry.get("model")
    if not model_name:
        raise SystemExit("LLM participant requires a 'model' field.")

    provider_kwargs = entry.get("params") or {}
    if not isinstance(provider_kwargs, dict):
        raise SystemExit("LLM 'params' must be a mapping of provider arguments.")

    chat_model = init_chat_model(model_name, **provider_kwargs)
    strategy = LLMStrategy(
        model=chat_model,
        system_prompt=entry.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        max_retries=max(0, int(entry.get("max_retries", 2))),
    )

    label = entry.get("label") or entry.get("name") or "llm"
    logger.info(f"Configured LLM participant '{label}' using model '{model_name}'.")
    return Participant(label=label, type="llm", strategy=strategy)


def _load_rl_participant(
    entry: Dict[str, Any], *, device: str, deterministic_default: bool
) -> Participant:
    checkpoint = entry.get("path")
    if not checkpoint:
        raise SystemExit("RL participant requires a 'path' to the checkpoint file.")

    det_flag = bool(entry.get("deterministic", deterministic_default))
    model_path = Path(checkpoint).expanduser()
    model = MaskablePPO.load(model_path, device=device)
    model.policy.set_training_mode(False)

    label = entry.get("label") or entry.get("name") or model_path.stem or "rl"
    logger.info(
        f"Configured RL participant '{label}' from checkpoint '{model_path}' (deterministic={det_flag})."
    )
    return Participant(label=label, type="rl", model=model, deterministic=det_flag)


def _load_static_participant(entry: Dict[str, Any]) -> Participant:
    strategy_name = entry.get("strategy") or entry.get("name")
    if not strategy_name:
        raise SystemExit("Static participant requires a 'strategy' field.")

    try:
        cls = STRATEGY_REGISTRY[strategy_name]
        try:
            strategy = cls.create_instance(random.Random())
        except NotImplementedError:
            strategy = cls()
    except KeyError as exc:
        raise SystemExit(f"Unknown static strategy '{strategy_name}'.") from exc

    label = entry.get("label") or strategy_name
    logger.info(
        f"Configured static participant '{label}' using strategy '{strategy_name}'."
    )
    return Participant(label=label, type="static", strategy=strategy)


def _load_participants(
    config_map: Dict[str, Any], *, device: str, deterministic: bool
) -> List[Participant]:
    participants: List[Participant] = []
    for entry in _iter_participant_entries(config_map):
        type_name = entry.get("type")
        if not type_name:
            raise SystemExit("Each participant definition must include a 'type'.")

        type_name = str(type_name).lower()
        if type_name == "llm":
            participant = _load_llm_participant(entry)
        elif type_name == "rl":
            participant = _load_rl_participant(
                entry, device=device, deterministic_default=deterministic
            )
        elif type_name == "static":
            participant = _load_static_participant(entry)
        else:
            raise SystemExit(f"Unsupported participant type '{type_name}'.")

        participants.append(participant)

    if len(participants) != king_config.NUM_PLAYERS:
        raise SystemExit(
            f"Configuration must define exactly {king_config.NUM_PLAYERS} participants."
        )

    logger.info(
        "Configured participants: "
        + ", ".join(f"{idx}:{p.label}" for idx, p in enumerate(participants))
    )
    return participants


def main() -> None:
    args = parse_args()

    config_map = _load_yaml_config(args.config)

    n_games = int(config_map.get("n_games", config_map.get("n_game", 1)))
    if n_games <= 0:
        raise SystemExit("'n_games' must be a positive integer.")

    deterministic = bool(config_map.get("deterministic", False))
    device = str(config_map.get("device", "cpu"))
    seed = config_map.get("seed")

    if seed is None:
        logger.info("No seed provided; results will vary between runs.")
    else:
        logger.info(f"Using RNG seed {seed}.")

    rng = random.Random(seed)

    participants = _load_participants(
        config_map, device=device, deterministic=deterministic
    )

    leaderboard: Dict[str, int] = {participant.label: 0 for participant in participants}
    points_board: Dict[str, int] = {
        participant.label: 0 for participant in participants
    }

    logger.info(
        f"Starting tournament for {n_games} game(s) | deterministic={deterministic} | device={device}"
    )

    for game_idx in range(1, n_games + 1):
        result = run_single_game(rng, participants, game_index=game_idx)
        winner = result["rankings"][0]
        leaderboard[winner] += 1
        points_table = POINTS_TABLE[: len(result["rankings"])]
        awarded_points: Dict[str, int] = {}
        for position, name in enumerate(result["rankings"]):
            points = points_table[position] if position < len(points_table) else 0
            points_board[name] += points
            awarded_points[name] = points

        logger.info(
            f"[Game {game_idx:02d}] Winner={winner} | Rankings={result['rankings']} "
            f"| Turns={result['turns']} | Points awarded={awarded_points}"
        )

    logger.info("Tournament complete. Final standings:")
    for label, wins in sorted(
        leaderboard.items(), key=lambda item: (-item[1], item[0])
    ):
        logger.info(f"  {label}: {wins} win(s)")

    logger.info("Aggregated points:")
    for label, points in sorted(
        points_board.items(), key=lambda item: (-item[1], item[0])
    ):
        logger.info(f"  {label}: {points} point(s)")


if __name__ == "__main__":
    main()
