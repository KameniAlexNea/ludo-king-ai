"""Run matches defined by a YAML configuration file."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from loguru import logger
from sb3_contrib import MaskablePPO

from ludo_rl.ludo.config import config
from ludo_rl.ludo.game import LudoGame
from ludo_rl.strategy.base import BaseStrategy
from ludo_rl.strategy.llm_agent import (
    DEFAULT_SYSTEM_PROMPT,
    LLMStrategy,
    init_chat_model,
)
from ludo_rl.strategy.registry import create as create_strategy
from ludo_rl.strategy.rl_agent import RLStrategy

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
    strategy: BaseStrategy


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


def attach_players(game: LudoGame, seats: Sequence[Participant]) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    for idx, player in enumerate(game.players):
        player.reset()
        participant = seats[idx]
        player.strategy_name = participant.strategy.name
        player._strategy = participant.strategy
        labels[idx] = participant.label
    return labels


def determine_rankings(game: LudoGame, finish_order: List[int]) -> List[int]:
    ordered = finish_order.copy()
    remaining = [idx for idx in range(config.NUM_PLAYERS) if idx not in ordered]

    def player_progress(player_index: int) -> tuple[int, int]:
        player = game.players[player_index]
        finished = sum(piece.position == 57 for piece in player.pieces)
        progress = sum(piece.position for piece in player.pieces)
        return finished, progress

    remaining.sort(key=lambda idx: player_progress(idx), reverse=True)
    ordered.extend(remaining)
    return ordered[: config.NUM_PLAYERS]


def run_single_game(
    base_rng: random.Random,
    participants: Sequence[Participant],
    *,
    game_index: int,
) -> Dict[str, object]:
    game = LudoGame()

    round_seed = base_rng.randint(0, 1_000_000)
    random.seed(round_seed)
    game.rng.seed(round_seed)

    logger.info(
        f"[Game {game_index:02d}] Starting with seed={round_seed} | participants="
        f"{', '.join(p.label for p in participants)}"
    )

    seat_labels = attach_players(game, participants)

    finish_order: List[int] = []
    turns_taken = 0
    current_index = 0

    while turns_taken < config.MAX_TURNS and len(finish_order) < config.NUM_PLAYERS:
        label = seat_labels[current_index]
        player = game.players[current_index]

        if player.has_won():
            if current_index not in finish_order:
                finish_order.append(current_index)
            current_index = (current_index + 1) % config.NUM_PLAYERS
            continue

        logger.info(f"[Game {game_index:02d}] Player {label} taking turn.")
        outcome = game.take_turn(current_index, rng=base_rng)
        _log_outcome(game_index, label, outcome)

        if not outcome.skipped and player.has_won() and current_index not in finish_order:
            finish_order.append(current_index)

        while (
            not outcome.skipped
            and outcome.extra_turn
            and len(finish_order) < config.NUM_PLAYERS
        ):
            logger.info(f"[Game {game_index:02d}] Player {label} earned an extra turn.")
            outcome = game.take_turn(current_index, rng=base_rng)
            _log_outcome(game_index, label, outcome)
            if (
                not outcome.skipped
                and player.has_won()
                and current_index not in finish_order
            ):
                finish_order.append(current_index)

        current_index = (current_index + 1) % config.NUM_PLAYERS
        turns_taken += 1

    rankings = determine_rankings(game, finish_order)
    ranking_labels = [seat_labels[idx] for idx in rankings]

    logger.info(
        f"[Game {game_index:02d}] Completed in {turns_taken} turns. Rankings: "
        f"{ranking_labels}"
    )

    return {
        "turns": turns_taken,
        "rankings": ranking_labels,
    }


def _log_outcome(game_index: int, label: str, outcome: Any) -> None:
    if outcome.skipped:
        logger.info(
            f"[Game {game_index:02d}] Player {label} skipped turn (dice={outcome.dice_roll})."
        )
        return

    move = outcome.move or {}
    result = outcome.result or {}

    piece = move.get("piece")
    new_pos = move.get("new_pos")
    dice = move.get("dice_roll")

    events = result.get("events", {}) if isinstance(result, dict) else {}
    extras = []
    if events.get("knockouts"):
        extras.append("knockout")
    if events.get("finished"):
        extras.append("finish")
    if outcome.extra_turn:
        extras.append("extra")
    extras_text = f" ({', '.join(extras)})" if extras else ""

    logger.info(
        f"[Game {game_index:02d}] Player {label} moved piece {getattr(piece, 'piece_id', '?')} "
        f"to {new_pos} with dice {dice}{extras_text}."
    )


def _ensure_yaml_available() -> None:
    if yaml is None:  # pragma: no cover - runtime guard
        raise SystemExit(
            "PyYAML is required for this script. Install it via 'pip install pyyaml'."
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
    return Participant(label=label, strategy=strategy)


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
    strategy = RLStrategy(model=model, deterministic=det_flag)

    label = entry.get("label") or entry.get("name") or model_path.stem or "rl"
    logger.info(
        f"Configured RL participant '{label}' from checkpoint '{model_path}'"
        f" (deterministic={det_flag})."
    )
    return Participant(label=label, strategy=strategy)


def _load_static_participant(entry: Dict[str, Any]) -> Participant:
    strategy_name = entry.get("strategy") or entry.get("name")
    if not strategy_name:
        raise SystemExit("Static participant requires a 'strategy' field.")

    try:
        strategy = create_strategy(strategy_name)
    except KeyError as exc:
        raise SystemExit(f"Unknown static strategy '{strategy_name}'.") from exc

    label = entry.get("label") or strategy_name
    logger.info(f"Configured static participant '{label}' using strategy '{strategy_name}'.")
    return Participant(label=label, strategy=strategy)


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

    if len(participants) != config.NUM_PLAYERS:
        raise SystemExit(
            f"Configuration must define exactly {config.NUM_PLAYERS} participants."
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

    logger.info(
        f"Starting tournament for {n_games} game(s) | deterministic={deterministic} "
        f"| device={device}"
    )

    for game_idx in range(1, n_games + 1):
        result = run_single_game(rng, participants, game_index=game_idx)
        winner = result["rankings"][0]
        leaderboard[winner] += 1
        logger.info(
            f"[Game {game_idx:02d}] Winner={winner} | Rankings={result['rankings']} "
            f"| Turns={result['turns']}"
        )

    logger.info("Tournament complete. Final standings:")
    for label, wins in sorted(leaderboard.items(), key=lambda item: (-item[1], item[0])):
        logger.info(f"  {label}: {wins} win(s)")


if __name__ == "__main__":
    main()
