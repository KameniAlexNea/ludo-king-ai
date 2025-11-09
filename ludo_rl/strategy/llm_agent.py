from __future__ import annotations

import importlib
import importlib.util
import random
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional, Sequence

from llm_output_parser import parse_json as _parse_json
from loguru import logger

from .base import BaseStrategy, BaseStrategyConfig
from .types import MoveOption, StrategyContext


def _load_langchain() -> tuple[Any, Any, Any, Any]:  # pragma: no cover - helper
    chat_spec = importlib.util.find_spec("langchain.chat_models")
    msg_spec = importlib.util.find_spec("langchain.messages")
    if chat_spec is None or msg_spec is None:
        raise ImportError(
            "LangChain chat models are not available. Install 'langchain'."
        )
    chat_module = importlib.import_module("langchain.chat_models")
    message_module = importlib.import_module("langchain.messages")
    return (
        getattr(chat_module, "init_chat_model"),
        getattr(message_module, "AIMessage"),
        getattr(message_module, "HumanMessage"),
        getattr(message_module, "SystemMessage"),
    )


try:  # pragma: no cover - optional dependency guard
    init_chat_model, AIMessage, HumanMessage, SystemMessage = _load_langchain()
except ImportError:  # Fallback types enable graceful degradation without LangChain.
    init_chat_model = None  # type: ignore

    class _FallbackMessage(dict):
        def __init__(self, content: str) -> None:
            super().__init__()
            self["content"] = content
            self.content = content

    class SystemMessage(_FallbackMessage):
        pass

    class HumanMessage(_FallbackMessage):
        pass

    class AIMessage(_FallbackMessage):
        pass


DEFAULT_SYSTEM_PROMPT = (
    "You are a seasoned Ludo strategist. Given the current dice roll and legal moves, "
    "choose the single best move for the active player. Respond ONLY with a JSON "
    "object containing the fields 'piece_id' (integer between 0 and 3) and 'reason'."
)

MOVE_TEMPLATE = (
    "Piece {piece_id}: current={current_pos}, new={new_pos}, progress={progress}, "
    "distance_to_goal={distance_to_goal}, capture={can_capture}, capture_count={capture_count}, "
    "enters_home={enters_home}, enters_safe_zone={enters_safe_zone}, forms_blockade={forms_blockade}, "
    "extra_turn={extra_turn}, risk={risk:.2f}, leaving_safe_zone={leaving_safe_zone}"
)


@dataclass(slots=True)
class LLMStrategyConfig(BaseStrategyConfig):
    model: Any
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_retries: int = 2

    def sample(self, rng=None) -> Dict[str, object]:  # noqa: D401
        return {
            "model": self.model,
            "system_prompt": self.system_prompt,
            "max_retries": self.max_retries,
        }


class LLMStrategy(BaseStrategy):
    """Strategy that delegates move selection to a LangChain chat model."""

    name = "llm"
    config: ClassVar[Optional[BaseStrategyConfig]] = None

    def __init__(
        self,
        model: Any,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.max_retries = max(0, max_retries)

    # --- Configuration helpers -------------------------------------------------
    @classmethod
    def configure(
        cls,
        model: Any,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_retries: int = 2,
    ) -> None:
        return cls(
            model=model,
            system_prompt=system_prompt,
            max_retries=max_retries,
        )

    @classmethod
    def configure_with_model_name(
        cls,
        model_name: str,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_retries: int = 2,
        **provider_kwargs,
    ) -> None:
        if init_chat_model is None:
            raise RuntimeError(
                "LangChain is not installed; cannot initialise chat model by name."
            )

        model = init_chat_model(model_name, **provider_kwargs)
        return cls.configure(
            model=model,
            system_prompt=system_prompt,
            max_retries=max_retries,
        )

    # --- Strategy entry point --------------------------------------------------
    def select_move(self, ctx: StrategyContext) -> Optional[MoveOption]:  # type: ignore[override]
        legal_moves = [move for move in ctx.moves if ctx.action_mask[move.piece_id]]
        if not legal_moves:
            return None

        messages = self._build_messages(ctx, legal_moves)

        for _ in range(self.max_retries + 1):
            try:
                response = self.model.invoke(messages)
                logger.debug(f"LLM response: {response}")
            except Exception:
                continue

            piece_id = self._extract_piece_id(response, legal_moves)
            if piece_id is None:
                continue
            for move in legal_moves:
                if move.piece_id == piece_id:
                    return move

        # Fallback: random legal move
        return random.choice(legal_moves)

    # --- Prompt construction ---------------------------------------------------
    def _build_messages(
        self,
        ctx: StrategyContext,
        legal_moves: Sequence[MoveOption],
    ) -> list:
        move_descriptions = [
            MOVE_TEMPLATE.format(
                piece_id=move.piece_id,
                current_pos=move.current_pos,
                new_pos=move.new_pos,
                progress=move.progress,
                distance_to_goal=move.distance_to_goal,
                can_capture=move.can_capture,
                capture_count=move.capture_count,
                enters_home=move.enters_home,
                enters_safe_zone=move.enters_safe_zone,
                forms_blockade=move.forms_blockade,
                extra_turn=move.extra_turn,
                risk=move.risk,
                leaving_safe_zone=move.leaving_safe_zone,
            )
            for move in legal_moves
        ]

        human_prompt = (
            "Dice roll: {dice}\n"
            "Legal moves (choose one):\n- {moves}\n"
            'Respond with JSON, e.g. {{"piece_id": 2, "reason": "Prefer capture"}}'
        ).format(
            dice=ctx.dice_roll,
            moves="\n- ".join(move_descriptions),
        )

        return [
            SystemMessage(self.system_prompt),
            HumanMessage(human_prompt),
        ]

    # --- Response parsing ------------------------------------------------------
    def _extract_piece_id(
        self, response: Any, legal_moves: Sequence[MoveOption]
    ) -> Optional[int]:
        content = self._get_content(response)
        if content is None:
            return None

        try:
            payload = _parse_json(content)
            if isinstance(payload, dict) and "piece_id" in payload:
                piece_id = int(payload.get("piece_id"))
            else:
                piece_id = None
        except Exception:
            piece_id = None

        if piece_id is None:
            return None

        if not any(move.piece_id == piece_id for move in legal_moves):
            return None
        return piece_id

    @staticmethod
    def _get_content(response: Any) -> Optional[str]:
        if isinstance(response, AIMessage):
            return response.content
        if isinstance(response, dict):
            return response.get("content")
        return str(response) if response is not None else None
