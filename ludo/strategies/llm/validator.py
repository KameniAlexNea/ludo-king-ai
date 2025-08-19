from typing import Dict
from .llm_exceptions import LLMValidationError


class GameContextValidator:
    """Validates game context to prevent injection and ensure data integrity."""

    @staticmethod
    def validate(game_context: Dict) -> None:
        """Validate game context. Raises LLMValidationError if invalid."""
        if not isinstance(game_context, dict):
            raise LLMValidationError("game_context must be a dictionary")

        # Check required fields
        required_fields = ["player_state", "opponents", "valid_moves"]
        for field in required_fields:
            if field not in game_context:
                raise LLMValidationError(f"Missing required field: {field}")

        # Validate player_state
        player_state = game_context["player_state"]
        if not isinstance(player_state, dict):
            raise LLMValidationError("player_state must be a dictionary")

        # Validate numeric fields with strict bounds
        finished_tokens = player_state.get("finished_tokens", 0)
        if not isinstance(finished_tokens, int) or not 0 <= finished_tokens <= 4:
            raise LLMValidationError(f"Invalid finished_tokens: {finished_tokens}")

        home_tokens = player_state.get("home_tokens", 0)
        if not isinstance(home_tokens, int) or not 0 <= home_tokens <= 4:
            raise LLMValidationError(f"Invalid home_tokens: {home_tokens}")

        # Validate opponents
        opponents = game_context["opponents"]
        if not isinstance(opponents, list) or len(opponents) > 3:
            raise LLMValidationError("Invalid opponents structure")

        for i, opp in enumerate(opponents):
            if not isinstance(opp, dict):
                raise LLMValidationError(f"Opponent {i} must be a dictionary")
            tokens_finished = opp.get("tokens_finished", 0)
            if not isinstance(tokens_finished, int) or not 0 <= tokens_finished <= 4:
                raise LLMValidationError(
                    f"Invalid tokens_finished for opponent {i}: {tokens_finished}"
                )

        # Validate moves
        valid_moves = game_context["valid_moves"]
        if not isinstance(valid_moves, list) or not valid_moves:
            raise LLMValidationError("valid_moves must be a non-empty list")

        for i, move in enumerate(valid_moves):
            if not isinstance(move, dict):
                raise LLMValidationError(f"Move {i} must be a dictionary")

            token_id = move.get("token_id")
            if not isinstance(token_id, int) or not 0 <= token_id <= 3:
                raise LLMValidationError(f"Invalid token_id in move {i}: {token_id}")

            move_type = move.get("move_type", "")
            if not isinstance(move_type, str) or len(move_type) > 50:
                raise LLMValidationError(f"Invalid move_type in move {i}")
