PROMPT = """You are playing Ludo. Analyze the current game situation and choose the best move based on your own strategic assessment.

LUDO RULES:
- OBJECTIVE: Move all 4 tokens around the board and into your home column first
- STARTING: Roll a 6 to move tokens out of home onto the starting square
- MOVEMENT: Move clockwise around the outer path by exact die count
- CAPTURING: Landing on opponent's token sends it back to their home (gives extra turn)
- SAFE SQUARES: Colored squares (your color) and star-marked squares prevent capture
- STACKING: Your own tokens can stack together and move as a group (cannot be captured)
- HOME COLUMN: After completing circuit, enter your colored home column by exact count
- EXTRA TURNS: Rolling 6, capturing opponent, or getting token home gives extra turn
- WINNING: First to get all 4 tokens into home column wins

GAME SITUATION:
- My progress: {my_progress}/4 tokens finished, {my_home_tokens} at home, {my_active_tokens} active
- Opponents' progress: {opponent_progress} (max: {max_opponent_progress}/4)
- Game phase: {game_phase}

AVAILABLE MOVES:
{moves_text}

MOVE TYPES EXPLAINED:
- CAPTURES OPPONENT: This move will send an opponent's token back to their home
- SAFE: This move positions your token in a safe spot where it cannot be captured
- RISKY: This move puts your token in a position where opponents might capture it

Analyze the situation and develop your own strategy. Consider the current game state, your position relative to opponents, and the potential outcomes of each move.

Choose the token ID (0-3) for your move. Respond with ONLY the token number.

DECISION: """


def create_prompt(game_context: dict, valid_moves: list[dict]) -> str:
    """Create structured prompt for LLM decision making with sanitized data."""
    # valid_moves = self._get_valid_moves(game_context)
    player_state: dict = game_context.get("player_state", {})
    opponents: list[dict] = game_context.get("opponents", [])

    # Build moves information safely (data already validated)
    moves_info = []
    for i, move in enumerate(valid_moves):
        token_id = move.get("token_id", 0)
        move_type = move.get("move_type", "unknown")
        strategic_value = move.get("strategic_value", 0.0)

        move_desc = f"Token {token_id}: {move_type} (value: {strategic_value:.2f})"

        if move.get("captures_opponent"):
            move_desc += " [CAPTURES OPPONENT]"
        if move.get("is_safe_move"):
            move_desc += " [SAFE]"
        else:
            move_desc += " [RISKY]"

        moves_info.append(move_desc)

    # Extract game state data (already validated)
    my_progress = player_state.get("finished_tokens", 0)
    my_home_tokens = player_state.get("home_tokens", 0)
    my_active_tokens = max(0, 4 - my_home_tokens - my_progress)

    # Extract opponent data (already validated)
    opponent_progress = [opp.get("tokens_finished", 0) for opp in opponents]
    max_opponent_progress = max(opponent_progress, default=0)

    # Determine game phase
    if my_progress == 0:
        game_phase = "Early"
    elif my_progress < 3:
        game_phase = "Mid"
    else:
        game_phase = "End"

    # Create prompt with validated data
    moves_text = "\n".join(f"{i + 1}. {move}" for i, move in enumerate(moves_info))

    prompt = PROMPT.format(
        my_progress=my_progress,
        my_home_tokens=my_home_tokens,
        my_active_tokens=my_active_tokens,
        opponent_progress=opponent_progress,
        max_opponent_progress=max_opponent_progress,
        game_phase=game_phase,
        moves_text=moves_text,
    )

    return prompt
