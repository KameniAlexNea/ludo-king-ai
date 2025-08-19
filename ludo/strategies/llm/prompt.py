PROMPT = """You are playing Ludo. Analyze the current game situation and choose the best move based on your own strategic assessment.

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
