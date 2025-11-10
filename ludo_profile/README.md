# Ludo Agent Profiling System

Analyze agent behavior in Ludo games to generate behavioral profiles.

## Features

- **Behavioral Metrics**: Aggression, risk-taking, exploration, finishing focus, blockade usage, defensiveness
- **Style Classification**: Aggressive, defensive, explorer, finisher, blockader, opportunist, balanced
- **Segmentation**: Fixed windows, game phases (opening/midgame/endgame), or adaptive
- **Detailed Reports**: Human-readable behavior descriptions and quantitative characteristics

## Quick Start

```python
from ludo_profile import GameAnalyzer
from ludo_profile.trace_builder import build_trace_from_dict

# Load game trace
trace_data = {
    "game_id": "game_001",
    "num_players": 4,
    "players": [
        {"index": 0, "color": "RED", "strategy": "rl_agent"},
        {"index": 1, "color": "BLUE", "strategy": "defensive"},
        # ...
    ],
    "moves": [
        {
            "step": 1,
            "player_index": 0,
            "dice_roll": 6,
            "piece_id": 2,
            "old_position": 0,
            "new_position": 1,
            "events": {"exited_home": True, "knockouts": [], ...},
            "extra_turn": True,
        },
        # ...
    ],
    "winner": 0,
}

trace = build_trace_from_dict(trace_data)

# Analyze
analyzer = GameAnalyzer()
profile = analyzer.analyze_game(trace, player_index=0, segmentation="phase")

# View results
print(f"Dominant style: {profile.overall_summary.dominant_style}")
for segment in profile.profile_segments:
    print(f"Steps {segment.step_range[0]}-{segment.step_range[1]}: {segment.style}")
    print(f"  Aggression: {segment.characteristics.aggression:.2f}")
    print(f"  Behaviors: {', '.join(segment.behaviors)}")

# Export to JSON
import json
print(json.dumps(profile.to_dict(), indent=2))
```

## Segmentation Strategies

### Fixed Window
```python
profile = analyzer.analyze_game(trace, player_index=0, 
                                segmentation="fixed", window_size=20)
```
Splits moves into fixed-size windows (default: 20 moves per segment).

### Game Phase
```python
profile = analyzer.analyze_game(trace, player_index=0, segmentation="phase")
```
Segments by game phase:
- **Opening**: Until all pieces out of home
- **Midgame**: Middle portion with most action
- **Endgame**: From first piece finishing to end

### Adaptive (Future)
```python
profile = analyzer.analyze_game(trace, player_index=0, segmentation="adaptive")
```
Detects behavioral change points automatically.

## Behavior Characteristics

Each segment is analyzed for:

- **Aggression** (0-1): Capture attempts / capture opportunities
- **Risk-taking** (0-1): Risky moves / risky situations
- **Exploration** (0-1): Unique pieces moved / 4 total pieces
- **Finishing** (0-1): Focus on advancing lead pieces
- **Blockade Usage** (0-1): Blockades formed / opportunities
- **Defensiveness** (0-1): Moves to safety / exposed situations

## Style Classifications

Profiles are classified into styles:

- **Aggressive**: High aggression, risk-taking; low defensiveness
- **Defensive**: High defensiveness; low aggression, risk-taking
- **Explorer**: Moves all pieces evenly
- **Finisher**: Focuses on advancing lead pieces
- **Blockader**: Forms blockades strategically
- **Opportunist**: Balanced aggression with tactical blocking
- **Balanced**: Moderate across all dimensions

## Building Traces

### From Dictionary (JSON)
```python
from ludo_profile.trace_builder import build_trace_from_dict
trace = build_trace_from_dict(json_data)
```

### From Game Instance
```python
from ludo_profile.trace_builder import build_trace_from_game

# During game, collect move history
move_history = []
# ... play game, append each move as:
# {"player": idx, "dice": roll, "piece": piece_id, 
#  "old": old_pos, "new": new_pos, "events": events_dict, "extra": bool}

trace = build_trace_from_game(game, move_history)
```

## Future: LLM Integration

Placeholder for LLM-based profiling:

```python
# TODO: Implement LLM profiler
from ludo_profile.llm_profiler import LLMProfiler

llm_profiler = LLMProfiler(model="gpt-4")
profile = llm_profiler.analyze_game(trace, player_index=0)
```

The LLM profiler will:
- Generate natural language summaries
- Identify complex behavioral patterns
- Provide strategic recommendations
- Compare to known strategies

## Output Format

Profiles can be exported to JSON:

```json
{
  "game_id": "game_001",
  "player_index": 0,
  "player_color": "RED",
  "player_strategy": "rl_agent",
  "profile_segments": [
    {
      "step_range": [1, 30],
      "style": "defensive",
      "confidence": 0.85,
      "characteristics": {
        "aggression": 0.2,
        "risk_taking": 0.15,
        "exploration": 0.8,
        "finishing": 0.1,
        "blockade_usage": 0.3,
        "defensiveness": 0.75
      },
      "behaviors": [
        "Got 3 piece(s) out of home",
        "Prioritized safety, moved to safe squares 8 time(s)",
        "Spread pieces evenly (moved 3/4 pieces)"
      ],
      "moves_count": 15,
      "captures_made": 1,
      "got_captured": 2,
      "blockades_formed": 1,
      "pieces_finished": 0
    }
  ],
  "overall_summary": {
    "dominant_style": "defensive",
    "style_distribution": {"defensive": 0.7, "finisher": 0.3},
    "key_transitions": [
      {"step": 75, "from": "defensive", "to": "finisher", "trigger": "piece_finished"}
    ],
    "total_captures": 3,
    "total_got_captured": 5,
    "total_blockades": 2,
    "total_finished": 4,
    "win_achieved": true
  }
}
```
