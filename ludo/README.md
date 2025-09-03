# Ludo Core Engine

Pure Python implementation of the Ludo board game used by RL environments (`ludo_rl` / `ludo_rls`) and heuristic strategies.

## Goals
* Deterministic, testable game rules
* Clean separation of mechanics (board, tokens, players) and strategy layer
* Minimal external dependencies (standard library only)
* Extensible strategy framework (classic heuristics + LLM-driven variants)

## Module Overview
```
ludo/
	board.py        # Board logic: movement validation, capture resolution
	constants.py    # Game, board, color, path constants & helper values
	game.py         # Orchestrates turns, dice, move validation/execution
	player.py       # Player container holding tokens & strategy binding
	token.py        # Token state & movement logic
	strategy.py     # StrategyFactory & registry helpers
	strategies/     # Concrete strategies (heuristics + LLM interface)
		base.py
		random_strategy.py
		killer.py / defensive.py / balanced.py / cautious.py / optimist.py / winner.py
		probabilistic*.py (experimental probabilistic decision heuristics)
		llm_strategy.py + llm/ (prompt + adapter for external LLM selection)
```

## Core Data Model
| Concept  | Responsibility |
|--------- | -------------- |
| Token    | Encapsulates a single piece's position, finish state, movement legality |
| Player   | Owns 4 tokens, color identity, strategy reference |
| Board    | Spatial rules: safe squares, capture detection, home column logic |
| LudoGame | High-level flow: dice rolling, move enumeration, executing & tracking history |

Positions:
* `-1` = home (not on board)
* `0..(MAIN_BOARD_SIZE-1)` = circular path
* `HOME_COLUMN_START .. FINISH_POSITION` = home column progression
* `FINISH_POSITION` (or token.is_finished()) = token completed

## Turn Lifecycle
1. Current player (by index) rolls dice (`roll_dice()`) – tracks consecutive sixes.
2. Valid moves enumerated via `get_valid_moves(player, dice_value)`:
	 * Filters impossible moves (blocked, exceeds finish)
	 * Annotates capture metadata: `captured_tokens`, `captures_opponent`
3. Strategy (or external policy) chooses a move (`token_id`).
4. `execute_move()` performs legality check, updates token & board, handles captures.
5. Extra turn awarded if: rolled six, captured, or finished a token (unless 3 consecutive sixes rule breaks chain).
6. Win check: if player finished all tokens -> `game_over` + `winner` set.
7. If no extra turn and game not over -> `next_turn()` advances index & increments turn counter.

## Move Result Schema (subset)
```python
{
	'success': True,
	'player_color': 'red',
	'token_id': 2,
	'dice_value': 6,
	'old_position': -1,
	'new_position': 0,
	'captured_tokens': [{'player_color': 'blue', 'token_id': 1}],
	'token_finished': False,
	'extra_turn': True,
	'game_won': False
}
```

## Strategy Framework
Interface (`strategies/base.py`):
```python
class Strategy(ABC):
		def decide(self, game_context: Dict) -> int: ...  # returns token_id 0..3
```
Helper methods provided (capture filters, safety classification, value ranking). Concrete strategies implement prioritized selection. The factory (`StrategyFactory`) exposes discovery:
```python
from ludo.strategy import StrategyFactory
names = StrategyFactory.get_available_strategies()
strategy = StrategyFactory.create_strategy('killer')
```

### Included Heuristic Styles
| Name        | Focus |
|------------ | ----- |
| random      | Uniform random valid token |
| killer      | Aggressive capture prioritization |
| defensive   | Safety & avoidance |
| balanced    | Mixed risk/reward heuristic |
| cautious    | Minimal exposure |
| optimist    | Progress & finishing emphasis |
| winner      | Finishing acceleration |
| probabilistic(_v2/_v3) | Experimental probability-weighted decisions |
| llm_strategy | Delegates move selection to LLM prompt interface |

## Integrating With RL
RL environments wrap `LudoGame` and replace heuristic `decide` with policy inference. Core game remains single-threaded & deterministic given RNG seed.

Key extension points:
* Observation extraction (in RL packages) pulls token positions, finished counts, capture opportunities.
* Reward shaping modules inspect `move_result` fields (captures, finish, extra_turn).

## Adding a New Strategy
1. Create `ludo/strategies/my_strategy.py`
2. Subclass `Strategy` and implement `decide()` using helpers.
3. Register in `ludo/strategies/__init__.py` STRATEGIES dict.
4. (Optional) Add description for discovery via factory.

## Determinism & Testing
Game randomness isolated to Python's `random`. Seed before constructing or call `random.seed(...)` prior to environment reset to reproduce sequences.

## LLM Strategy Notes
`llm_strategy.py` + `strategies/llm/` provide prompt templates and adapter layer. Excluded by default in factory listings when `avoid_llm=True`.

## Performance Considerations
* Minimal allocations inside hot loops (tokens pre-instantiated)
* Move validation vectorizable if needed; current per-move iteration sufficient for standard training scales
* Capture resolution centralized in `Board.execute_move` to simplify reward logic upstream

## License
See root `LICENSE`.

