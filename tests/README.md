# Ludo King AI - Structured Test Suite Documentation

## Overview

This document describes the comprehensive test suite for the Ludo King AI project, including test data models, scenarios, and expected behaviors for strategic AI validation.

## Test Architecture

### 1. Test Data Models (`tests/test_models.py`)

#### Core Data Structures

```python
@dataclass
class TokenPosition:
    """Represents a token's position on the board."""
    token_id: int
    position: int  # -1 = home, 0-55 = board positions, 56+ = finished
    is_safe: bool = False
    can_capture: Optional[int] = None

@dataclass
class PlayerState:
    """Complete state of a player for testing."""
    color: str
    tokens: List[TokenPosition]
    finished_count: int = 0
    active_count: int = 0

@dataclass
class GameState:
    """Complete game state for test scenarios."""
    current_player: str
    players: List[PlayerState]
    turn_count: int
    dice_value: int
    consecutive_sixes: int = 0
    game_over: bool = False

@dataclass
class ValidMove:
    """Represents a valid move option."""
    token_id: int
    from_position: int
    to_position: int
    move_type: str  # "normal", "capture", "enter_home_stretch", "finish", "exit_home"
    captures_opponent: bool = False
    reaches_safe_spot: bool = False
    enters_finish: bool = False
```

#### Test Scenarios

| Scenario | Description | Purpose |
|----------|-------------|---------|
| `GAME_START` | Initial game state with all tokens in home | Test basic token deployment |
| `TOKEN_CAPTURE` | Opportunity to capture opponent token | Test aggressive vs defensive behavior |
| `HOME_STRETCH` | Tokens approaching finish line | Test endgame priority decisions |
| `STRATEGIC_DECISION` | Multiple viable move options | Test strategy differentiation |
| `ENDGAME` | Close to winning with finish opportunities | Test winning focus |
| `DEFENSIVE_PLAY` | Vulnerable position requiring safety | Test risk management |
| `AGGRESSIVE_PLAY` | Opportunity for offensive moves | Test aggression levels |
| `MULTI_OPTION_CHOICE` | Complex scenario with 3+ valid moves | Test strategic priorities |

#### Expected Behaviors

| Behavior | Description | Applicable Strategies |
|----------|-------------|---------------------|
| `CAPTURE_OPPONENT` | Prioritize capturing enemy tokens | Killer, Winner (opportunistic) |
| `MOVE_TO_SAFETY` | Move tokens to safe positions | Defensive, Cautious |
| `ADVANCE_CLOSEST` | Move token closest to finish | Balanced, Optimist |
| `BLOCK_OPPONENT` | Position to block opponent progress | Defensive |
| `EXIT_HOME` | Get tokens out of home base | All (when rolling 6) |
| `ENTER_FINISH` | Move tokens into finish area | Winner, Balanced |
| `RANDOM_CHOICE` | Unpredictable decision making | Random |

### 2. Test Framework (`tests/test_framework.py`)

#### Core Test Classes

- **`LudoTestFramework`**: Base class with common test utilities
- **`StrategyTestSuite`**: Tests for AI strategy validation
- **`GameFlowTestSuite`**: Tests for complete game scenarios
- **`PerformanceTestSuite`**: Performance and scalability tests

#### Key Test Methods

```python
def assertValidMove(self, token_id: int, valid_moves: List[Dict])
    """Assert that token_id corresponds to a valid move."""

def assertStrategyBehavior(self, strategy: Strategy, context: Dict, expected_behavior: ExpectedBehavior)
    """Assert that strategy exhibits expected behavior."""
```

### 3. Unit Tests (`tests/test_unit.py`)

#### Component Test Coverage

- **`TokenTestCase`**: Token state management, movement, capture
- **`PlayerTestCase`**: Player initialization, token counting, win conditions
- **`BoardTestCase`**: Board setup, position properties, capture detection
- **`GameLogicTestCase`**: Game flow, dice rolling, move validation
- **`IntegrationTestCase`**: Complete game scenarios with AI strategies

## Strategy Behavior Profiles

### 1. Killer Strategy
- **Primary Focus**: Capture opponents
- **Secondary Focus**: Advance aggressively
- **Risk Tolerance**: High
- **Expected Behavior**: Always chooses capture moves when available

### 2. Winner Strategy  
- **Primary Focus**: Finish tokens
- **Secondary Focus**: Strategic advancement
- **Risk Tolerance**: Medium
- **Expected Behavior**: Prioritizes tokens closest to finish line

### 3. Optimist Strategy
- **Primary Focus**: Forward advancement
- **Secondary Focus**: Exit home quickly
- **Risk Tolerance**: High
- **Expected Behavior**: Takes risks for faster progress

### 4. Defensive Strategy
- **Primary Focus**: Token safety
- **Secondary Focus**: Blocking opponents
- **Risk Tolerance**: Low
- **Expected Behavior**: Moves to safe positions when available

### 5. Balanced Strategy
- **Primary Focus**: Context-dependent decisions
- **Secondary Focus**: Adaptive play
- **Risk Tolerance**: Medium
- **Expected Behavior**: Weighs multiple factors for optimal moves

### 6. Cautious Strategy
- **Primary Focus**: Extreme safety
- **Secondary Focus**: Slow, secure advancement
- **Risk Tolerance**: Very Low
- **Expected Behavior**: Avoids any risk of capture

### 7. Random Strategy
- **Primary Focus**: Unpredictable choices
- **Secondary Focus**: Random selection
- **Risk Tolerance**: Variable
- **Expected Behavior**: Shows variability across multiple runs

## Test Execution

### Running Tests

```bash
# Run all tests
python tests/run_tests.py --type all --verbose

# Run specific test types
python tests/run_tests.py --type unit
python tests/run_tests.py --type framework  
python tests/run_tests.py --type integration
python tests/run_tests.py --type performance

# Run strategy behavior analysis
python tests/run_tests.py --analysis
```

### Using Tox

```bash
# Run with coverage
tox -e test

# Run linting
tox -e lint

# Run formatting
tox -e format

# Run all environments
tox
```

### Direct Test Execution

```bash
# Run specific test file
python -m pytest tests/test_unit.py -v

# Run with coverage
coverage run -m unittest discover -s tests -p "test_*.py"
coverage report -m
```

## Test Scenarios in Detail

### 1. Game Start Scenario

**Setup:**
- All tokens in home
- Dice roll: 6
- Valid moves: Exit home with token 0

**Expected Behaviors:**
- All strategies: Choose exit_home move
- Reasoning: Only legal move available

### 2. Token Capture Scenario

**Setup:**
- Player token at position 10
- Opponent token at position 14  
- Dice roll: 4
- Valid moves: Capture opponent

**Expected Behaviors:**
- Killer: Always capture
- Defensive: Capture if safe
- Random: Variable choice

### 3. Multi-Choice Scenario

**Setup:**
- Multiple tokens in play
- Dice roll: 3
- Valid moves: Normal advance, Capture, Move to safety, Exit home

**Expected Behaviors:**
- Killer: Prioritize capture
- Defensive: Prioritize safety
- Winner: Prioritize advancement
- Balanced: Context-dependent choice

### 4. Endgame Scenario

**Setup:**
- 2 tokens finished, 1 at position 55, 1 at position 50
- Dice roll: 1
- Valid moves: Finish token, Advance token

**Expected Behaviors:**
- Winner: Choose finish move
- All others: Should also prioritize finishing

### 5. Defensive Scenario

**Setup:**
- Tokens vulnerable to capture
- Safe position available
- Dice roll: 3

**Expected Behaviors:**
- Defensive/Cautious: Move to safety
- Killer/Optimist: May take risks
- Balanced: Assess threat level

## Input/Output Data Models

### Input: Game Context
```python
{
    'current_situation': {
        'player_color': str,
        'dice_value': int,
        'consecutive_sixes': int,
        'turn_count': int
    },
    'player_state': {
        'tokens_home': int,
        'tokens_active': int,
        'tokens_finished': int
    },
    'opponents': [
        {
            'color': str,
            'tokens_finished': int,
            'tokens_active': int,
            'threat_level': float
        }
    ],
    'valid_moves': [
        {
            'token_id': int,
            'from_position': int,
            'to_position': int,
            'move_type': str,
            'captures_opponent': bool,
            'reaches_safe_spot': bool,
            'enters_finish': bool
        }
    ],
    'strategic_analysis': {
        'capture_opportunities': int,
        'safe_moves': int,
        'finish_opportunities': int,
        'danger_level': str
    }
}
```

### Output: Strategy Decision
```python
{
    'selected_token_id': int,  # 0-3, which token to move
    'confidence': float,       # Optional: decision confidence
    'reasoning': str          # Optional: decision explanation
}
```

## Expected Method Signatures

### Strategy Interface
```python
class Strategy(ABC):
    def decide(self, game_context: Dict) -> int:
        """
        Make strategic decision based on game context.
        
        Args:
            game_context: Complete game state and available moves
            
        Returns:
            int: token_id to move (0-3)
        """
        pass
```

### Test Validation Methods
```python
def validate_decision(strategy_name: str, context: Dict, decision: int) -> bool:
    """Validate that decision aligns with strategy behavior profile."""

def measure_consistency(strategy_name: str, context: Dict, runs: int = 10) -> float:
    """Measure decision consistency across multiple runs."""

def analyze_behavior_pattern(strategy_name: str, scenarios: List[Dict]) -> Dict:
    """Analyze strategy behavior across multiple scenarios."""
```

## Performance Benchmarks

### Expected Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Decision Speed | < 10ms per decision | Average time for strategy.decide() |
| Memory Usage | < 100MB total | Memory footprint during testing |
| Game Completion | < 200 turns | Average turns to finish game |
| Strategy Consistency | > 95% same scenario | Non-random strategies only |
| Test Coverage | > 90% code coverage | All game components |

### Scalability Tests

- **Tournament Simulation**: 100+ games between strategies
- **Concurrent Decisions**: Multiple strategies deciding simultaneously  
- **Stress Testing**: 1000+ rapid decisions
- **Memory Leaks**: Long-running games and repeated initializations

## Integration with CI/CD

### Automated Testing Pipeline

1. **Unit Tests**: Core component validation
2. **Strategy Tests**: AI behavior verification
3. **Integration Tests**: Complete game scenarios
4. **Performance Tests**: Speed and memory benchmarks
5. **Coverage Report**: Code coverage analysis
6. **Lint/Format**: Code quality checks

### Test Reports

- **JUnit XML**: For CI integration
- **Coverage Reports**: HTML and terminal output
- **Performance Metrics**: JSON format for tracking
- **Strategy Analysis**: Behavior pattern documentation

This test suite provides comprehensive validation of the Ludo King AI system, ensuring reliable game mechanics and predictable strategy behaviors while maintaining performance standards.
