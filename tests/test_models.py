"""
Test Data Models and Fixtures for Ludo King AI Testing
Defines structured test cases, scenarios, and expected behaviors.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum


class TestScenario(Enum):
    """Main test scenarios for the Ludo game."""
    GAME_START = "game_start"
    TOKEN_CAPTURE = "token_capture"
    HOME_STRETCH = "home_stretch" 
    STRATEGIC_DECISION = "strategic_decision"
    ENDGAME = "endgame"
    DEFENSIVE_PLAY = "defensive_play"
    AGGRESSIVE_PLAY = "aggressive_play"
    MULTI_OPTION_CHOICE = "multi_option_choice"


class ExpectedBehavior(Enum):
    """Expected AI behavior types."""
    CAPTURE_OPPONENT = "capture_opponent"
    MOVE_TO_SAFETY = "move_to_safety"
    ADVANCE_CLOSEST = "advance_closest"
    BLOCK_OPPONENT = "block_opponent"
    EXIT_HOME = "exit_home"
    ENTER_FINISH = "enter_finish"
    RANDOM_CHOICE = "random_choice"


@dataclass
class TokenPosition:
    """Represents a token's position on the board."""
    token_id: int
    position: int  # -1 = home, 0-55 = board positions, 56+ = finished
    is_safe: bool = False
    can_capture: Optional[int] = None  # opponent token_id that can be captured


@dataclass
class PlayerState:
    """Complete state of a player for testing."""
    color: str
    tokens: List[TokenPosition]
    finished_count: int = 0
    active_count: int = 0
    
    def __post_init__(self):
        self.finished_count = len([t for t in self.tokens if t.position >= 56])
        self.active_count = len([t for t in self.tokens if 0 <= t.position <= 55])


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


@dataclass
class DecisionContext:
    """AI decision context for testing."""
    game_state: GameState
    valid_moves: List[ValidMove]
    strategic_analysis: Dict[str, Any]
    expected_priorities: List[str]  # Priority order for this scenario


@dataclass
class StrategyTestCase:
    """Test case for a specific strategy."""
    strategy_name: str
    scenario: TestScenario
    description: str
    input_context: DecisionContext
    expected_behavior: ExpectedBehavior
    expected_token_id: Optional[int] = None
    alternative_valid_choices: List[int] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class GameFlowTestCase:
    """Test case for complete game flow scenarios."""
    scenario_name: str
    description: str
    initial_setup: GameState
    sequence_of_moves: List[Tuple[int, int]]  # (dice_value, expected_token_id)
    expected_outcome: str
    max_turns: int = 50


class TestDataFactory:
    """Factory for creating structured test data."""
    
    @staticmethod
    def create_game_start_scenario() -> DecisionContext:
        """Test scenario: Game start - first move."""
        game_state = GameState(
            current_player="RED",
            players=[
                PlayerState("RED", [
                    TokenPosition(0, -1), TokenPosition(1, -1),
                    TokenPosition(2, -1), TokenPosition(3, -1)
                ]),
                PlayerState("BLUE", [
                    TokenPosition(0, -1), TokenPosition(1, -1),
                    TokenPosition(2, -1), TokenPosition(3, -1)
                ])
            ],
            turn_count=1,
            dice_value=6
        )
        
        valid_moves = [ValidMove(0, -1, 0, "exit_home")]
        
        return DecisionContext(
            game_state=game_state,
            valid_moves=valid_moves,
            strategic_analysis={"exit_opportunities": 1, "capture_opportunities": 0},
            expected_priorities=["exit_home"]
        )
    
    @staticmethod
    def create_capture_scenario() -> DecisionContext:
        """Test scenario: Token capture opportunity."""
        game_state = GameState(
            current_player="RED", 
            players=[
                PlayerState("RED", [
                    TokenPosition(0, 10), TokenPosition(1, -1),
                    TokenPosition(2, -1), TokenPosition(3, -1)
                ]),
                PlayerState("BLUE", [
                    TokenPosition(0, 14), TokenPosition(1, -1),
                    TokenPosition(2, -1), TokenPosition(3, -1)
                ])
            ],
            turn_count=5,
            dice_value=4
        )
        
        valid_moves = [
            ValidMove(0, 10, 14, "capture", captures_opponent=True),
        ]
        
        return DecisionContext(
            game_state=game_state,
            valid_moves=valid_moves,
            strategic_analysis={"capture_opportunities": 1, "safe_moves": 0},
            expected_priorities=["capture_opponent"]
        )
    
    @staticmethod
    def create_multi_choice_scenario() -> DecisionContext:
        """Test scenario: Multiple strategic options."""
        game_state = GameState(
            current_player="RED",
            players=[
                PlayerState("RED", [
                    TokenPosition(0, 10), TokenPosition(1, 25),
                    TokenPosition(2, 50), TokenPosition(3, -1)
                ]),
                PlayerState("BLUE", [
                    TokenPosition(0, 14), TokenPosition(1, 28),
                    TokenPosition(2, 45), TokenPosition(3, -1)
                ])
            ],
            turn_count=15,
            dice_value=3
        )
        
        valid_moves = [
            ValidMove(0, 10, 13, "normal"),
            ValidMove(1, 25, 28, "capture", captures_opponent=True),
            ValidMove(2, 50, 53, "normal", reaches_safe_spot=True),
            ValidMove(3, -1, 0, "exit_home")
        ]
        
        return DecisionContext(
            game_state=game_state,
            valid_moves=valid_moves,
            strategic_analysis={
                "capture_opportunities": 1,
                "safe_moves": 1,
                "exit_opportunities": 1,
                "advancement_opportunities": 2
            },
            expected_priorities=["capture_opponent", "move_to_safety", "advance_closest", "exit_home"]
        )
    
    @staticmethod
    def create_endgame_scenario() -> DecisionContext:
        """Test scenario: Endgame with finishing opportunity."""
        game_state = GameState(
            current_player="RED",
            players=[
                PlayerState("RED", [
                    TokenPosition(0, 56), TokenPosition(1, 56),
                    TokenPosition(2, 55), TokenPosition(3, 50)
                ]),
                PlayerState("BLUE", [
                    TokenPosition(0, 56), TokenPosition(1, 45),
                    TokenPosition(2, 40), TokenPosition(3, -1)
                ])
            ],
            turn_count=45,
            dice_value=1
        )
        
        valid_moves = [
            ValidMove(2, 55, 56, "finish", enters_finish=True),
            ValidMove(3, 50, 51, "normal")
        ]
        
        return DecisionContext(
            game_state=game_state,
            valid_moves=valid_moves,
            strategic_analysis={
                "finish_opportunities": 1,
                "advancement_opportunities": 1,
                "winning_position": True
            },
            expected_priorities=["enter_finish", "advance_closest"]
        )
    
    @staticmethod
    def create_defensive_scenario() -> DecisionContext:
        """Test scenario: Defensive play needed."""
        game_state = GameState(
            current_player="RED",
            players=[
                PlayerState("RED", [
                    TokenPosition(0, 12), TokenPosition(1, 25),
                    TokenPosition(2, 30), TokenPosition(3, -1)
                ]),
                PlayerState("BLUE", [
                    TokenPosition(0, 56), TokenPosition(1, 56),
                    TokenPosition(2, 9), TokenPosition(3, 27)
                ])
            ],
            turn_count=25,
            dice_value=3
        )
        
        valid_moves = [
            ValidMove(0, 12, 15, "normal"),  # Could be captured next turn
            ValidMove(1, 25, 28, "normal"),  # Safe advancement 
            ValidMove(2, 30, 33, "normal", reaches_safe_spot=True),  # To safety
        ]
        
        return DecisionContext(
            game_state=game_state,
            valid_moves=valid_moves,
            strategic_analysis={
                "danger_level": "high",
                "safe_moves": 1,
                "threatened_tokens": [0],
                "opponent_threat_level": 0.8
            },
            expected_priorities=["move_to_safety", "advance_closest"]
        )


class StrategyBehaviorProfiles:
    """Expected behavior profiles for each AI strategy."""
    
    KILLER_PROFILE = {
        "primary_focus": "capture_opponent",
        "secondary_focus": "advance_closest", 
        "avoids": ["defensive_moves"],
        "risk_tolerance": "high",
        "decision_pattern": "aggressive"
    }
    
    WINNER_PROFILE = {
        "primary_focus": "enter_finish",
        "secondary_focus": "advance_closest",
        "avoids": ["risky_moves"],
        "risk_tolerance": "medium",
        "decision_pattern": "goal_oriented"
    }
    
    OPTIMIST_PROFILE = {
        "primary_focus": "advance_closest",
        "secondary_focus": "exit_home",
        "avoids": ["defensive_moves"],
        "risk_tolerance": "high",
        "decision_pattern": "forward_thinking"
    }
    
    DEFENSIVE_PROFILE = {
        "primary_focus": "move_to_safety",
        "secondary_focus": "block_opponent",
        "avoids": ["risky_advances"],
        "risk_tolerance": "low",
        "decision_pattern": "safety_first"
    }
    
    BALANCED_PROFILE = {
        "primary_focus": "context_dependent",
        "secondary_focus": "advance_closest",
        "avoids": ["extreme_risks"],
        "risk_tolerance": "medium",
        "decision_pattern": "adaptive"
    }
    
    CAUTIOUS_PROFILE = {
        "primary_focus": "move_to_safety",
        "secondary_focus": "slow_advancement",
        "avoids": ["any_risk"],
        "risk_tolerance": "very_low",
        "decision_pattern": "ultra_conservative"
    }
    
    RANDOM_PROFILE = {
        "primary_focus": "random_choice",
        "secondary_focus": "random_choice",
        "avoids": [],
        "risk_tolerance": "variable",
        "decision_pattern": "unpredictable"
    }


class TestScenarioBuilder:
    """Builder for creating complex test scenarios."""
    
    def __init__(self):
        self.test_cases = []
    
    def add_strategy_test(self, strategy_name: str, scenario: TestScenario, 
                         context: DecisionContext, expected_behavior: ExpectedBehavior,
                         expected_token: Optional[int] = None) -> 'TestScenarioBuilder':
        """Add a strategy-specific test case."""
        test_case = StrategyTestCase(
            strategy_name=strategy_name,
            scenario=scenario,
            description=f"{strategy_name} strategy in {scenario.value} scenario",
            input_context=context,
            expected_behavior=expected_behavior,
            expected_token_id=expected_token
        )
        self.test_cases.append(test_case)
        return self
    
    def build_all_strategy_tests(self) -> List[StrategyTestCase]:
        """Build comprehensive test suite for all strategies."""
        all_tests = []
        scenarios = [
            (TestDataFactory.create_game_start_scenario(), TestScenario.GAME_START),
            (TestDataFactory.create_capture_scenario(), TestScenario.TOKEN_CAPTURE), 
            (TestDataFactory.create_multi_choice_scenario(), TestScenario.MULTI_OPTION_CHOICE),
            (TestDataFactory.create_endgame_scenario(), TestScenario.ENDGAME),
            (TestDataFactory.create_defensive_scenario(), TestScenario.DEFENSIVE_PLAY)
        ]
        
        strategies = ["killer", "winner", "optimist", "defensive", "balanced", "cautious", "random"]
        
        for context, scenario in scenarios:
            for strategy in strategies:
                expected_behavior = self._get_expected_behavior(strategy, scenario)
                expected_token = self._get_expected_token(strategy, context)
                
                test_case = StrategyTestCase(
                    strategy_name=strategy,
                    scenario=scenario,
                    description=f"{strategy} strategy handling {scenario.value}",
                    input_context=context,
                    expected_behavior=expected_behavior,
                    expected_token_id=expected_token,
                    reasoning=self._get_reasoning(strategy, scenario)
                )
                all_tests.append(test_case)
        
        return all_tests
    
    def _get_expected_behavior(self, strategy: str, scenario: TestScenario) -> ExpectedBehavior:
        """Determine expected behavior based on strategy and scenario."""
        behavior_map = {
            "killer": {
                TestScenario.TOKEN_CAPTURE: ExpectedBehavior.CAPTURE_OPPONENT,
                TestScenario.GAME_START: ExpectedBehavior.EXIT_HOME,
                TestScenario.ENDGAME: ExpectedBehavior.ENTER_FINISH,
                TestScenario.MULTI_OPTION_CHOICE: ExpectedBehavior.CAPTURE_OPPONENT,
                TestScenario.DEFENSIVE_PLAY: ExpectedBehavior.ADVANCE_CLOSEST
            },
            "winner": {
                TestScenario.ENDGAME: ExpectedBehavior.ENTER_FINISH,
                TestScenario.TOKEN_CAPTURE: ExpectedBehavior.CAPTURE_OPPONENT,
                TestScenario.GAME_START: ExpectedBehavior.EXIT_HOME,
                TestScenario.MULTI_OPTION_CHOICE: ExpectedBehavior.ADVANCE_CLOSEST,
                TestScenario.DEFENSIVE_PLAY: ExpectedBehavior.ADVANCE_CLOSEST
            },
            "defensive": {
                TestScenario.DEFENSIVE_PLAY: ExpectedBehavior.MOVE_TO_SAFETY,
                TestScenario.TOKEN_CAPTURE: ExpectedBehavior.CAPTURE_OPPONENT,
                TestScenario.ENDGAME: ExpectedBehavior.ENTER_FINISH,
                TestScenario.GAME_START: ExpectedBehavior.EXIT_HOME,
                TestScenario.MULTI_OPTION_CHOICE: ExpectedBehavior.MOVE_TO_SAFETY
            },
            "cautious": {
                TestScenario.DEFENSIVE_PLAY: ExpectedBehavior.MOVE_TO_SAFETY,
                TestScenario.MULTI_OPTION_CHOICE: ExpectedBehavior.MOVE_TO_SAFETY,
                TestScenario.TOKEN_CAPTURE: ExpectedBehavior.MOVE_TO_SAFETY,
                TestScenario.ENDGAME: ExpectedBehavior.ENTER_FINISH,
                TestScenario.GAME_START: ExpectedBehavior.EXIT_HOME
            },
            "random": {
                TestScenario.TOKEN_CAPTURE: ExpectedBehavior.RANDOM_CHOICE,
                TestScenario.GAME_START: ExpectedBehavior.EXIT_HOME,
                TestScenario.ENDGAME: ExpectedBehavior.RANDOM_CHOICE,
                TestScenario.MULTI_OPTION_CHOICE: ExpectedBehavior.RANDOM_CHOICE,
                TestScenario.DEFENSIVE_PLAY: ExpectedBehavior.RANDOM_CHOICE
            }
        }
        
        # Default fallback
        default_behaviors = {
            TestScenario.GAME_START: ExpectedBehavior.EXIT_HOME,
            TestScenario.TOKEN_CAPTURE: ExpectedBehavior.CAPTURE_OPPONENT,
            TestScenario.ENDGAME: ExpectedBehavior.ENTER_FINISH,
            TestScenario.MULTI_OPTION_CHOICE: ExpectedBehavior.ADVANCE_CLOSEST,
            TestScenario.DEFENSIVE_PLAY: ExpectedBehavior.ADVANCE_CLOSEST
        }
        
        return behavior_map.get(strategy, {}).get(scenario, default_behaviors[scenario])
    
    def _get_expected_token(self, strategy: str, context: DecisionContext) -> Optional[int]:
        """Get expected token choice for strategy."""
        valid_moves = context.valid_moves
        
        if not valid_moves:
            return None
            
        # Strategy-specific token selection logic
        if strategy == "killer":
            # Prefer capture moves
            for move in valid_moves:
                if move.captures_opponent:
                    return move.token_id
        
        elif strategy == "winner":
            # Prefer finishing moves
            for move in valid_moves:
                if move.enters_finish:
                    return move.token_id
        
        elif strategy in ["defensive", "cautious"]:
            # Prefer safe moves
            for move in valid_moves:
                if move.reaches_safe_spot:
                    return move.token_id
        
        # Default: first available move
        return valid_moves[0].token_id if valid_moves else None
    
    def _get_reasoning(self, strategy: str, scenario: TestScenario) -> str:
        """Get reasoning for expected behavior."""
        reasoning_map = {
            ("killer", TestScenario.TOKEN_CAPTURE): "Aggressive strategy prioritizes capturing opponents",
            ("winner", TestScenario.ENDGAME): "Goal-oriented strategy focuses on finishing tokens",
            ("defensive", TestScenario.DEFENSIVE_PLAY): "Safety-first approach moves to secure positions",
            ("cautious", TestScenario.MULTI_OPTION_CHOICE): "Ultra-conservative play avoids all risks",
            ("optimist", TestScenario.GAME_START): "Forward-thinking strategy advances aggressively"
        }
        
        return reasoning_map.get((strategy, scenario), f"{strategy} strategy standard behavior")


# Export key classes for testing
__all__ = [
    'TestScenario', 'ExpectedBehavior', 'TokenPosition', 'PlayerState', 
    'GameState', 'ValidMove', 'DecisionContext', 'StrategyTestCase',
    'GameFlowTestCase', 'TestDataFactory', 'StrategyBehaviorProfiles',
    'TestScenarioBuilder'
]
