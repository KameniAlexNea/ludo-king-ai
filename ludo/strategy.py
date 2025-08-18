"""
Strategic decision-making system for Ludo AI.
Different AI personalities with distinct playing styles.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import random


class Strategy(ABC):
    """
    Abstract base class for Ludo AI strategies.
    Each strategy implements a different playing style.
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def decide(self, game_context: Dict) -> int:
        """
        Make a strategic decision based on game context.
        
        Args:
            game_context: Complete game state and available moves
            
        Returns:
            int: token_id to move (0-3)
        """
        pass
    
    def _get_valid_moves(self, game_context: Dict) -> List[Dict]:
        """Helper to get valid moves from context."""
        return game_context.get('valid_moves', [])
    
    def _get_move_by_type(self, valid_moves: List[Dict], move_type: str) -> Optional[Dict]:
        """Get first move of specified type."""
        for move in valid_moves:
            if move['move_type'] == move_type:
                return move
        return None
    
    def _get_moves_by_type(self, valid_moves: List[Dict], move_type: str) -> List[Dict]:
        """Get all moves of specified type."""
        return [move for move in valid_moves if move['move_type'] == move_type]
    
    def _get_capture_moves(self, valid_moves: List[Dict]) -> List[Dict]:
        """Get all moves that capture opponents."""
        return [move for move in valid_moves if move['captures_opponent']]
    
    def _get_safe_moves(self, valid_moves: List[Dict]) -> List[Dict]:
        """Get all safe moves."""
        return [move for move in valid_moves if move['is_safe_move']]
    
    def _get_risky_moves(self, valid_moves: List[Dict]) -> List[Dict]:
        """Get all risky moves."""
        return [move for move in valid_moves if not move['is_safe_move']]
    
    def _get_highest_value_move(self, valid_moves: List[Dict]) -> Optional[Dict]:
        """Get move with highest strategic value."""
        if not valid_moves:
            return None
        return max(valid_moves, key=lambda m: m['strategic_value'])
    
    def _get_lowest_value_move(self, valid_moves: List[Dict]) -> Optional[Dict]:
        """Get move with lowest strategic value."""
        if not valid_moves:
            return None
        return min(valid_moves, key=lambda m: m['strategic_value'])


class KillerStrategy(Strategy):
    """
    Aggressive strategy focused on capturing opponents.
    Prioritizes offensive moves and disrupting opponents.
    """
    
    def __init__(self):
        super().__init__(
            "Killer",
            "Aggressive strategy that prioritizes capturing opponents and blocking their progress"
        )
    
    def decide(self, game_context: Dict) -> int:
        valid_moves = self._get_valid_moves(game_context)
        
        if not valid_moves:
            return 0
        
        # Priority 1: Capture opponents (highest priority)
        capture_moves = self._get_capture_moves(valid_moves)
        if capture_moves:
            # Choose capture that affects the most threatening opponent
            best_capture = max(capture_moves, key=lambda m: len(m.get('captured_tokens', [])))
            return best_capture['token_id']
        
        # Priority 2: Finish tokens (secure points)
        finish_move = self._get_move_by_type(valid_moves, 'finish')
        if finish_move:
            return finish_move['token_id']
        
        # Priority 3: Exit home with 6 to get more pieces in play
        exit_move = self._get_move_by_type(valid_moves, 'exit_home')
        if exit_move:
            return exit_move['token_id']
        
        # Priority 4: Risky moves for aggressive positioning
        risky_moves = self._get_risky_moves(valid_moves)
        if risky_moves:
            # Choose the most aggressive risky move
            best_risky = self._get_highest_value_move(risky_moves)
            return best_risky['token_id']
        
        # Fallback: Highest value move
        best_move = self._get_highest_value_move(valid_moves)
        return best_move['token_id']


class WinnerStrategy(Strategy):
    """
    Victory-focused strategy that prioritizes finishing tokens.
    Plays conservatively but efficiently toward winning.
    """
    
    def __init__(self):
        super().__init__(
            "Winner",
            "Victory-focused strategy that prioritizes finishing tokens and safe progression"
        )
    
    def decide(self, game_context: Dict) -> int:
        valid_moves = self._get_valid_moves(game_context)
        
        if not valid_moves:
            return 0
        
        # Priority 1: Finish tokens (main goal)
        finish_move = self._get_move_by_type(valid_moves, 'finish')
        if finish_move:
            return finish_move['token_id']
        
        # Priority 2: Advance in home column (close to finishing)
        home_column_moves = self._get_moves_by_type(valid_moves, 'advance_home_column')
        if home_column_moves:
            # Choose the one closest to finishing
            best_home = max(home_column_moves, key=lambda m: m['strategic_value'])
            return best_home['token_id']
        
        # Priority 3: Capture only if very safe or strategically important
        capture_moves = self._get_capture_moves(valid_moves)
        safe_captures = [m for m in capture_moves if m['is_safe_move']]
        if safe_captures:
            return safe_captures[0]['token_id']
        
        # Priority 4: Safe moves toward home
        safe_moves = self._get_safe_moves(valid_moves)
        if safe_moves:
            best_safe = self._get_highest_value_move(safe_moves)
            return best_safe['token_id']
        
        # Priority 5: Exit home only if needed
        exit_move = self._get_move_by_type(valid_moves, 'exit_home')
        if exit_move:
            return exit_move['token_id']
        
        # Fallback: Most conservative move
        best_move = self._get_highest_value_move(valid_moves)
        return best_move['token_id']


class OptimistStrategy(Strategy):
    """
    Optimistic strategy that takes calculated risks.
    Believes in favorable outcomes and plays boldly.
    """
    
    def __init__(self):
        super().__init__(
            "Optimist",
            "Optimistic strategy that takes calculated risks and plays boldly for big gains"
        )
    
    def decide(self, game_context: Dict) -> int:
        valid_moves = self._get_valid_moves(game_context)
        
        if not valid_moves:
            return 0
        
        # Priority 1: High-value risky moves (optimistic about outcomes)
        risky_moves = self._get_risky_moves(valid_moves)
        high_value_risky = [m for m in risky_moves if m['strategic_value'] >= 10.0]
        if high_value_risky:
            best_risky = self._get_highest_value_move(high_value_risky)
            return best_risky['token_id']
        
        # Priority 2: Capture moves (confident about not being captured back)
        capture_moves = self._get_capture_moves(valid_moves)
        if capture_moves:
            best_capture = self._get_highest_value_move(capture_moves)
            return best_capture['token_id']
        
        # Priority 3: Exit home aggressively
        exit_move = self._get_move_by_type(valid_moves, 'exit_home')
        if exit_move:
            return exit_move['token_id']
        
        # Priority 4: Finish tokens when available
        finish_move = self._get_move_by_type(valid_moves, 'finish')
        if finish_move:
            return finish_move['token_id']
        
        # Priority 5: Highest value move overall
        best_move = self._get_highest_value_move(valid_moves)
        return best_move['token_id']


class DefensiveStrategy(Strategy):
    """
    Defensive strategy that prioritizes safety and protection.
    Avoids risks and plays conservatively.
    """
    
    def __init__(self):
        super().__init__(
            "Defensive",
            "Conservative strategy that prioritizes safety and avoids unnecessary risks"
        )
    
    def decide(self, game_context: Dict) -> int:
        valid_moves = self._get_valid_moves(game_context)
        
        if not valid_moves:
            return 0
        
        # Priority 1: Finish tokens (completely safe)
        finish_move = self._get_move_by_type(valid_moves, 'finish')
        if finish_move:
            return finish_move['token_id']
        
        # Priority 2: Safe moves only
        safe_moves = self._get_safe_moves(valid_moves)
        if safe_moves:
            # Among safe moves, prefer advancing in home column
            home_column_safe = [m for m in safe_moves if m['move_type'] == 'advance_home_column']
            if home_column_safe:
                return self._get_highest_value_move(home_column_safe)['token_id']
            
            # Otherwise, best safe move
            best_safe = self._get_highest_value_move(safe_moves)
            return best_safe['token_id']
        
        # Priority 3: Capture only if opponent is very threatening
        opponents = game_context.get('opponents', [])
        high_threat_opponents = [opp for opp in opponents if opp.get('threat_level', 0) > 0.7]
        
        if high_threat_opponents:
            capture_moves = self._get_capture_moves(valid_moves)
            if capture_moves:
                return capture_moves[0]['token_id']
        
        # Priority 4: Exit home only if no other choice
        exit_move = self._get_move_by_type(valid_moves, 'exit_home')
        if exit_move:
            return exit_move['token_id']
        
        # Fallback: Lowest risk move
        best_move = self._get_lowest_value_move(valid_moves)
        return best_move['token_id']


class BalancedStrategy(Strategy):
    """
    Balanced strategy that adapts based on game situation.
    Switches between offensive and defensive play as needed.
    """
    
    def __init__(self):
        super().__init__(
            "Balanced",
            "Adaptive strategy that balances offense and defense based on game situation"
        )
    
    def decide(self, game_context: Dict) -> int:
        valid_moves = self._get_valid_moves(game_context)
        
        if not valid_moves:
            return 0
        
        player_state = game_context.get('player_state', {})
        opponents = game_context.get('opponents', [])
        
        # Analyze game situation
        my_progress = player_state.get('finished_tokens', 0) / 4.0
        opponent_max_progress = max([opp.get('tokens_finished', 0) for opp in opponents], default=0) / 4.0
        behind = my_progress < opponent_max_progress - 0.25  # Significantly behind
        ahead = my_progress > opponent_max_progress + 0.25   # Significantly ahead
        
        # Priority 1: Always finish when possible
        finish_move = self._get_move_by_type(valid_moves, 'finish')
        if finish_move:
            return finish_move['token_id']
        
        # Adaptive strategy based on position
        if ahead:
            # Play defensively when ahead
            return self._defensive_choice(valid_moves)
        elif behind:
            # Play aggressively when behind
            return self._aggressive_choice(valid_moves)
        else:
            # Balanced play when even
            return self._balanced_choice(valid_moves, game_context)
    
    def _defensive_choice(self, valid_moves: List[Dict]) -> int:
        """Make defensive choice when ahead."""
        safe_moves = self._get_safe_moves(valid_moves)
        if safe_moves:
            return self._get_highest_value_move(safe_moves)['token_id']
        return self._get_highest_value_move(valid_moves)['token_id']
    
    def _aggressive_choice(self, valid_moves: List[Dict]) -> int:
        """Make aggressive choice when behind."""
        capture_moves = self._get_capture_moves(valid_moves)
        if capture_moves:
            return self._get_highest_value_move(capture_moves)['token_id']
        
        risky_moves = self._get_risky_moves(valid_moves)
        if risky_moves:
            return self._get_highest_value_move(risky_moves)['token_id']
        
        return self._get_highest_value_move(valid_moves)['token_id']
    
    def _balanced_choice(self, valid_moves: List[Dict], game_context: Dict) -> int:
        """Make balanced choice when even."""
        # Capture if opportunity is good
        capture_moves = self._get_capture_moves(valid_moves)
        if capture_moves:
            return capture_moves[0]['token_id']
        
        # Exit home if needed
        exit_move = self._get_move_by_type(valid_moves, 'exit_home')
        if exit_move:
            return exit_move['token_id']
        
        # Otherwise, best strategic move
        return self._get_highest_value_move(valid_moves)['token_id']


class RandomStrategy(Strategy):
    """
    Random strategy for baseline comparison.
    Makes completely random valid moves.
    """
    
    def __init__(self):
        super().__init__(
            "Random",
            "Baseline strategy that makes random valid moves"
        )
    
    def decide(self, game_context: Dict) -> int:
        valid_moves = self._get_valid_moves(game_context)
        
        if not valid_moves:
            return 0
        
        # Completely random choice
        random_move = random.choice(valid_moves)
        return random_move['token_id']


class CautiousStrategy(Strategy):
    """
    Very cautious strategy that avoids all risks.
    Only makes moves that are guaranteed safe.
    """
    
    def __init__(self):
        super().__init__(
            "Cautious",
            "Extremely conservative strategy that only makes guaranteed safe moves"
        )
    
    def decide(self, game_context: Dict) -> int:
        valid_moves = self._get_valid_moves(game_context)
        
        if not valid_moves:
            return 0
        
        # Priority 1: Finish tokens
        finish_move = self._get_move_by_type(valid_moves, 'finish')
        if finish_move:
            return finish_move['token_id']
        
        # Priority 2: Home column moves (always safe)
        home_moves = self._get_moves_by_type(valid_moves, 'advance_home_column')
        if home_moves:
            return self._get_highest_value_move(home_moves)['token_id']
        
        # Priority 3: Only very safe moves
        safe_moves = self._get_safe_moves(valid_moves)
        very_safe = [m for m in safe_moves if m['strategic_value'] <= 10.0]  # Conservative values
        if very_safe:
            return self._get_highest_value_move(very_safe)['token_id']
        
        # Priority 4: Exit home only if absolutely necessary
        player_state = game_context.get('player_state', {})
        active_tokens = player_state.get('active_tokens', 0)
        
        if active_tokens == 0:  # Must exit home
            exit_move = self._get_move_by_type(valid_moves, 'exit_home')
            if exit_move:
                return exit_move['token_id']
        
        # Fallback: Least risky move
        if safe_moves:
            return self._get_lowest_value_move(safe_moves)['token_id']
        
        return self._get_lowest_value_move(valid_moves)['token_id']


# Strategy Factory
class StrategyFactory:
    """Factory class for creating strategy instances."""
    
    _strategies = {
        'killer': KillerStrategy,
        'winner': WinnerStrategy,
        'optimist': OptimistStrategy,
        'defensive': DefensiveStrategy,
        'balanced': BalancedStrategy,
        'random': RandomStrategy,
        'cautious': CautiousStrategy
    }
    
    @classmethod
    def create_strategy(cls, strategy_name: str) -> Strategy:
        """
        Create a strategy instance by name.
        
        Args:
            strategy_name: Name of the strategy to create
            
        Returns:
            Strategy: Instance of the requested strategy
            
        Raises:
            ValueError: If strategy name is not recognized
        """
        strategy_name = strategy_name.lower()
        if strategy_name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {available}")
        
        return cls._strategies[strategy_name]()
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available strategy names."""
        return list(cls._strategies.keys())
    
    @classmethod
    def get_strategy_descriptions(cls) -> Dict[str, str]:
        """Get descriptions of all available strategies."""
        descriptions = {}
        for name, strategy_class in cls._strategies.items():
            strategy = strategy_class()
            descriptions[name] = strategy.description
        return descriptions


# Usage example and testing
if __name__ == "__main__":
    # Example of how to use strategies
    print("Available Strategies:")
    print("=" * 50)
    
    descriptions = StrategyFactory.get_strategy_descriptions()
    for name, desc in descriptions.items():
        print(f"{name.upper()}: {desc}")
    
    print("\nExample Usage:")
    print("=" * 50)
    
    # Create different strategies
    killer = StrategyFactory.create_strategy('killer')
    winner = StrategyFactory.create_strategy('winner')
    
    print(f"Created {killer.name} strategy: {killer.description}")
    print(f"Created {winner.name} strategy: {winner.description}")
    
    # Mock game context for testing
    mock_context = {
        'valid_moves': [
            {
                'token_id': 0,
                'move_type': 'advance_main_board',
                'strategic_value': 8.0,
                'is_safe_move': True,
                'captures_opponent': False
            },
            {
                'token_id': 1,
                'move_type': 'advance_main_board',
                'strategic_value': 12.0,
                'is_safe_move': False,
                'captures_opponent': True,
                'captured_tokens': [{'player_color': 'blue', 'token_id': 2}]
            }
        ],
        'player_state': {'finished_tokens': 1, 'active_tokens': 2},
        'opponents': [{'tokens_finished': 0, 'threat_level': 0.3}]
    }
    
    print(f"\nKiller strategy chooses token: {killer.decide(mock_context)}")
    print(f"Winner strategy chooses token: {winner.decide(mock_context)}")
