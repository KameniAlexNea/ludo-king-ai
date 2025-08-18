"""
Strategic decision-making system for Ludo AI.
Strategy factory and main interface.
"""

from typing import List, Dict
from .strategies import Strategy, STRATEGIES


# Strategy Factory
class StrategyFactory:
    """Factory class for creating strategy instances."""

    _strategies = STRATEGIES

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
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. Available: {available}"
            )

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
    killer = StrategyFactory.create_strategy("killer")
    winner = StrategyFactory.create_strategy("winner")

    print(f"Created {killer.name} strategy: {killer.description}")
    print(f"Created {winner.name} strategy: {winner.description}")

    # Mock game context for testing
    mock_context = {
        "valid_moves": [
            {
                "token_id": 0,
                "move_type": "advance_main_board",
                "strategic_value": 8.0,
                "is_safe_move": True,
                "captures_opponent": False,
            },
            {
                "token_id": 1,
                "move_type": "advance_main_board",
                "strategic_value": 12.0,
                "is_safe_move": False,
                "captures_opponent": True,
                "captured_tokens": [{"player_color": "blue", "token_id": 2}],
            },
        ],
        "player_state": {"finished_tokens": 1, "active_tokens": 2},
        "opponents": [{"tokens_finished": 0, "threat_level": 0.3}],
    }

    print(f"\nKiller strategy chooses token: {killer.decide(mock_context)}")
    print(f"Winner strategy chooses token: {winner.decide(mock_context)}")
