from .types import MoveOption, StrategyContext


class BaseStrategy:
    """Base class for all strategies."""

    name: str = "base"

    def select_move(self, ctx: StrategyContext) -> MoveOption:
        """Select a move from the list of valid moves.

        Args:
            ctx (StrategyContext): The context containing valid moves and observation.
        Returns:
            MoveOption: The selected move.
        """
        raise NotImplementedError
