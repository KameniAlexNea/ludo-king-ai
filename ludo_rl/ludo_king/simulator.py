from __future__ import annotations

from dataclasses import dataclass, field

from .game import Game
from .types import Move


@dataclass(slots=True)
class Simulator:
    agent_index: int = 0
    game: Game = field(init=False)

    def __post_init__(self) -> None:
        # Expect Game to be constructed by caller with players and strategies.
        # Provide a simple default with random strategies if not injected.
        raise RuntimeError(
            "Simulator requires an explicit Game instance; construct with Simulator.for_game(Game)."
        )

    @classmethod
    def for_game(cls, game: Game, agent_index: int = 0) -> "Simulator":
        obj = object.__new__(cls)
        obj.agent_index = agent_index
        obj.game = game
        return obj

    def step(self, agent_move: Move) -> tuple[bool, bool]:
        """Apply agent move, then simulate opponents unless extra turn.

        Returns (terminated, extra_turn_for_agent)
        """
        res = self.game.apply_move(agent_move)
        extra = res.extra_turn

        # simulate others if no extra turn
        if not extra:
            total_players = len(self.game.players)
            idx = (self.agent_index + 1) % total_players
            while idx != self.agent_index:
                if self.game.players[idx].check_won():
                    idx = (idx + 1) % total_players
                    continue
                dice = self.game.roll_dice()
                legals = self.game.legal_moves(idx, dice)
                # Build agent-relative board stack for this player color
                agent_color = int(self.game.players[idx].color)
                board_stack = self.game.board.build_tensor(agent_color)
                mv = self.game.players[idx].choose(board_stack, dice, legals)
                if mv is not None:
                    self.game.apply_move(mv)
                idx = (idx + 1) % total_players

        terminated = self.game.players[self.agent_index].check_won()
        return terminated, extra
