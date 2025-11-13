from __future__ import annotations

import random
from dataclasses import dataclass, field

from .game import Game
from .types import Move, MoveResult


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

    def _update_transition_summaries(
        self, mover_index: int, move: Move, result: MoveResult
    ) -> None:
        """Update board transition summaries based on a move result."""
        agent_player = self.game.players[self.agent_index]
        agent_color = int(agent_player.color)

        # Update movement heatmap at the destination position
        if result.new_position > 0:
            # Convert mover's relative position to agent's relative position
            if mover_index == self.agent_index:
                agent_rel_pos = result.new_position
            else:
                mover_color = int(self.game.players[mover_index].color)
                # Only track main track movements (1-51)
                if 1 <= result.new_position <= 51:
                    abs_pos = self.game.board.absolute_position(
                        mover_color, result.new_position
                    )
                    agent_rel_pos = self.game.board.relative_position(
                        agent_color, abs_pos
                    )
                else:
                    agent_rel_pos = -1

            if agent_rel_pos != -1:
                self.game.board.movement_heatmap[agent_rel_pos] += 1.0
                # Add reward at this position
                if result.rewards and mover_index in result.rewards:
                    self.game.board.reward_heatmap[agent_rel_pos] += result.rewards[
                        mover_index
                    ]

        # Track knockouts
        if result.events.knockouts:
            for knockout in result.events.knockouts:
                abs_pos = knockout.get("abs_pos")
                if abs_pos is not None:
                    agent_rel_pos = self.game.board.relative_position(
                        agent_color, abs_pos
                    )
                    if agent_rel_pos != -1:
                        if knockout["player"] == self.agent_index:
                            # Opponent knocked out my piece
                            self.game.board.opp_knockouts[agent_rel_pos] = 1.0
                        elif mover_index == self.agent_index:
                            # I knocked out opponent piece
                            self.game.board.my_knockouts[agent_rel_pos] = 1.0

        # Track new blockades
        if result.events.blockades:
            for blockade in result.events.blockades:
                blockade_player = blockade.get("player", mover_index)
                blockade_rel_pos = blockade.get(
                    "rel"
                )  # Note: field is "rel" in game.py
                if blockade_rel_pos and 1 <= blockade_rel_pos <= 51:
                    blockade_color = int(self.game.players[blockade_player].color)
                    abs_pos = self.game.board.absolute_position(
                        blockade_color, blockade_rel_pos
                    )
                    agent_rel_pos = self.game.board.relative_position(
                        agent_color, abs_pos
                    )
                    if agent_rel_pos != -1:
                        self.game.board.new_blockades[agent_rel_pos] = 1.0

    def step(self, agent_move: Move) -> tuple[bool, bool]:
        """Apply agent move, then simulate opponents unless extra turn.

        Returns (terminated, extra_turn_for_agent)
        """
        # Reset transition summaries at the start of a new turn cycle
        self.game.board.reset_transition_summaries()

        # Apply agent's move
        res = self.game.apply_move(agent_move)
        self._update_transition_summaries(self.agent_index, agent_move, res)
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
                    opp_res = self.game.apply_move(mv)
                    self._update_transition_summaries(idx, mv, opp_res)
                idx = (idx + 1) % total_players

        terminated = self.game.players[self.agent_index].check_won()
        return terminated, extra

    def step_opponents_only(self) -> None:
        """Simulate all opponents in turn order until it returns to the agent.

        Handles extra turns for opponents according to game rules.
        """
        # Reset transition summaries when agent had no valid moves
        self.game.board.reset_transition_summaries()

        total = len(self.game.players)
        idx = (self.agent_index + 1) % total
        while idx != self.agent_index:
            player = self.game.players[idx]
            if player.check_won():
                idx = (idx + 1) % total
                continue

            extra = True
            while extra:
                dice = self.game.roll_dice()
                legal = self.game.legal_moves(idx, dice)
                if not legal:
                    extra = False
                    continue
                agent_color = int(player.color)
                board_stack = self.game.board.build_tensor(agent_color)
                decision = None
                if hasattr(player, "choose"):
                    try:
                        decision = player.choose(board_stack, dice, legal)
                    except Exception:
                        decision = None
                mv = decision if decision is not None else random.choice(legal)
                result = self.game.apply_move(mv)
                self._update_transition_summaries(idx, mv, result)
                extra = result.extra_turn and result.events.move_resolved

            idx = (idx + 1) % total
