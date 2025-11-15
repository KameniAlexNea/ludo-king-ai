from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from .game import Game
from .types import Move, MoveResult


@dataclass(slots=True)
class Simulator:
    agent_index: int = 0
    game: Game = field(init=False)
    # Token-sequence observation buffers
    history_T: int = 10
    _pos_hist: np.ndarray = field(default=None, init=False, repr=False)
    _dice_hist: np.ndarray = field(default=None, init=False, repr=False)
    _mask_hist: np.ndarray = field(default=None, init=False, repr=False)
    _player_hist: np.ndarray = field(default=None, init=False, repr=False)
    _token_colors: np.ndarray = field(default=None, init=False, repr=False)
    _token_exists_mask: np.ndarray = field(default=None, init=False, repr=False)
    _hist_len: int = field(default=0, init=False, repr=False)
    _hist_ptr: int = field(default=0, init=False, repr=False)
    _agent_reward_acc: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        # Expect Game to be constructed by caller with players and strategies.
        # Provide a simple default with random strategies if not injected.
        raise RuntimeError(
            "Simulator requires an explicit Game instance; construct with Simulator.for_game(Game)."
        )

    @classmethod
    def for_game(cls, game: Game, agent_index: int = 0) -> "Simulator":
        """
        Create a Simulator instance for a given Game and agent index.

        :param game: The Game instance to simulate.
        :type game: Game
        :param agent_index: The index of the agent to simulate for, defaults to 0
        :type agent_index: int, optional
        :return: A Simulator instance configured for the given game and agent.
        :rtype: Simulator
        """
        obj = object.__new__(cls)
        obj.agent_index = agent_index
        obj.game = game
        # Initialize token sequence buffers directly
        obj.history_T = 10
        agent_color = int(game.players[agent_index].color)
        obj._pos_hist = np.zeros((10, 16), dtype=np.int64)
        obj._dice_hist = np.zeros((10,), dtype=np.int64)
        obj._mask_hist = np.zeros((10, 16), dtype=np.bool_)
        obj._player_hist = np.zeros((10,), dtype=np.int64)
        obj._token_colors = game.board.token_colors(agent_color)
        obj._token_exists_mask = game.board.token_exists_mask(agent_color)
        obj._hist_len = 0
        obj._hist_ptr = 0
        obj._agent_reward_acc = 0.0
        return obj

    # --- Token sequence observation helpers ---

    def _append_history(self, dice: int, player_idx: int) -> None:
        agent_color = int(self.game.players[self.agent_index].color)
        frame_pos = self.game.board.all_token_positions(agent_color)
        i = self._hist_ptr
        self._pos_hist[i, :] = frame_pos
        self._dice_hist[i] = int(dice)
        self._mask_hist[i, :] = self._token_exists_mask
        self._player_hist[i] = int(player_idx)
        self._hist_ptr = (self._hist_ptr + 1) % self.history_T
        self._hist_len = min(self._hist_len + 1, self.history_T)

    def get_token_sequence_observation(self, current_dice: int) -> dict:
        """Return a dict with positions (T,16), dice_history (T,), token_mask (T,16),
        player_history (T,), token_colors (16,), current_dice (1,). Older frames are zero-masked.
        """
        T = self.history_T
        out_pos = np.zeros((T, 16), dtype=np.int64)
        out_dice = np.zeros((T,), dtype=np.int64)
        out_mask = np.zeros((T, 16), dtype=np.bool_)
        out_player = np.zeros((T,), dtype=np.int64)
        k = self._hist_len
        if k > 0:
            # Gather in chronological order
            # Oldest index is (ptr - k) mod T
            start = (self._hist_ptr - k) % T
            if start + k <= T:
                out_pos[T - k : T, :] = self._pos_hist[start : start + k, :]
                out_dice[T - k : T] = self._dice_hist[start : start + k]
                out_mask[T - k : T, :] = self._mask_hist[start : start + k, :]
                out_player[T - k : T] = self._player_hist[start : start + k]
            else:
                first = T - start
                out_pos[T - k : T - k + first, :] = self._pos_hist[start:T, :]
                out_pos[T - k + first : T, :] = self._pos_hist[0 : k - first, :]
                out_dice[T - k : T - k + first] = self._dice_hist[start:T]
                out_dice[T - k + first : T] = self._dice_hist[0 : k - first]
                out_mask[T - k : T - k + first, :] = self._mask_hist[start:T, :]
                out_mask[T - k + first : T, :] = self._mask_hist[0 : k - first, :]
                out_player[T - k : T - k + first] = self._player_hist[start:T]
                out_player[T - k + first : T] = self._player_hist[0 : k - first]
        return {
            "positions": out_pos,
            "dice_history": out_dice,
            "token_mask": out_mask,
            "player_history": out_player,
            "token_colors": self._token_colors,
            "current_dice": np.asarray([int(current_dice)], dtype=np.int64),
        }

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

        # Track when opponent hits agent's blockade
        if (
            result.events.hit_blockade
            and not result.events.move_resolved
            and mover_index != self.agent_index
        ):
            # Opponent failed to move due to blockade - check if it's the agent's blockade
            target_rel = result.new_position  # Position they tried to move to
            mover_color = int(self.game.players[mover_index].color)
            if 1 <= target_rel <= 51:
                abs_pos = self.game.board.absolute_position(mover_color, target_rel)
                agent_rel_pos = self.game.board.relative_position(agent_color, abs_pos)
                if agent_rel_pos != -1:
                    # Check if agent has a blockade at this position
                    agent_pieces_at_pos = self.game.board.count_at_relative(
                        agent_color, agent_rel_pos
                    )
                    if agent_pieces_at_pos >= 2:
                        # Agent's blockade stopped the opponent!
                        self.game.board.blockade_hits[agent_rel_pos] = 1.0

    def step(self, agent_move: Move) -> tuple[bool, bool]:
        """Apply agent move, then simulate opponents unless extra turn.

        Returns (terminated, extra_turn_for_agent)
        """
        # Reset transition summaries at the start of a new turn cycle
        self.game.board.reset_transition_summaries()

        # Apply agent's move
        res = self.game.apply_move(agent_move)
        # Accumulate immediate agent reward from own move
        self._agent_reward_acc += float(res.rewards.get(self.agent_index, 0.0)) if res.rewards else 0.0
        self._update_transition_summaries(self.agent_index, agent_move, res)
        # Log atomic move in history
        self._append_history(agent_move.dice_roll, self.agent_index)
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
                    # Accumulate opponent-driven rewards affecting agent (e.g., their finish)
                    self._agent_reward_acc += float(opp_res.rewards.get(self.agent_index, 0.0)) if opp_res.rewards else 0.0
                    self._append_history(dice, idx)
                idx = (idx + 1) % total_players

        terminated = self.game.players[self.agent_index].check_won()
        return terminated, extra

    def step_opponents_only(self, reset_summaries: bool = True) -> None:
        """Simulate all opponents in turn order until it returns to the agent.

        Handles extra turns for opponents according to game rules.

        Args:
            reset_summaries: If True, reset transition summaries before simulating.
                           Set to False when accumulating multiple opponent rounds
                           between agent turns (e.g., when agent has no valid moves).
        """
        # Reset transition summaries only if requested
        if reset_summaries:
            self.game.board.reset_transition_summaries()
        # Reset accumulation for this opponents phase
        self._agent_reward_acc = 0.0

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
                    except Exception as e:
                        logger.warning(
                            f"Opponent strategy failed for player {idx}, falling back to random: {e}"
                        )
                        decision = None
                mv = decision if decision is not None else random.choice(legal)
                result = self.game.apply_move(mv)
                self._update_transition_summaries(idx, mv, result)
                # Accumulate opponent-driven rewards affecting agent
                self._agent_reward_acc += float(result.rewards.get(self.agent_index, 0.0)) if result.rewards else 0.0
                self._append_history(dice, idx)
                extra = result.extra_turn and result.events.move_resolved

            idx = (idx + 1) % total
