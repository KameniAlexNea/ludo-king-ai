from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from .config import config
from .game import Game
from .types import Move, MoveResult


@dataclass(slots=True)
class Simulator:
    agent_index: int = 0
    game: Game = field(init=False)
    # Token-sequence observation buffers
    history_T: int = config.HISTORY_LENGTH
    _pos_hist: np.ndarray = field(default=None, init=False, repr=False)
    _dice_hist: np.ndarray = field(default=None, init=False, repr=False)
    _mask_hist: np.ndarray = field(default=None, init=False, repr=False)
    _player_hist: np.ndarray = field(default=None, init=False, repr=False)
    _token_colors: np.ndarray = field(default=None, init=False, repr=False)
    _token_exists_mask: np.ndarray = field(default=None, init=False, repr=False)
    _hist_len: int = field(default=0, init=False, repr=False)
    _hist_ptr: int = field(default=0, init=False, repr=False)
    _agent_reward_acc: float = field(default=0.0, init=False, repr=False)
    # Pre-allocated output buffers for get_token_sequence_observation
    _out_pos: np.ndarray = field(default=None, init=False, repr=False)
    _out_dice: np.ndarray = field(default=None, init=False, repr=False)
    _out_mask: np.ndarray = field(default=None, init=False, repr=False)
    _out_player: np.ndarray = field(default=None, init=False, repr=False)
    _out_current_dice: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Expect Game to be constructed by caller with players and strategies.
        # Provide a simple default with random strategies if not injected.
        raise RuntimeError(
            "Simulator requires an explicit Game instance; construct with Simulator.for_game(Game)."
        )

    def get_agent_reward(self) -> float:
        """Get the accumulated agent reward since last reset, without resetting."""
        return self._agent_reward_acc

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
        obj.history_T = config.HISTORY_LENGTH
        agent_color = int(game.players[agent_index].color)
        obj._pos_hist = np.zeros((config.HISTORY_LENGTH, 16), dtype=np.int64)
        obj._dice_hist = np.zeros((config.HISTORY_LENGTH,), dtype=np.int64)
        obj._mask_hist = np.zeros((config.HISTORY_LENGTH, 16), dtype=np.bool_)
        obj._player_hist = np.zeros((config.HISTORY_LENGTH,), dtype=np.int64)
        obj._token_colors = game.board.token_colors(agent_color)
        obj._token_exists_mask = game.board.token_exists_mask(agent_color)
        obj._hist_len = 0
        obj._hist_ptr = 0
        obj._agent_reward_acc = 0.0
        # Pre-allocate output buffers for get_token_sequence_observation
        obj._out_pos = np.zeros((config.HISTORY_LENGTH, 16), dtype=np.int64)
        obj._out_dice = np.zeros((config.HISTORY_LENGTH,), dtype=np.int64)
        obj._out_mask = np.zeros((config.HISTORY_LENGTH, 16), dtype=np.bool_)
        obj._out_player = np.zeros((config.HISTORY_LENGTH,), dtype=np.int64)
        obj._out_current_dice = np.zeros((1,), dtype=np.int64)
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

        Note: Uses pre-allocated buffers for efficiency. The returned arrays are
        views into internal buffers and should be copied if persistence is needed.
        """
        T = self.history_T
        k = self._hist_len

        # Zero out the pre-allocated buffers
        self._out_pos.fill(0)
        self._out_dice.fill(0)
        self._out_mask.fill(False)
        self._out_player.fill(0)

        if k > 0:
            # Gather in chronological order
            # Oldest index is (ptr - k) mod T
            start = (self._hist_ptr - k) % T
            if start + k <= T:
                self._out_pos[T - k : T, :] = self._pos_hist[start : start + k, :]
                self._out_dice[T - k : T] = self._dice_hist[start : start + k]
                self._out_mask[T - k : T, :] = self._mask_hist[start : start + k, :]
                self._out_player[T - k : T] = self._player_hist[start : start + k]
            else:
                first = T - start
                self._out_pos[T - k : T - k + first, :] = self._pos_hist[start:T, :]
                self._out_pos[T - k + first : T, :] = self._pos_hist[0 : k - first, :]
                self._out_dice[T - k : T - k + first] = self._dice_hist[start:T]
                self._out_dice[T - k + first : T] = self._dice_hist[0 : k - first]
                self._out_mask[T - k : T - k + first, :] = self._mask_hist[start:T, :]
                self._out_mask[T - k + first : T, :] = self._mask_hist[0 : k - first, :]
                self._out_player[T - k : T - k + first] = self._player_hist[start:T]
                self._out_player[T - k + first : T] = self._player_hist[0 : k - first]

        self._out_current_dice[0] = int(current_dice)

        return {
            "positions": self._out_pos,
            "dice_history": self._out_dice,
            "token_mask": self._out_mask,
            "player_history": self._out_player,
            "token_colors": self._token_colors,
            "current_dice": self._out_current_dice,
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

    def _get_opponent_move(self, player_idx: int, legal_moves: list[Move]) -> Move:
        """Get move from opponent strategy with fallback to random.

        Args:
            player_idx: Index of the opponent player.
            legal_moves: List of legal moves available.

        Returns:
            The chosen move from strategy or random fallback.
        """
        player = self.game.players[player_idx]
        player_color = int(player.color)
        board_stack = self.game.board.build_tensor(player_color)
        dice = legal_moves[0].dice_roll  # All moves have same dice

        decision = None
        if hasattr(player, "choose"):
            try:
                decision = player.choose(board_stack, dice, legal_moves)
            except Exception as e:
                logger.warning(
                    f"Opponent strategy failed for player {player_idx}, "
                    f"falling back to random: {e}"
                )
                decision = None

        return decision if decision is not None else random.choice(legal_moves)

    def _process_move_result(
        self, player_idx: int, move: Move, result: MoveResult
    ) -> None:
        """Process a move result: update summaries, accumulate rewards, log history.

        Args:
            player_idx: Index of the player who made the move.
            move: The move that was applied.
            result: The result from applying the move.
        """
        self._update_transition_summaries(player_idx, move, result)
        # Accumulate rewards affecting agent
        self._agent_reward_acc += (
            float(result.rewards.get(self.agent_index, 0.0)) if result.rewards else 0.0
        )
        self._append_history(move.dice_roll, player_idx)

    def _simulate_single_opponent(self, player_idx: int) -> None:
        """Simulate all turns for a single opponent (handles extra turns).

        Args:
            player_idx: Index of the opponent to simulate.
        """
        player = self.game.players[player_idx]
        if player.check_won():
            return

        extra = True
        extra_count = 0

        while extra and extra_count < config.MAX_EXTRA_TURNS:
            extra_count += 1
            dice = self.game.roll_dice()
            legal = self.game.legal_moves(player_idx, dice)

            if not legal:
                break

            mv = self._get_opponent_move(player_idx, legal)
            result = self.game.apply_move(mv)
            self._process_move_result(player_idx, mv, result)
            extra = result.extra_turn and result.events.move_resolved

    def _simulate_all_opponents(self) -> None:
        """Simulate all opponents in turn order until it returns to the agent."""
        total = len(self.game.players)
        idx = (self.agent_index + 1) % total

        while idx != self.agent_index:
            self._simulate_single_opponent(idx)
            idx = (idx + 1) % total

    def step(self, agent_move: Move) -> tuple[bool, bool]:
        """Apply agent move, then simulate opponents unless extra turn.

        Returns (terminated, extra_turn_for_agent)
        """
        # Reset transition summaries at the start of a new turn cycle
        self.game.board.reset_transition_summaries()

        # Apply agent's move
        res = self.game.apply_move(agent_move)
        self._process_move_result(self.agent_index, agent_move, res)
        extra = res.extra_turn

        # Simulate others if no extra turn
        if not extra:
            self._simulate_all_opponents()

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
        if reset_summaries:
            self.game.board.reset_transition_summaries()
            self._agent_reward_acc = 0.0

        self._simulate_all_opponents()
