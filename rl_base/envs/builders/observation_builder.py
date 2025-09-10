"""Observation building utilities for LudoGymEnv."""

from typing import List

import numpy as np

from ludo.constants import BoardConstants, Colors, GameConstants
from ludo.game import LudoGame

from ..model import BaseEnvConfig


class ObservationBuilder:
    """Handles observation construction for the Ludo environment."""

    def __init__(self, cfg: BaseEnvConfig, game: LudoGame, agent_color: str):
        self.cfg = cfg
        self.game = game
        self.agent_color = agent_color
        self.start_position = BoardConstants.START_POSITIONS.get(agent_color)
        self.obs_size = self._compute_observation_size()

    def _compute_observation_size(self) -> int:
        base = 0
        # agent token positions (4)
        base += 4
        # opponents token positions (12)
        base += 12
        # finished tokens per player (4)
        base += 4
        # current player color one-hot (4)
        base += 4
        # scalar flags / stats:
        # can_finish, agent_finished_fraction, opp_mean_finished_fraction, agent_mean_token_progress
        base += 4
        # dice one-hot
        base += 6
        # distance to finish for each token (4)
        base += 4
        # tokens at home count (1)
        base += 1
        if self.cfg.obs_cfg.include_turn_index:
            base += 1
        if self.cfg.obs_cfg.include_blocking_count:
            base += 1
        return base

    def _remove_agent_start(self, pos: int) -> int:
        """
        Normalize position relative to agent's start position.
        Makes the agent's start position always appear as 0.

        Args:
            pos: Original position on the board

        Returns:
            Position relative to agent's start (agent start becomes 0)
        """
        if (
            pos == GameConstants.HOME_POSITION
            or pos >= BoardConstants.HOME_COLUMN_START
        ):
            # Keep home (-1) and home-column (>=100) positions unchanged
            return pos
        return (pos - self.start_position + 1) % GameConstants.MAIN_BOARD_SIZE

    def _normalize_position(self, pos: int) -> float:
        """Map a raw board position to [-1,1] strictly monotonically.

        Ordering enforced:
          -1 (HOME) < 0..<51 (main board) < 100..105 (home column to finish)
        We compress the numeric gap 52..99 (unused) so progression is uniform
        across the actually reachable sequence of states.

        Monotonic guarantee: if x < y in the above logical ordering, then
        _normalize_position(x) < _normalize_position(y).
        """
        pos = self._remove_agent_start(pos)
        # Special home (not yet entered play)
        if pos == GameConstants.HOME_POSITION:
            return -1.0

        # Build a compact rank skipping unreachable gap 52..99
        # Ranks: 0 -> home (-1), 1..52 -> board 0..51, 53..58 -> home column 100..105
        if 0 <= pos < BoardConstants.HOME_COLUMN_START:
            rank = pos + 1  # 1..52
        elif pos >= BoardConstants.HOME_COLUMN_START:
            rank = (GameConstants.MAIN_BOARD_SIZE + 1) + (
                pos - BoardConstants.HOME_COLUMN_START
            )  # 53..(53+5)
        else:
            # Any unexpected negative (other than HOME) fallback just treat as home
            return -1.0

        total_ranks = (
            1 + GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE
        )  # 59
        # Normalize rank to [-1,1]
        return -1.0 + 2.0 * (rank / (total_ranks - 1))

    def _build_observation(
        self, turns: int, pending_agent_dice: int = None
    ) -> np.ndarray:
        # Cache map for faster lookups
        players_by_color = {p.color.value: p for p in self.game.players}
        agent_player = players_by_color[self.agent_color]

        vec: List[float] = []
        # agent tokens
        for t in agent_player.tokens:
            vec.append(self._normalize_position(t.position))
        # opponents tokens in fixed global order excluding agent
        for color in Colors.ALL_COLORS:
            if color == self.agent_color:
                continue
            opp = players_by_color[color]
            for t in opp.tokens:
                vec.append(self._normalize_position(t.position))
        # finished counts
        for color in Colors.ALL_COLORS:
            pl = players_by_color[color]
            vec.append(pl.get_finished_tokens_count() / GameConstants.TOKENS_PER_PLAYER)
        # player color one-hot in fixed order R,G,Y,B
        for color in Colors.ALL_COLORS:
            vec.append(1.0 if color == self.agent_color else 0.0)

        # can any finish (single arithmetic check per token)
        def remaining_steps_to_finish(pos: int) -> int:
            if pos == GameConstants.HOME_POSITION:
                return 1_000_000  # effectively infinite (needs a 6 to exit first)
            if pos == GameConstants.FINISH_POSITION:
                return 1_000_000
            if pos >= BoardConstants.HOME_COLUMN_START:
                return GameConstants.FINISH_POSITION - pos  # direct steps inside column
            # main board token
            home_entry = BoardConstants.HOME_COLUMN_ENTRIES[self.agent_color]
            if pos <= home_entry:
                to_entry = home_entry - pos + 1  # +1 step to move past entry into 100
            else:
                # wrap around board then enter
                to_entry = (GameConstants.MAIN_BOARD_SIZE - pos) + home_entry + 1
            # 5 steps from 100..105
            return to_entry + (
                GameConstants.FINISH_POSITION - BoardConstants.HOME_COLUMN_START
            )

        can_finish = 0.0
        for t in agent_player.tokens:
            steps = remaining_steps_to_finish(t.position)
            if 1 <= steps <= GameConstants.DICE_MAX:
                can_finish = 1.0
                break
        vec.append(can_finish)
        # dice one-hot (6 values for 1-6)
        if pending_agent_dice is None:
            vec.extend([0.0] * 6)
        else:
            one_hot = [0.0] * 6
            if 1 <= pending_agent_dice <= 6:
                one_hot[pending_agent_dice - 1] = 1.0
            vec.extend(one_hot)
        # distance to finish for each token
        for t in agent_player.tokens:
            pos = t.position
            if pos == GameConstants.FINISH_POSITION:
                dist = 0.0
            elif pos == GameConstants.HOME_POSITION:
                dist = 1.0  # still at home
            else:
                total_path = (
                    GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE
                )
                # compute steps progressed along this color's path (mirrors logic below for mean_token_progress)
                if pos >= BoardConstants.HOME_COLUMN_START:
                    steps_done = GameConstants.MAIN_BOARD_SIZE + (
                        pos - BoardConstants.HOME_COLUMN_START
                    )
                else:
                    start_pos = BoardConstants.START_POSITIONS.get(self.agent_color)
                    if pos >= start_pos:
                        steps_done = pos - start_pos
                    else:
                        steps_done = (GameConstants.MAIN_BOARD_SIZE - start_pos) + pos
                remaining_steps = max(0, total_path - steps_done)
                dist = min(1.0, remaining_steps / total_path)
            vec.append(dist)
        # tokens at home count
        tokens_at_home = (
            sum(1 for t in agent_player.tokens if t.position < 0)
            / GameConstants.TOKENS_PER_PLAYER
        )
        vec.append(tokens_at_home)
        # progress stats
        agent_progress = (
            agent_player.get_finished_tokens_count() / GameConstants.TOKENS_PER_PLAYER
        )
        opp_progresses = []
        for color in Colors.ALL_COLORS:
            if color == self.agent_color:
                continue
            pl = players_by_color[color]
            opp_progresses.append(
                pl.get_finished_tokens_count() / GameConstants.TOKENS_PER_PLAYER
            )
        opp_mean = (
            sum(opp_progresses) / max(1, len(opp_progresses)) if opp_progresses else 0.0
        )
        vec.append(agent_progress)
        vec.append(opp_mean)
        # agent mean token progress (path coverage)
        total_path = GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE
        start_pos = BoardConstants.START_POSITIONS.get(self.agent_color)

        def steps_done_for(pos: int) -> int:
            if pos == GameConstants.HOME_POSITION:
                return 0
            if pos == GameConstants.FINISH_POSITION:
                return total_path  # full progress
            if pos >= BoardConstants.HOME_COLUMN_START:
                return GameConstants.MAIN_BOARD_SIZE + (
                    pos - BoardConstants.HOME_COLUMN_START
                )
            # main board portion relative to this color's start
            if pos >= start_pos:
                return pos - start_pos
            return (GameConstants.MAIN_BOARD_SIZE - start_pos) + pos

        token_progress_vals = []
        for t in agent_player.tokens:
            sd = steps_done_for(t.position)
            token_progress_vals.append(min(1.0, sd / total_path))
        mean_token_progress = sum(token_progress_vals) / max(
            1, len(token_progress_vals)
        )
        vec.append(mean_token_progress)
        # turn index scaled
        if self.cfg.obs_cfg.include_turn_index:
            vec.append(
                min(GameConstants.TURN_INDEX_MAX_SCALE, turns / self.cfg.max_turns)
            )
        # blocking count
        if self.cfg.obs_cfg.include_blocking_count:
            blocking_positions = self.game.board.get_blocking_positions(
                self.agent_color
            )
            vec.append(
                min(
                    GameConstants.TURN_INDEX_MAX_SCALE,
                    len(blocking_positions)
                    / GameConstants.BLOCKING_COUNT_NORMALIZATION,
                )
            )  # normalize roughly
        obs = np.asarray(vec, dtype=np.float32)
        if obs.shape[0] != self.obs_size:
            raise ValueError(
                f"Observation length mismatch: got {obs.shape[0]} expected {self.obs_size}"
            )
        return obs
