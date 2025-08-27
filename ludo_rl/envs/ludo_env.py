"""Gymnasium environment wrapping core Ludo engine.

Design goals:
- Thin adapter over existing game logic (`ludo.game.LudoGame`, strategies)
- Clean observation builder with extensible features
- Configurable reward shaping (modular components)
- Deterministic seeding & reproducibility
- Support vectorization (Stable-Baselines3 / RLlib)
- Compatible with Gymnasium API (reset(seed=...), step returning (obs, reward, terminated, truncated, info))

Observation vector (default v0 schema):
    [ agent_token_positions(4), agent_token_states(4 one-hot aggregated -> active/home/home_column/finished counts),
      opponents_token_positions(3*4), finished_tokens_per_player(4), dice_value/6, can_any_finish, progress_fraction_agent,
      opponent_mean_progress, turn_relative_position_indicator ]

All numeric continuous features are scaled into [-1, 1] or [0,1] then shifted.

Action space: Discrete(4) choose token index. Invalid selections resolved to NOOP with penalty.

Future extensions: maskable action space, multi-agent self-play environment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ludo.constants import Colors, GameConstants
from ludo.game import LudoGame
from ludo.player import Player, PlayerColor
from ludo.strategy import StrategyFactory

# --------------------------------------------------------------------------------------
# Config dataclasses
# --------------------------------------------------------------------------------------


@dataclass
class RewardConfig:
        """Reward shaping configuration (scaled for stable RL training).

        Magnitudes chosen to avoid sparse domination while preserving signal:
            - Win / lose kept an order of magnitude above per-move signals (10 / -10)
            - Progress shaping small dense signal encourages forward motion
            - Capture / finish moderate bonuses; capture symmetrical with loss
            - Illegal action mildly discouraged (mask should usually prevent it)
        """

        # Primary events
        capture: float = 2.0
        got_captured: float = -2.5
        finish_token: float = 5.0
        win: float = 10.0
        lose: float = -10.0

        # Dense shaping (per total normalized progress delta across 4 tokens)
        progress_scale: float = 0.5

        # Miscellaneous
        time_penalty: float = -0.001
        illegal_action: float = -0.05
        extra_turn: float = 0.3
        blocking_bonus: float = 0.15
        diversity_bonus: float = 0.2  # first time a token leaves home


@dataclass
class ObservationConfig:
    include_blocking_count: bool = True
    include_turn_index: bool = True
    include_raw_dice: bool = True
    normalize_positions: bool = True


@dataclass
class OpponentsConfig:
    strategies: List[str] = field(
        default_factory=lambda: ["balanced", "probabilistic_v3", "optimist"]
    )  # length 3


@dataclass
class EnvConfig:
    agent_color: str = Colors.RED
    max_turns: int = 1000
    reward_cfg: RewardConfig = field(default_factory=RewardConfig)
    obs_cfg: ObservationConfig = field(default_factory=ObservationConfig)
    opponents: OpponentsConfig = field(default_factory=OpponentsConfig)
    seed: Optional[int] = None


# --------------------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------------------


class LudoGymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "name": "LudoGymEnv-v0"}

    def __init__(
        self, config: Optional[EnvConfig] = None
    ):  # gym style accepts **kwargs
        super().__init__()
        self.cfg = config or EnvConfig()
        self.rng = random.Random(self.cfg.seed)

        # Build core game with fixed 4 players in canonical order (R,G,Y,B) if present
        order = [
            PlayerColor.RED,
            PlayerColor.GREEN,
            PlayerColor.YELLOW,
            PlayerColor.BLUE,
        ]
        self.game = LudoGame(order)

        # Attach opponent strategies
        self.agent_color = self.cfg.agent_color
        for p in self.game.players:
            if p.color.value != self.agent_color:
                idx = [c for c in Colors.ALL_COLORS if c != self.agent_color].index(
                    p.color.value
                )
                strat_name = self.cfg.opponents.strategies[
                    idx % len(self.cfg.opponents.strategies)
                ]
                try:
                    p.set_strategy(StrategyFactory.create_strategy(strat_name))
                except Exception:
                    pass

        # Spaces
        obs_dim = self._compute_observation_size()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(GameConstants.TOKENS_PER_PLAYER)

        # Episode / bookkeeping
        self.turns = 0  # count of agent decision turns (not full game cycles)
        self.episode_steps = 0
        self.done = False
        self.last_obs: Optional[np.ndarray] = None
        self._token_activation_flags = {
            i: False for i in range(GameConstants.TOKENS_PER_PLAYER)
        }

        # Internal per-turn state
        self._pending_agent_dice: Optional[int] = None
        self._pending_valid_moves: List[Dict] = []
        self._last_progress_sum: float = 0.0

    # ----------------------------------------------------------------------------------
    # Gym API
    # ----------------------------------------------------------------------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):  # type: ignore[override]
        if seed is not None:
            self.cfg.seed = seed
        if self.cfg.seed is not None:
            self.rng.seed(self.cfg.seed)
            # Seed global random so core game (which uses random module) is reproducible
            random.seed(self.cfg.seed)
        # Recreate game for clean state
        order = [
            PlayerColor.RED,
            PlayerColor.GREEN,
            PlayerColor.YELLOW,
            PlayerColor.BLUE,
        ]
        self.game = LudoGame(order)
        # Reattach strategies
        for p in self.game.players:
            if p.color.value != self.agent_color:
                idx = [c for c in Colors.ALL_COLORS if c != self.agent_color].index(
                    p.color.value
                )
                strat_name = self.cfg.opponents.strategies[
                    idx % len(self.cfg.opponents.strategies)
                ]
                try:
                    p.set_strategy(StrategyFactory.create_strategy(strat_name))
                except Exception:
                    pass
        self.turns = 0
        self.episode_steps = 0
        self.done = False
        self._token_activation_flags = {
            i: False for i in range(GameConstants.TOKENS_PER_PLAYER)
        }
        self._pending_agent_dice = None
        self._pending_valid_moves = []
        self._last_progress_sum = self._compute_agent_progress_sum()

        # Advance until it's agent's turn (initial current_player_index is 0 so usually unnecessary)
        self._ensure_agent_turn()
        # Roll initial dice for agent decision
        self._roll_new_agent_dice()
        obs = self._build_observation()
        self.last_obs = obs
        info = {}
        return obs, info

    def step(self, action: int):  # type: ignore[override]
        """Advance environment by one agent decision (one dice roll).

        Correct turn mechanics:
          - One agent decision per step
          - Extra turns generate a new decision (another step) without opponents moving
          - Opponents play complete sequence (including their extra turns) between agent decisions
        """
        if self.done:
            return self.last_obs, 0.0, True, False, {}

        rcfg = self.cfg.reward_cfg
        reward_components: List[float] = []
        agent_player = next(p for p in self.game.players if p.color.value == self.agent_color)

        # Ensure we have a pending dice & valid moves (should always be true except pathological cases)
        if self._pending_agent_dice is None:
            self._roll_new_agent_dice()

        dice_value = self._pending_agent_dice
        valid_moves = self._pending_valid_moves

        # Compute progress baseline
        progress_before = self._last_progress_sum

        # Handle no-move situation
        no_moves_available = len(valid_moves) == 0
        chosen_move: Optional[Dict] = None
        illegal = False
        if not no_moves_available:
            valid_token_ids = [m["token_id"] for m in valid_moves]
            if action not in valid_token_ids:
                illegal = True
                reward_components.append(rcfg.illegal_action)
                # choose fallback (first valid) for execution so environment state advances
                exec_token_id = valid_token_ids[0]
            else:
                exec_token_id = action
            chosen_move = next(m for m in valid_moves if m["token_id"] == exec_token_id)
            move_res = self.game.execute_move(agent_player, exec_token_id, dice_value)

            if move_res.get("captured_tokens"):
                reward_components.append(rcfg.capture * len(move_res["captured_tokens"]))
            if move_res.get("token_finished"):
                reward_components.append(rcfg.finish_token)
            if move_res.get("extra_turn"):
                reward_components.append(rcfg.extra_turn)
            # Diversity bonus
            tok = agent_player.tokens[exec_token_id]
            if tok.position >= 0 and not self._token_activation_flags[exec_token_id]:
                reward_components.append(rcfg.diversity_bonus)
                self._token_activation_flags[exec_token_id] = True
            extra_turn = move_res.get("extra_turn", False)
        else:
            # No valid moves: treat as skipped turn (no illegal penalty)
            extra_turn = False

        # Opponent simulation if no extra turn and game not over
        if not extra_turn and not self.game.game_over:
            # Advance turn for agent (dice consumed)
            self.game.next_turn()
            self._simulate_opponents(reward_components)

        # Progress shaping (after agent + opponents if any)
        progress_after = self._compute_agent_progress_sum()
        delta = progress_after - progress_before
        if abs(delta) > 1e-9:
            reward_components.append(delta * rcfg.progress_scale)
        self._last_progress_sum = progress_after

        # Time penalty (light)
        reward_components.append(rcfg.time_penalty)

        # Terminal checks
        terminated = False
        truncated = False
        if agent_player.has_won():
            reward_components.append(rcfg.win)
            terminated = True
        elif any(p.has_won() for p in self.game.players if p.color.value != self.agent_color):
            reward_components.append(rcfg.lose)
            terminated = True

        self.turns += 1
        self.episode_steps += 1
        if self.turns >= self.cfg.max_turns and not terminated:
            truncated = True

        # Prepare next agent dice if continuing (extra turn) and not done
        if not terminated and not truncated and not self.game.game_over:
            if extra_turn:
                self._roll_new_agent_dice()  # agent retains turn
            else:
                self._ensure_agent_turn()  # move through opponents done above
                if not self.game.game_over:
                    self._roll_new_agent_dice()

        obs = self._build_observation()
        self.last_obs = obs
        self.done = terminated or truncated
        info = {
            "reward_components": reward_components,
            "dice": self._pending_agent_dice,
            "illegal_action": illegal,
            "action_mask": self.action_masks(),
            "had_extra_turn": extra_turn,
        }
        total_reward = float(sum(reward_components))
        return obs, total_reward, terminated, truncated, info

    # ----------------------------------------------------------------------------------
    # Observation
    # ----------------------------------------------------------------------------------
    def _compute_observation_size(self) -> int:
        base = 0
        # agent token positions (4)
        base += 4
        # opponents token positions (12)
        base += 12
        # finished tokens per player (4)
        base += 4
        # flags: can_finish, dice_value, progress stats, turn index
        base += 4  # can_finish, dice_norm, agent_progress, opp_mean_progress
        if self.cfg.obs_cfg.include_turn_index:
            base += 1
        if self.cfg.obs_cfg.include_blocking_count:
            base += 1
        return base

    def _normalize_position(self, pos: int) -> float:
        if pos == -1:
            return -1.0
        if pos >= 100:
            depth = (pos - 100) / 5.0  # 0..1
            return 0.5 + depth * 0.5
        return (pos / (GameConstants.MAIN_BOARD_SIZE - 1)) * 0.5  # [0,0.5]

    def _build_observation(self) -> np.ndarray:
        agent_player = next(p for p in self.game.players if p.color.value == self.agent_color)
        vec: List[float] = []
        # agent tokens
        for t in agent_player.tokens:
            vec.append(self._normalize_position(t.position))
        # opponents tokens in fixed global order excluding agent
        for color in Colors.ALL_COLORS:
            if color == self.agent_color:
                continue
            opp = next(p for p in self.game.players if p.color.value == color)
            for t in opp.tokens:
                vec.append(self._normalize_position(t.position))
        # finished counts
        for color in Colors.ALL_COLORS:
            pl = next(p for p in self.game.players if p.color.value == color)
            vec.append(pl.get_finished_tokens_count() / GameConstants.TOKENS_PER_PLAYER)
        # can any finish
        can_finish = 0.0
        for t in agent_player.tokens:
            if 0 <= t.position < 100:
                remaining = 105 - t.position
                if remaining <= 6:
                    can_finish = 1.0
                    break
        vec.append(can_finish)
        # dice norm (pending dice for current decision)
        if self._pending_agent_dice is None:
            vec.append(0.0)
        else:
            vec.append((self._pending_agent_dice - 3.5) / 3.5)
        # progress stats
        agent_progress = (
            agent_player.get_finished_tokens_count() / GameConstants.TOKENS_PER_PLAYER
        )
        opp_progresses = []
        for color in Colors.ALL_COLORS:
            if color == self.agent_color:
                continue
            pl = next(p for p in self.game.players if p.color.value == color)
            opp_progresses.append(
                pl.get_finished_tokens_count() / GameConstants.TOKENS_PER_PLAYER
            )
        opp_mean = (
            sum(opp_progresses) / max(1, len(opp_progresses)) if opp_progresses else 0.0
        )
        vec.append(agent_progress)
        vec.append(opp_mean)
        # turn index scaled
        if self.cfg.obs_cfg.include_turn_index:
            vec.append(min(1.0, self.turns / self.cfg.max_turns))
        # blocking count
        if self.cfg.obs_cfg.include_blocking_count:
            blocking_positions = self.game.board.get_blocking_positions(self.agent_color)
            vec.append(min(1.0, len(blocking_positions) / 6.0))  # normalize roughly
        return np.asarray(vec, dtype=np.float32)

    # ----------------------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------------------
    def _roll_dice(self) -> int:
        # Use core game mechanics (seeding done via global random seed)
        return self.game.roll_dice()

    def _roll_new_agent_dice(self):
        self._pending_agent_dice = self._roll_dice()
        agent_player = next(p for p in self.game.players if p.color.value == self.agent_color)
        self._pending_valid_moves = self.game.get_valid_moves(agent_player, self._pending_agent_dice)

    def _ensure_agent_turn(self):
        # Simulate opponents until agent color is current player or game over
        while not self.game.game_over and self.game.get_current_player().color.value != self.agent_color:
            self._simulate_single_opponent_turn()

    def _simulate_single_opponent_turn(self):
        current_player = self.game.get_current_player()
        if current_player.color.value == self.agent_color:
            return
        dice_value = self._roll_dice()
        valid_moves = self.game.get_valid_moves(current_player, dice_value)
        if not valid_moves:
            # no moves -> turn ends
            self.game.next_turn()
            return
        # Strategy decision
        try:
            ctx = self._make_strategy_context(current_player, dice_value, valid_moves)
            token_choice = current_player.make_strategic_decision(ctx)
        except Exception:
            token_choice = valid_moves[0]["token_id"]
        move_res = self.game.execute_move(current_player, token_choice, dice_value)
        # Handle extra turn chain for opponent
        if move_res.get("extra_turn") and not self.game.game_over:
            # Recursively continue same player
            self._simulate_single_opponent_turn()
        else:
            if not self.game.game_over:
                self.game.next_turn()

    def _simulate_opponents(self, reward_components: List[float]):
        # Simulate until it becomes agent's turn
        while not self.game.game_over and self.game.get_current_player().color.value != self.agent_color:
            cur_player = self.game.get_current_player()
            pre_agent_tokens = self._snapshot_agent_tokens()
            self._simulate_single_opponent_turn()
            # Check captures of agent tokens during this opponent sequence
            post_agent_tokens = self._snapshot_agent_tokens()
            # If any agent token returned home (position -1) while previously on board -> captured
            for before, after in zip(pre_agent_tokens, post_agent_tokens):
                if before >= 0 and after == -1:
                    reward_components.append(self.cfg.reward_cfg.got_captured)

    def _snapshot_agent_tokens(self) -> List[int]:
        player = next(p for p in self.game.players if p.color.value == self.agent_color)
        return [t.position for t in player.tokens]

    def _compute_agent_progress_sum(self) -> float:
        player = next(p for p in self.game.players if p.color.value == self.agent_color)
        total = 0.0
        for t in player.tokens:
            if t.position == -1:
                continue
            if 0 <= t.position < GameConstants.MAIN_BOARD_SIZE:
                total += t.position / float(GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE)
            elif t.position >= 100:
                # Home column progress: 52..58 mapped
                offset = (t.position - 100) + GameConstants.MAIN_BOARD_SIZE
                total += offset / float(GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE)
        return total

    # Action masking (for algorithms that support it)
    def action_masks(self) -> np.ndarray:
        mask = np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=np.int8)
        if self._pending_valid_moves:
            valid_ids = {m["token_id"] for m in self._pending_valid_moves}
            for i in range(GameConstants.TOKENS_PER_PLAYER):
                if i in valid_ids:
                    mask[i] = 1
        return mask

    def _make_strategy_context(
        self, player: Player, dice_value: int, valid_moves: List[Dict]
    ):
        # Basic context bridging existing strategies expecting a structure similar to tournaments
        board_state = self.game.board.get_board_state_for_ai(player)
        opponents = []
        for p in self.game.players:
            if p is player:
                continue
            opponents.append(p.get_game_state())
        ctx = {
            "player_state": player.get_game_state(),
            "board": board_state,
            "valid_moves": valid_moves,
            "dice_value": dice_value,
            "opponents": opponents,
        }
        return ctx

    def render(self):  # minimal
        print(f"Turn {self.turns} agent_color={self.agent_color}")

    def close(self):
        pass


__all__ = [
    "LudoGymEnv",
    "EnvConfig",
    "RewardConfig",
    "ObservationConfig",
    "OpponentsConfig",
]
