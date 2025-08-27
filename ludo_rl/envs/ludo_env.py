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
from typing import Any, Dict, List, Optional

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
    capture: float = 8.0
    got_captured: float = -6.0
    finish_token: float = 12.0
    win: float = 120.0
    lose: float = -60.0
    progress_scale: float = 1.5  # multiplied by normalized progress delta sum
    time_penalty: float = -0.01
    illegal_action: float = -0.2
    extra_turn: float = 0.5
    blocking_bonus: float = 0.3  # reward for creating/maintaining blocks
    diversity_bonus: float = (
        0.1  # reward per newly activated token (first time leaving home)
    )


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
        self.turns = 0
        self.episode_steps = 0
        self.done = False
        self.last_obs: Optional[np.ndarray] = None
        self._token_activation_flags = {
            i: False for i in range(GameConstants.TOKENS_PER_PLAYER)
        }

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
        obs = self._build_observation(last_dice=None)
        self.last_obs = obs
        info = {}
        return obs, info

    def step(self, action: int):  # type: ignore[override]
        if self.done:
            return self.last_obs, 0.0, True, False, {}

        reward_components = []
        rcfg = self.cfg.reward_cfg

        # 1. Agent turn sequence (may include extra turns)
        agent_player = next(
            p for p in self.game.players if p.color.value == self.agent_color
        )
        total_reward = 0.0
        terminated = False
        truncated = False
        dice_value = None

        extra_turn_chain = True
        local_turns = 0
        while extra_turn_chain and not self.game.game_over:
            local_turns += 1
            dice_value = self._roll_dice()
            valid_moves = self.game.get_valid_moves(agent_player, dice_value)
            token_choice = action
            illegal = token_choice not in [m["token_id"] for m in valid_moves]
            if illegal:
                reward_components.append(rcfg.illegal_action)
                # fallback: choose a move (simple heuristic)
                if valid_moves:
                    token_choice = valid_moves[0]["token_id"]
                else:
                    token_choice = 0
            if valid_moves:
                move_res = self.game.execute_move(
                    agent_player, token_choice, dice_value
                )
                # progress delta measure
                # total_progress = sum(
                #     t.position
                #     for t in agent_player.tokens
                #     if t.position >= 0 and t.position < 100
                # )
                # Reward pieces
                if move_res.get("captured_tokens"):
                    reward_components.append(
                        rcfg.capture * len(move_res["captured_tokens"])
                    )
                if move_res.get("token_finished"):
                    reward_components.append(rcfg.finish_token)
                if move_res.get("extra_turn"):
                    reward_components.append(rcfg.extra_turn)
                # Token diversity activation
                tok = agent_player.tokens[token_choice]
                if tok.position >= 0 and not self._token_activation_flags[token_choice]:
                    reward_components.append(rcfg.diversity_bonus)
                    self._token_activation_flags[token_choice] = True
                # simple progress shaping (difference vs baseline?)
                # (Could track previous positions; omitted for brevity)
            else:
                extra_turn_chain = False
                break
            extra_turn_chain = move_res.get("extra_turn", False)
            # If extra turn we use same external action again (agent chooses one token index) - simple baseline

            if local_turns > 10:  # fail-safe
                break

        # 2. Opponents each get one turn sequentially until it returns to agent (no recursion of extra turns for simplicity)
        if not self.game.game_over:
            for p in self.game.players:
                if p.color.value == self.agent_color:
                    continue
                dice_val = self._roll_dice()
                val_moves = self.game.get_valid_moves(p, dice_val)
                if val_moves:
                    # ask strategy
                    ctx = self._make_strategy_context(p, dice_val, val_moves)
                    try:
                        token_choice = p.make_strategic_decision(ctx)
                    except Exception:
                        token_choice = val_moves[0]["token_id"]
                    mv_res = self.game.execute_move(p, token_choice, dice_val)
                    # negative reward if agent tokens captured
                    if mv_res.get("captured_tokens"):
                        for ct in mv_res["captured_tokens"]:
                            if ct["player_color"] == self.agent_color:
                                reward_components.append(rcfg.got_captured)

        # 3. Time penalty
        reward_components.append(rcfg.time_penalty)

        # 4. Terminal conditions
        if agent_player.has_won():
            reward_components.append(rcfg.win)
            terminated = True
        elif any(
            p.has_won() for p in self.game.players if p.color.value != self.agent_color
        ):
            reward_components.append(rcfg.lose)
            terminated = True

        self.turns += 1
        self.episode_steps += 1
        if self.turns >= self.cfg.max_turns and not terminated:
            truncated = True

        total_reward = float(sum(reward_components))
        obs = self._build_observation(last_dice=dice_value)
        self.last_obs = obs
        self.done = terminated or truncated
        info = {"reward_components": reward_components, "dice": dice_value}
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

    def _build_observation(self, last_dice: Optional[int]) -> np.ndarray:
        agent_player = next(
            p for p in self.game.players if p.color.value == self.agent_color
        )
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
        # dice norm
        if last_dice is None:
            vec.append(0.0)
        else:
            vec.append((last_dice - 3.5) / 3.5)  # ~[-1,1]
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
            # simple placeholder (could query board blocking positions)
            vec.append(0.0)
        return np.asarray(vec, dtype=np.float32)

    # ----------------------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------------------
    def _roll_dice(self) -> int:
        return self.rng.randint(GameConstants.DICE_MIN, GameConstants.DICE_MAX)

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
