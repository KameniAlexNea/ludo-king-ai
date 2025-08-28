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

import math
import random
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ludo.constants import BoardConstants, Colors, GameConstants
from ludo.game import LudoGame
from ludo.player import Player, PlayerColor
from ludo.strategy import StrategyFactory

from .builders.observation_builder import ObservationBuilder
from .calculators.reward_calculator import RewardCalculator
from .model import EnvConfig
from .simulators.opponent_simulator import OpponentSimulator
from .utils.move_utils import MoveUtils

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

        # Sample opponent strategies for this environment instance
        self.opponent_strategies = self.rng.sample(self.cfg.opponents.candidates, 3)

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
                strat_name = self.opponent_strategies[
                    idx % len(self.opponent_strategies)
                ]
                try:
                    p.set_strategy(StrategyFactory.create_strategy(strat_name))
                except Exception:
                    pass

        # Spaces
        obs_dim = self.obs_builder._compute_observation_size()
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

        self.obs_builder = ObservationBuilder(self.cfg, self.game, self.agent_color)
        self.reward_calc = RewardCalculator(self.cfg, self.game, self.agent_color)
        self.opp_simulator = OpponentSimulator(
            self.cfg, self.game, self.agent_color, self.move_utils._roll_dice, self.move_utils._make_strategy_context
        )
        self.move_utils = MoveUtils(self.cfg, self.game, self.agent_color)

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
        # Sample new opponent strategies for this episode
        self.opponent_strategies = self.rng.sample(self.cfg.opponents.candidates, 3)
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
                strat_name = self.opponent_strategies[
                    idx % len(self.opponent_strategies)
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
        self._last_progress_sum = self.move_utils._compute_agent_progress_sum()

        # Advance until it's agent's turn (initial current_player_index is 0 so usually unnecessary)
        self.opp_simulator._ensure_agent_turn()
        # Roll initial dice for agent decision
        self._pending_agent_dice, self._pending_valid_moves = self.move_utils._roll_new_agent_dice()
        obs = self.obs_builder._build_observation(self.turns, self._pending_agent_dice)
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
        agent_player = next(
            p for p in self.game.players if p.color.value == self.agent_color
        )

        # Ensure we have a pending dice & valid moves (should always be true except pathological cases)
        if self._pending_agent_dice is None:
            self._pending_agent_dice, self._pending_valid_moves = self.move_utils._roll_new_agent_dice()

        dice_value = self._pending_agent_dice
        valid_moves = self._pending_valid_moves

        # Compute progress baseline
        progress_before = self.move_utils._compute_agent_progress_sum()

        # Handle no-move situation
        no_moves_available = len(valid_moves) == 0
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
            move_res = self.game.execute_move(agent_player, exec_token_id, dice_value)

            if move_res.get("captured_tokens"):
                capture_reward = self.reward_calc.compute_capture_reward(move_res, reward_components)
                reward_components.append(capture_reward)
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
            self.opp_simulator._simulate_opponents(reward_components)

        # Progress shaping (after agent + opponents if any)
        progress_after = self.move_utils._compute_agent_progress_sum()
        progress_reward = self.reward_calc.compute_progress_reward(progress_before, progress_after)
        if progress_reward != 0.0:
            reward_components.append(progress_reward)
        self._last_progress_sum = progress_after

        # Time penalty (light)
        reward_components.append(rcfg.time_penalty)

        # Terminal checks
        opponents = [p for p in self.game.players if p.color.value != self.agent_color]
        terminal_reward = self.reward_calc.get_terminal_reward(agent_player, opponents)
        if terminal_reward != 0.0:
            reward_components.append(terminal_reward)
            terminated = True

        self.turns += 1
        self.episode_steps += 1
        if self.turns >= self.cfg.max_turns and not terminated:
            truncated = True

        # Prepare next agent dice if continuing (extra turn) and not done
        if not terminated and not truncated and not self.game.game_over:
            if extra_turn:
                self._pending_agent_dice, self._pending_valid_moves = self.move_utils._roll_new_agent_dice()  # agent retains turn
            else:
                self.opp_simulator._ensure_agent_turn()  # move through opponents done above
                if not self.game.game_over:
                    self._pending_agent_dice, self._pending_valid_moves = self.move_utils._roll_new_agent_dice()

        obs = self.obs_builder._build_observation(self.turns, self._pending_agent_dice)
        self.last_obs = obs
        self.done = terminated or truncated
        info = {
            "reward_components": reward_components,
            "dice": self._pending_agent_dice,
            "illegal_action": illegal,
            "action_mask": self.move_utils.action_masks(self._pending_valid_moves),
            "had_extra_turn": extra_turn,
        }
        total_reward = float(sum(reward_components))
        return obs, total_reward, terminated, truncated, info

    # ----------------------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------------------
    def render(self):  # minimal
        print(f"Turn {self.turns} agent_color={self.agent_color}")

    def close(self):
        pass
