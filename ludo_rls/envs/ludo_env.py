"""Gymnasium environment wrapping core Ludo engine.

Design goals:
- Thin adapter over existing game logic (`ludo.game.LudoGame`, strategies)
- Clean observation builder with extensible features
- Configurable reward shaping (modular components)
- Deterministic seeding & reproducibility
- Support vectorization (Stable-Baselines3 / RLlib)
- Compatible with Gymnasium API (reset(seed=...), step returning (obs, reward, terminated, truncated, info))

Self-play implementation:
- All 4 players are controlled by the same PPO model
- Agent position is randomized each episode to learn all perspectives
- Self-play simulator handles other players' turns using the same model
- No external strategies used - pure self-play

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
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ludo.constants import Colors, GameConstants
from ludo.game import LudoGame
from ludo.player import PlayerColor
from ludo.strategy import StrategyFactory

from .builders.observation_builder import ObservationBuilder

# from .calculators.reward_calculator import RewardCalculator
from .calculators.simple_reward_calculator import (
    SimpleRewardCalculator as RewardCalculator,
)
from .model import EnvConfig
from .simulators.self_play_simulator import SelfPlaySimulator
from .utils.move_utils import MoveUtils


class LudoGymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "name": "LudoGymEnv-v0"}

    def __init__(
        self, config: Optional[EnvConfig] = None, model: Optional[Any] = None
    ):  # gym style accepts **kwargs
        super().__init__()
        self.cfg = config or EnvConfig()
        self.rng = random.Random(self.cfg.seed)
        self.model = model

        # Build core game with fixed 4 players in canonical order (R,G,Y,B) if present
        order = [
            PlayerColor.RED,
            PlayerColor.GREEN,
            PlayerColor.YELLOW,
            PlayerColor.BLUE,
        ]
        self.game = LudoGame(order)

        # Attach opponent strategies (deferred to reset for proper seeding)
        self.agent_color = self.cfg.agent_color

        # Spaces (will be properly set after objects are created)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(50,),
            dtype=np.float32,  # Temporary shape
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

        # Initialize state variables that reset() expects
        self._pending_agent_dice = None
        self._pending_valid_moves = []
        self._last_progress_sum = 0.0

        # Create utilities in correct dependency order
        self.move_utils = MoveUtils(self.cfg, self.game, self.agent_color)
        self.obs_builder = ObservationBuilder(self.cfg, self.game, self.agent_color)
        self.reward_calc = RewardCalculator(self.cfg, self.game, self.agent_color)
        self.self_play_simulator = SelfPlaySimulator(
            self.cfg,
            self.game,
            self.agent_color,
            self.move_utils._roll_dice,
            self.move_utils._make_strategy_context,
            self.model,
        )

        # Now that all objects are created, set the proper observation space
        obs_dim = self.obs_builder._compute_observation_size()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):  # type: ignore[override]
        # Only reseed when explicit seed provided. This preserves stochasticity across episodes.
        if seed is not None:
            self.cfg.seed = seed
            self.rng.seed(seed)
            random.seed(seed)
        # Randomize agent color for self-play learning all positions
        self.agent_color = self.rng.choice(list(Colors.ALL_COLORS))
        # Recreate game for clean state
        order = [
            PlayerColor.RED,
            PlayerColor.GREEN,
            PlayerColor.YELLOW,
            PlayerColor.BLUE,
        ]
        self.game = LudoGame(order)
        # IMPORTANT: Rebuild helper objects so they reference the new game instance.
        # Previously these held stale pointers to the game created during __init__, causing
        # observations/rewards/opponent simulation to operate on an out-of-date game state
        # after each reset. This manifested as features like can_finish never updating when
        # tests mutated token positions post-reset (obs_builder still saw old tokens at home).
        self.move_utils = MoveUtils(self.cfg, self.game, self.agent_color)
        self.obs_builder.game = (
            self.game
        )  # safe to reuse builder instance but update pointer
        self.reward_calc.game = self.game
        # Recreate self-play simulator because it closes over move_utils methods
        self.self_play_simulator = SelfPlaySimulator(
            self.cfg,
            self.game,
            self.agent_color,
            self.move_utils._roll_dice,
            self.move_utils._make_strategy_context,
            self.model,
        )
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
        self.self_play_simulator._ensure_agent_turn()
        # Roll initial dice for agent decision
        self._pending_agent_dice, self._pending_valid_moves = (
            self.move_utils._roll_new_agent_dice()
        )
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

        reward_components: List[float] = []
        agent_player = next(
            p for p in self.game.players if p.color.value == self.agent_color
        )

        # Ensure we have a pending dice & valid moves (should always be true except pathological cases)
        if self._pending_agent_dice is None:
            self._pending_agent_dice, self._pending_valid_moves = (
                self.move_utils._roll_new_agent_dice()
            )

        dice_value = self._pending_agent_dice
        valid_moves = self._pending_valid_moves

        # Compute progress baseline
        progress_before = self.move_utils._compute_agent_progress_sum()

        # Handle no-move situation
        no_moves_available = len(valid_moves) == 0
        illegal = False
        move_res = {}
        diversity_bonus_triggered = False

        if not no_moves_available:
            valid_token_ids = [m["token_id"] for m in valid_moves]
            # Convert action to int in case it's a numpy array
            action = int(action)
            if action not in valid_token_ids:
                illegal = True
                # choose fallback (first valid) for execution so environment state advances
                exec_token_id = valid_token_ids[0]
            else:
                exec_token_id = action
            # Capture start position for progress calculation inside reward_calc
            start_pos = agent_player.tokens[exec_token_id].position
            move_res = self.game.execute_move(agent_player, exec_token_id, dice_value)
            move_res["start_position"] = start_pos

            # Check diversity bonus
            tok = agent_player.tokens[exec_token_id]
            if tok.position >= 0 and not self._token_activation_flags[exec_token_id]:
                diversity_bonus_triggered = True
                self._token_activation_flags[exec_token_id] = True

            extra_turn = move_res.get("extra_turn", False)
        else:
            # No valid moves: treat as skipped turn (no illegal penalty)
            extra_turn = False

        # Opponent simulation if no extra turn and game not over
        if not extra_turn and not self.game.game_over:
            # Advance turn for agent (dice consumed)
            self.game.next_turn()
            self.self_play_simulator._simulate_other_players(reward_components)

        # Progress shaping (after agent + opponents if any)
        progress_after = self.move_utils._compute_agent_progress_sum()
        progress_delta = progress_after - progress_before

        # Use comprehensive reward calculation
        step_components = self.reward_calc.compute_comprehensive_reward(
            move_res=move_res,
            progress_delta=progress_delta,
            extra_turn=extra_turn,
            diversity_bonus=diversity_bonus_triggered,
            illegal_action=illegal,
        )
        # Opponent components already accumulated in reward_components; capture their sum
        opponent_total = sum(reward_components)
        # Append atomic step components for logging only
        reward_components.extend(step_components.values())
        total_reward = opponent_total + sum(step_components.values())

        # Terminal checks
        opponents = [p for p in self.game.players if p.color.value != self.agent_color]
        terminal_reward = self.reward_calc.get_terminal_reward(agent_player, opponents)
        terminated = False
        truncated = False

        if terminal_reward != 0.0:
            terminated = True
            total_reward += terminal_reward

        self.turns += 1
        self.episode_steps += 1
        if self.turns >= self.cfg.max_turns and not terminated:
            truncated = True

        # Prepare next agent dice if continuing (extra turn) and not done
        if not terminated and not truncated and not self.game.game_over:
            if extra_turn:
                # Agent retained turn, just roll new dice
                self._pending_agent_dice, self._pending_valid_moves = (
                    self.move_utils._roll_new_agent_dice()
                )
            else:
                # Ensure pointer back to agent then roll
                self.self_play_simulator._ensure_agent_turn()
                if not self.game.game_over:
                    self._pending_agent_dice, self._pending_valid_moves = (
                        self.move_utils._roll_new_agent_dice()
                    )

        obs = self.obs_builder._build_observation(self.turns, self._pending_agent_dice)
        self.last_obs = obs
        self.done = terminated or truncated
        info = {
            "reward_components": reward_components,
            "step_breakdown": step_components,
            "dice": self._pending_agent_dice,
            "illegal_action": illegal,
            "action_mask": self.move_utils.action_masks(self._pending_valid_moves),
            "had_extra_turn": extra_turn,
            "progress_delta": progress_delta,
        }
        return obs, total_reward, terminated, truncated, info

    def render(self):  # minimal
        print(f"Turn {self.turns} agent_color={self.agent_color}")

    def set_model(self, model: Any):
        self.model = model
        # Update the self-play simulator with the new model
        self.self_play_simulator.model = model

    def close(self):
        pass
