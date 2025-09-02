"""Self-play Ludo environment (shared single policy controlling all 4 players).

Semantics Option A (sequential turns):
Each env.step() represents exactly one player decision (possibly followed by extra chained decisions via repeated steps if six rolled, etc.).
The acting player's color is stored in `agent_color` and changes automatically when turn advances.
All four players share one policy (PPO) and are exposed to the learner uniformly in board order (R -> G -> Y -> B -> ...).

Observation (per current player perspective) includes:
    - Current player's 4 token normalized positions
    - Other 3 players' token positions (12)
    - Finished token fractions per player (4)
    - Finish-possible flag, normalized dice value, progress stats etc. (remaining scalars)

Action space: Discrete(4) selecting which of the current player's tokens to move; invalid choices incur penalty and fallback to first valid.

Rewards: Shaped per move for current acting player only. Terminal reward granted when that player wins (or loses relative to others finishing first).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ludo.constants import GameConstants
from ludo.game import LudoGame
from ludo.player import PlayerColor

from .builders.observation_builder import ObservationBuilder

# from .calculators.reward_calculator import RewardCalculator
from .calculators.simple_reward_calculator import (
    SimpleRewardCalculator as RewardCalculator,
)
from .model import EnvConfig
from .utils.move_utils import MoveUtils


class LudoGymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "name": "LudoSelfPlayEnv-v1"}

    def __init__(
        self, config: Optional[EnvConfig] = None, model: Optional[Any] = None
    ):  # gym style accepts **kwargs
        super().__init__()
        self.cfg = config or EnvConfig()
        self.rng = random.Random(self.cfg.seed)
        self.model = model

        # Build core game with fixed 4 players in canonical order (R,G,Y,B)
        order = [
            PlayerColor.RED,
            PlayerColor.GREEN,
            PlayerColor.YELLOW,
            PlayerColor.BLUE,
        ]
        self.game = LudoGame(order)
        # Current acting player perspective (no randomness now)
        self.agent_color = self.game.get_current_player().color.value

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
        # Per-player token activation diversity flags: {color: {token_id: bool}}
        self._token_activation_flags = {
            p.color.value: {i: False for i in range(GameConstants.TOKENS_PER_PLAYER)}
            for p in self.game.players
        }

        # Initialize state variables that reset() expects
        self._pending_agent_dice = None
        self._pending_valid_moves = []
        self._last_progress_sum = 0.0

        # Create utilities in correct dependency order
        self.move_utils = MoveUtils(self.cfg, self.game, self.agent_color)
        self.obs_builder = ObservationBuilder(self.cfg, self.game, self.agent_color)
        self.reward_calc = RewardCalculator(self.cfg, self.game, self.agent_color)
        # No separate simulator – each env.step corresponds to exactly one player decision.

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
        # Recreate game for clean state (fixed order)
        order = [
            PlayerColor.RED,
            PlayerColor.GREEN,
            PlayerColor.YELLOW,
            PlayerColor.BLUE,
        ]
        self.game = LudoGame(order)
        # Set current player perspective deterministically
        self.agent_color = self.game.get_current_player().color.value
        # Rebuild helper objects references
        self.move_utils = MoveUtils(self.cfg, self.game, self.agent_color)
        self.obs_builder.game = self.game
        self.obs_builder.agent_color = self.agent_color
        self.reward_calc.game = self.game
        self.reward_calc.agent_color = self.agent_color
        self.turns = 0
        self.episode_steps = 0
        self.done = False
        self._token_activation_flags = {
            p.color.value: {i: False for i in range(GameConstants.TOKENS_PER_PLAYER)}
            for p in self.game.players
        }
        self._pending_agent_dice = None
        self._pending_valid_moves = []
        self._last_progress_sum = self.move_utils._compute_agent_progress_sum()

        # Roll initial dice for first player's decision
        self._pending_agent_dice, self._pending_valid_moves = (
            self.move_utils._roll_new_agent_dice()
        )
        obs = self.obs_builder._build_observation(self.turns, self._pending_agent_dice)
        self.last_obs = obs
        return obs, {}

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
                if self.cfg.use_action_mask:
                    # Auto-correct silently (no illegal penalty) when mask enabled
                    exec_token_id = valid_token_ids[0]
                else:
                    illegal = True
                    exec_token_id = valid_token_ids[0]
            else:
                exec_token_id = action
            # Capture start position for progress calculation inside reward_calc
            start_pos = agent_player.tokens[exec_token_id].position
            move_res = self.game.execute_move(agent_player, exec_token_id, dice_value)
            move_res["start_position"] = start_pos

            # Check diversity bonus
            tok = agent_player.tokens[exec_token_id]
            flags_for_player = self._token_activation_flags[agent_player.color.value]
            if tok.position >= 0 and not flags_for_player[exec_token_id]:
                diversity_bonus_triggered = True
                flags_for_player[exec_token_id] = True

            extra_turn = move_res.get("extra_turn", False)
        else:
            # No valid moves: treat as skipped turn (no illegal penalty)
            extra_turn = False

        # Advance to next player if no extra turn
        if not extra_turn and not self.game.game_over:
            self.game.next_turn()
            self.agent_color = self.game.get_current_player().color.value
            # Sync utility perspectives
            self.move_utils.agent_color = self.agent_color
            self.obs_builder.agent_color = self.agent_color
            self.reward_calc.agent_color = self.agent_color

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

        # Prepare next dice for the (possibly same or next) player if continuing
        if not terminated and not truncated and not self.game.game_over:
            if extra_turn:
                # Agent retained turn, just roll new dice
                self._pending_agent_dice, self._pending_valid_moves = (
                    self.move_utils._roll_new_agent_dice()
                )
            else:
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
        # Placeholder for compatibility; model usage external in training loop.
        self.model = model

    def close(self):
        pass
