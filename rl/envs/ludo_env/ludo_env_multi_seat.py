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
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ludo_engine.core import LudoGame, PlayerColor
from ludo_engine.models import Colors, GameConstants, MoveResult
from ludo_engine.strategies.strategy import StrategyFactory

from ..builders.observation_builder import ObservationBuilder
from ..calculators.simple_reward_calculator import (
    SimpleRewardCalculator as RewardCalculator,
)
from ..models.model_multi_seat import EnvConfig
from ..simulators.opponent_simulator_multi_seat import OpponentSimulator
from ..utils.move_utils import MoveUtils


class LudoGymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "name": "LudoGymEnv-v0"}

    def __init__(
        self, config: Optional[EnvConfig] = None
    ):  # gym style accepts **kwargs
        super().__init__()
        self.cfg = config or EnvConfig()
        self.rng = random.Random(self.cfg.seed)

        # Sample opponent strategies for this environment instance
        # Don't sample here - wait for reset() to ensure proper seeding
        self.opponent_strategies = []

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

        # Initialize state variables that reset() expects
        self._pending_agent_dice = None
        self._pending_valid_moves = []
        self._last_progress_sum = 0.0
        # Training progress hint (0..1). Can be set by trainer via set_attr during training.
        self._training_progress = None

        # Create utilities in correct dependency order
        self.move_utils = MoveUtils(self.cfg, self.game, self.agent_color)
        self.obs_builder = ObservationBuilder(self.cfg, self.game, self.agent_color)
        self.reward_calc = RewardCalculator(self.cfg, self.game, self.agent_color)
        self.opp_simulator = OpponentSimulator(
            self.cfg,
            self.game,
            self.agent_color,
            self.game.roll_dice,
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
            np.random.seed(seed)

        # 1) Possibly randomize agent color and pick opponents (curriculum-aware)
        if self.cfg.randomize_agent_seat:
            self.agent_color = self.rng.choice(list(Colors.ALL_COLORS))
        self.opponent_strategies = self._select_opponents()

        # 2) Rebuild game and helper objects, attach strategies
        self._rebuild_game_and_helpers()

        # 3) Initialize episode bookkeeping and first agent decision
        self.turns = 0
        self.episode_steps = 0
        self.done = False
        self._pending_agent_dice = None
        self._pending_valid_moves = []
        self._last_progress_sum = self.move_utils._compute_agent_progress_sum()

        # Ensure agent's turn and roll initial dice
        self.opp_simulator._ensure_agent_turn()
        self._pending_agent_dice, self._pending_valid_moves = (
            self.move_utils._roll_new_agent_dice()
        )
        obs = self.obs_builder._build_observation(self.turns, self._pending_agent_dice)
        self.last_obs = obs

        # Diagnostics
        if not hasattr(self, "_episode_count"):
            self._episode_count = 0
        info = {
            "episode": int(self._episode_count),
            "opponents": list(self.opponent_strategies),
            "progress": (
                None
                if self._training_progress is None
                else float(self._training_progress)
            ),
        }
        self._episode_count += 1
        return obs, info

    # ------------------------------
    # Internal helpers (refactor)
    # ------------------------------
    def _select_opponents(self) -> List[str]:
        """Select 3 opponent strategy names based on curriculum settings.

        Falls back to uniform sampling from candidates if curriculum disabled
        or misconfigured.
        """
        # Curriculum disabled/25% random for diversity => uniform sample
        if not self.cfg.opponent_curriculum.enabled or self.rng.random() < 0.25:
            return self.rng.sample(self.cfg.opponents.candidates, 3)

        occ = self.cfg.opponent_curriculum
        phase_idx = self._determine_curriculum_phase()

        # Filter buckets by allowed candidates
        allowed = set(self.cfg.opponents.candidates)

        def sample_from(bucket: List[str], k: int) -> List[str]:
            pool = [s for s in bucket if not allowed or s in allowed]
            if not pool:
                return []
            if len(pool) < k:
                return [self.rng.choice(pool) for _ in range(k)]
            return self.rng.sample(pool, k)

        if phase_idx == 0:
            picks = sample_from(occ.poor, 2) + sample_from(occ.medium, 1)
        elif phase_idx == 1:
            picks = (
                sample_from(occ.poor, 1)
                + sample_from(occ.medium, 1)
                + sample_from(occ.hard, 1)
            )
        else:
            picks = sample_from(occ.hard, 2) + sample_from(occ.medium, 1)

        if len(picks) < 3:
            return self.rng.sample(self.cfg.opponents.candidates, 3)
        return picks

    def _determine_curriculum_phase(self) -> int:
        """Map training progress or episode count to phase index [0..3]."""
        occ = self.cfg.opponent_curriculum
        # Prefer global training progress if provided
        if occ.use_progress and self._training_progress is not None:
            p = max(0.0, min(1.0, float(self._training_progress)))
            b = occ.progress_boundaries
            if p < b[0]:
                return 0
            if p < b[1]:
                return 1
            if p < b[2]:
                return 2
            return 3
        # Fallback to per-env episode-based thresholds
        if not hasattr(self, "_episode_count"):
            self._episode_count = 0
        for i, limit in enumerate(occ.phase_episodes):
            if self._episode_count < limit:
                return i
        return len(occ.phase_episodes) - 1

    def _rebuild_game_and_helpers(self) -> None:
        """Reset game state and rewire all helper objects and opponent strategies."""
        order = [
            PlayerColor.RED,
            PlayerColor.GREEN,
            PlayerColor.YELLOW,
            PlayerColor.BLUE,
        ]
        self.game = LudoGame(order)

        # Refresh helpers to bind to the new game instance
        self.move_utils = MoveUtils(self.cfg, self.game, self.agent_color)
        # Rebuild observation builder if agent color changed
        self.obs_builder = ObservationBuilder(self.cfg, self.game, self.agent_color)
        self.reward_calc.game = self.game
        self.opp_simulator = OpponentSimulator(
            self.cfg,
            self.game,
            self.agent_color,
            self.game.roll_dice,
        )

        # Attach chosen opponent strategies
        non_agent_colors = [c for c in Colors.ALL_COLORS if c != self.agent_color]
        for i, color in enumerate(non_agent_colors):
            player = self.game.get_player_from_color(color)
            strat_name = self.opponent_strategies[i]
            try:
                player.set_strategy(StrategyFactory.create_strategy(strat_name))
            except Exception:
                # Keep default if creation fails
                pass

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
        agent_player = self.game.get_player_from_color(self.agent_color)

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
        diversity_bonus_triggered = False

        start_pos = -1  # Default for no-move cases
        if no_moves_available:
            # No valid moves available this turn (e.g., no 6 to exit, or blocked)
            # This is NOT an illegal action by the agent; simply skip move execution.
            illegal = False
            # Skip the turn (no move execution)
            extra_turn = False
            # Create a dummy move result for no-move case

            move_res = MoveResult(
                success=True,
                player_color=self.agent_color,
                token_id=0,
                dice_value=dice_value,
                old_position=-1,
                new_position=-1,
                captured_tokens=[],
                finished_token=False,
                extra_turn=False,
                error=None,
                game_won=False,
            )
        else:
            valid_token_ids = [m.token_id for m in valid_moves]
            # Convert action to int in case it's a numpy array
            action = int(action)
            if action not in valid_token_ids:
                illegal = True
                # For illegal actions, we have two options:
                # 1. Execute a random valid move (current approach)
                # 2. Skip the turn entirely
                # Option 1 is better for learning as it maintains game flow
                exec_token_id = self.rng.choice(
                    valid_token_ids
                )  # Random instead of first
            else:
                exec_token_id = action
            # Capture start position for progress calculation inside reward_calc
            start_pos = agent_player.tokens[exec_token_id].position
            move_res = self.game.execute_move(agent_player, exec_token_id, dice_value)
            # Note: start_pos stored separately since move_res is immutable dataclass

            # Check diversity bonus - give bonus every time a token is activated from home
            tok = agent_player.tokens[exec_token_id]
            if tok.position >= 0 and start_pos < 0:
                diversity_bonus_triggered = True
                # Allow repeated bonuses for reactivating tokens

            extra_turn = move_res.extra_turn

        # Opponent simulation if no extra turn and game not over
        if not extra_turn and not self.game.game_over:
            # Advance turn for agent (dice consumed)
            self.game.next_turn()
            self.opp_simulator._simulate_opponents(reward_components)

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
            reward_components=reward_components,
            start_position=start_pos,
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
                self.opp_simulator._ensure_agent_turn()
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

    def close(self):
        pass
