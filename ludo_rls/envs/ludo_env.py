"""Single-seat training Ludo environment.

Option B implementation: At each episode reset, one player color is randomly selected
as the *training seat*. Only decisions of that seat are exposed via env.step to the
RL algorithm. All other seats are simulated internally using a *frozen snapshot* of
the policy parameters captured at reset (or a no-op random fallback if model absent).

Benefits:
 - Removes self-canceling rewards (captures / being captured) because only one
     perspective produces learning signals.
 - Reduces trajectory length (≈ 1/4 decisions emitted) focusing updates on target seat.
 - Stable opponents during the episode (snapshot) while refreshing each new episode.

Reward semantics:
 - Per-step reward components only when training seat acts.
 - Terminal win reward if training seat wins; no explicit loss penalty (draw penalty applies on timeout).
 - Opponent turns produce no external step so generate no direct rewards.

Observation: always from the training seat perspective (even across internal opponent turns, which are hidden).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import gymnasium as gym
from ludo_engine.models import MoveResult, ValidMove
import numpy as np
from gymnasium import spaces
from ludo_engine.core import LudoGame, PlayerColor
from ludo_engine.models import GameConstants
from stable_baselines3 import PPO

from ludo_rls.envs.calculators.simple_reward_calculator import (
    SimpleRewardCalculator as RewardCalculator,
)
from ludo_rls.envs.model import EnvConfig
from ludo_rls.envs.simulators.opponent_simulator import OpponentSimulator
from rl_base.envs.builders.observation_builder import ObservationBuilder
from rl_base.envs.utils.move_utils import MoveUtils


class LudoGymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "name": "LudoSelfPlayEnv-v1"}

    def __init__(
        self, config: Optional[EnvConfig] = None, model: Optional[PPO] = None
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
        # Training (learning) seat for this episode (decided in reset)
        self.training_color: str = self.game.get_current_player().color.value
        self.agent_color = self.training_color  # alias maintained for builders

        self._frozen_policy = None  # snapshot of model.policy at reset for opponents

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
        self.opp_simulator = None

        # Now that all objects are created, set the proper observation space
        obs_dim = self.obs_builder._compute_observation_size()
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,  # Temporary shape
        )
        self.action_space = spaces.Discrete(GameConstants.TOKENS_PER_PLAYER)

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
        # (Re)select training color (Option B always single-seat); randomize if enabled
        if self.cfg.randomize_training_color:
            self.training_color = self.rng.choice([i.value for i in order])
        else:
            self.training_color = self.game.get_current_player().color.value
        self.agent_color = self.training_color

        # Snapshot frozen opponent policy
        self._snapshot_policy()
        # Initialize or update simulator with current state
        self.opp_simulator = OpponentSimulator(
            self.cfg,
            self.game,
            self.training_color,
            self._frozen_policy,
            self.obs_builder,
            self._policy_action,  # Pass method reference
            self.rng,
        )
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

        # Advance internal simulation until it's training seat's turn
        self._advance_until_training_turn()
        obs = self.obs_builder._build_observation(self.turns, self._pending_agent_dice)
        self.last_obs = obs
        return obs, {}

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _snapshot_policy(self):
        """Capture a shallow snapshot reference of the current policy for opponents.

        We don't deep copy tensors (costly); we simply freeze by disabling grad and
        using deterministic inference from current model. If model absent, stays None.
        """
        if self.model is not None and hasattr(self.model, "policy"):
            self._frozen_policy = self.model.policy  # reference (weights won't change this episode if PPO updates after rollouts)
        else:
            self._frozen_policy = None

    def _policy_action(self, policy, obs: np.ndarray, valid_moves: List[ValidMove]) -> int:
        if not valid_moves:
            return -1
        mask = np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=np.int8)
        for m in valid_moves:
            mask[m.token_id] = 1
        # Fallback random if no policy
        if policy is None:
            legal_ids = [m.token_id for m in valid_moves]
            return self.rng.choice(legal_ids)
        obs_batch = obs[None, :]
        action, _ = policy.predict(obs_batch, deterministic=True)
        act = int(action)
        if mask[act] == 0:
            # pick first legal to keep game moving
            for i, v in enumerate(mask):
                if v == 1:
                    return i
        return act

    def _advance_until_training_turn(
        self, reward_components: Optional[List[float]] = None
    ):
        """Simulate opponent turns until it's training_color's turn, accumulating capture penalties if provided."""
        self.opp_simulator._simulate_until_agent_turn(reward_components, self.turns)
        # After simulation, if it's the agent's turn, roll dice
        if (
            not self.game.game_over
            and self.game.get_current_player().color.value == self.training_color
        ):
            self._pending_agent_dice, self._pending_valid_moves = (
                self.move_utils._roll_new_agent_dice()
            )

    def step(self, action: int):  # type: ignore[override]
        """Advance environment by one *training seat* decision.

        Opponent turns are internally simulated and not exposed. Extra turns for the training
        seat are returned consecutively (still a new step). After the move, if no extra turn,
        opponents are simulated until training seat's next turn or game end.
        """
        if self.done:
            return self.last_obs, 0.0, True, False, {}

        reward_components: List[float] = []
        agent_player = self.game.get_player_from_color(self.training_color)
        self.agent_color = self.training_color  # ensure consistency for builders

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
        move_res: MoveResult = None
        diversity_bonus_triggered = False
        token_positions_before: Optional[List[int]] = None
        masked_autocorrect = False

        if not no_moves_available:
            valid_token_ids = [m.token_id for m in valid_moves]
            action = int(action)  # ensure plain int
            token_positions_before = [t.position for t in agent_player.tokens]
            if action not in valid_token_ids:
                if self.cfg.use_action_mask:
                    exec_token_id = valid_token_ids[0]
                    masked_autocorrect = True
                else:
                    illegal = True
                    exec_token_id = valid_token_ids[0]
            else:
                exec_token_id = action

            start_pos = agent_player.tokens[exec_token_id].position
            move_res = self.game.execute_move(agent_player, exec_token_id, dice_value)
            move_res.old_position = start_pos

            tok = agent_player.tokens[exec_token_id]
            flags_for_player = self._token_activation_flags[agent_player.color.value]
            if tok.position >= 0 and not flags_for_player[exec_token_id]:
                diversity_bonus_triggered = True
                flags_for_player[exec_token_id] = True

            extra_turn = move_res.extra_turn
        else:
            extra_turn = False  # skipped turn

        # Advance to next player if no extra turn
        # If no extra turn: simulate opponents until training seat again
        if not extra_turn and not self.game.game_over:
            self.game.next_turn()
            self._advance_until_training_turn(reward_components)
            # sync (may still be training seat or game over)
            self.move_utils.agent_color = self.training_color
            self.obs_builder.agent_color = self.training_color
            self.reward_calc.agent_color = self.training_color

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
            token_positions_before=token_positions_before,
            masked_autocorrect=masked_autocorrect,
            reward_components=reward_components,
        )
        # Opponent components already accumulated in reward_components; capture their sum
        opponent_total = sum(reward_components)
        # Append atomic step components for logging only
        reward_components.extend(step_components.values())
        total_reward = opponent_total + sum(step_components.values())

        # Terminal checks
        opponents = [
            p for p in self.game.players if p.color.value != self.training_color
        ]
        # We'll compute truncated later and may re-request terminal reward including draw if needed
        terminal_reward = self.reward_calc.get_terminal_reward(
            agent_player, opponents, truncated=False
        )
        terminated = False
        truncated = False

        if terminal_reward != 0.0:
            terminated = True
            total_reward += terminal_reward

        self.turns += 1
        self.episode_steps += 1
        if self.turns >= self.cfg.max_turns and not terminated:
            truncated = True
            draw_reward = self.reward_calc.get_terminal_reward(
                agent_player, opponents, truncated=True
            )
            total_reward += draw_reward

        # Prepare next dice for the (possibly same or next) player if continuing
        if not terminated and not truncated and not self.game.game_over:
            if extra_turn:
                self._pending_agent_dice, self._pending_valid_moves = (
                    self.move_utils._roll_new_agent_dice()
                )
            else:
                # opponents already advanced; dice set in _advance_until_training_turn
                pass

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

    def set_model(self, model: PPO):
        # Placeholder for compatibility; model usage external in training loop.
        self.model = model
        self._snapshot_policy()

    def close(self):
        pass
