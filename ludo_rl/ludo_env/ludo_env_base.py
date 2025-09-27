import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger
from ludo_engine.core import LudoGame, Player, PlayerColor
from ludo_engine.models import (
    ALL_COLORS,
    GameConstants,
    MoveResult,
    MoveType,
    ValidMove,
)
from ludo_engine.strategies.base import Strategy  # type: ignore
from ludo_engine.strategies.strategy import StrategyFactory  # type: ignore

from ludo_rl.config import EnvConfig
from ludo_rl.ludo_env.observation import ObservationBuilder
from ludo_rl.utils.move_utils import MoveUtils
from ludo_rl.utils.reward_calculator import RewardCalculator


@dataclass
class EpisodeStats:
    """Tracks episode-level statistics."""

    captured_opponents: int = 0
    captured_by_opponents: int = 0
    capture_ops_available: int = 0
    capture_ops_taken: int = 0
    finish_ops_available: int = 0
    finish_ops_taken: int = 0
    home_exit_ops_available: int = 0
    home_exit_ops_taken: int = 0


@dataclass
class StepInfo:
    """Information returned from a step."""

    illegal_action: bool
    illegal_actions_total: int
    action_mask: np.ndarray
    captured_opponents: int
    captured_by_opponents: int
    episode_captured_opponents: int
    episode_captured_by_opponents: int
    finished_tokens: int
    episode_capture_ops_available: int
    episode_capture_ops_taken: int
    episode_finish_ops_available: int
    episode_finish_ops_taken: int
    episode_home_exit_ops_available: int
    episode_home_exit_ops_taken: int


class LudoRLEnvBase(gym.Env):
    """Base Ludo RL environment with action masking support."""

    metadata = {"render_modes": ["human"], "name": "LudoRLEnvBase-v0"}

    def __init__(self, cfg: EnvConfig):
        super().__init__()
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

        # Core game state
        self.agent_color = None
        self.game: Optional[LudoGame] = None
        self.obs_builder: Optional[ObservationBuilder] = None

        # Action and observation spaces
        # Observation space size will be set after obs_builder initialization
        self.action_space = spaces.Discrete(GameConstants.TOKENS_PER_PLAYER, seed=cfg.seed)

        # Turn state
        self.pending_dice: Optional[int] = None
        self.pending_valid_moves: List[ValidMove] = []

        # Episode tracking
        self.episode_count = 0
        self.current_turn = 0
        self.illegal_actions = 0

        # Statistics
        self.episode_stats = EpisodeStats()
        self.captured_by_opponents_this_turn = 0

        # Components
        self.reward_calc = RewardCalculator()

        # Initialize spaces (will be updated on reset)
        self.observation_space = None

        self.opponents: Optional[List[str]] = None  # can be list of Strategy or names
        self.fixed_num_players: Optional[int] = None  # can be set externally

        self._initialize_game_state()

    def set_capture_scale(self, scale: float) -> None:
        """Set capture reward scale dynamically."""
        self.cfg.reward.capture_reward_scale = scale

    # ---- hooks for subclasses -------------------------------------------------
    def on_reset_before_attach(self, options: Optional[Dict[str, Any]] = None) -> None:
        """Subclass hook before attaching opponents; game and obs_builder are initialized."""
        return None

    def attach_opponents(self, options: Optional[Dict[str, Any]] = None) -> None:
        """Subclass must implement opponent strategy attachment (Strategy or names)."""
        raise NotImplementedError

    def extra_reset_info(self) -> Dict[str, Any]:
        """Optional extra info to include on reset."""
        return {}

    # ---- helpers --------------------------------------------------------------
    def _attach_strategies_mixed(self, strategies: List) -> None:
        """Attach strategies to non-agent players. Items can be Strategy instances or names."""
        if len(strategies) != len(self.game.players) - 1:
            raise ValueError(
                f"Number of strategies ({len(strategies)}) must match number of opponent players ({len(self.game.players) - 1})"
            )
        colors = [(p.color, p) for p in self.game.players if p.color != self.agent_color]
        for strat, (color, player) in zip(strategies, colors):
            try:
                if Strategy is not object and isinstance(strat, Strategy):
                    player.set_strategy(strat)
                elif StrategyFactory is not None:
                    player.set_strategy(StrategyFactory.create_strategy(strat))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to set strategy for player {color}: {e}"
                ) from e

    # ---- gym api --------------------------------------------------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        """Reset the environment for a new episode."""
        self._setup_random_seed(seed)

        # Merge attributes set by set_attr into options
        options = options or {}
        if self.opponents:
            options["opponents"] = self.opponents
        if self.fixed_num_players:
            options["fixed_num_players"] = self.fixed_num_players

        # Allow callers to temporarily force a player count for this reset
        orig_fixed = getattr(self.cfg, "fixed_num_players", None)
        if "fixed_num_players" in options:
            self.cfg.fixed_num_players = options["fixed_num_players"]
        self._initialize_game_state()
        # restore original fixed_num_players so the env.cfg is unchanged across resets
        self.cfg.fixed_num_players = orig_fixed
        self._reset_episode_stats()

        # Subclass-specific setup and opponent attachment
        self.on_reset_before_attach(options)
        self.attach_opponents(options)

        # Start the first agent turn
        self._advance_to_agent_turn()
        self._roll_dice_for_agent()

        obs = self._build_observation()
        info = self._build_reset_info()

        self.episode_count += 1
        return obs, info

    def _setup_random_seed(self, seed: Optional[int]) -> None:
        """Set up random seeds for reproducibility."""
        if seed is not None:
            self.rng.seed(seed)
            random.seed(seed)
            np.random.seed(seed)

    def _initialize_game_state(self) -> None:
        """Initialize the game and agent color."""
        if self.cfg.randomize_agent:
            self.agent_color = self.rng.choice(ALL_COLORS)

        # determine number of players for this episode
        if self.cfg.fixed_num_players is not None:
            num_players = int(self.cfg.fixed_num_players)
        else:
            num_players = int(self.rng.choice(self.cfg.allowed_player_counts))

        # build chosen_colors according to requested groupings:
        # 2 players -> agent and opposite (index + 2)
        # 3 players -> agent and next two clockwise (index, index+1, index+2)
        # 4 players -> all colors rotated so agent is first
        start_idx = ALL_COLORS.index(self.agent_color)
        if num_players == 2:
            chosen_colors = [
                ALL_COLORS[start_idx],
                ALL_COLORS[(start_idx + 2) % len(ALL_COLORS)],
            ]
        elif num_players == 3:
            chosen_colors = [
                ALL_COLORS[(start_idx + i) % len(ALL_COLORS)] for i in range(3)
            ]
        elif num_players >= len(ALL_COLORS):
            # default to full set rotated so agent is first
            chosen_colors = [
                ALL_COLORS[(start_idx + i) % len(ALL_COLORS)]
                for i in range(len(ALL_COLORS))
            ]
        else:
            # fallback: take the first num_players clockwise including agent
            chosen_colors = [
                ALL_COLORS[(start_idx + i) % len(ALL_COLORS)]
                for i in range(num_players)
            ]

        # create game with selected colors (agent will be first in the game's player order)
        self.game = LudoGame(chosen_colors)
        self.obs_builder = ObservationBuilder(self.cfg, self.game, self.agent_color)

        # Update observation space with correct size
        # Support discrete observation encoding if requested
        if getattr(self.cfg, "obs", None) and getattr(self.cfg.obs, "discrete", False):
            # Build MultiDiscrete nvec from observation builder
            nvec = self.obs_builder.compute_discrete_dims()
            # gym expects an array-like of ints for MultiDiscrete
            self.observation_space = spaces.MultiDiscrete(nvec)
        else:
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.obs_builder.size,), dtype=np.float32
            )

    def _reset_episode_stats(self) -> None:
        """Reset all episode-level statistics and counters."""
        self.current_turn = 0
        self.illegal_actions = 0
        self.episode_stats = EpisodeStats()
        self.captured_by_opponents_this_turn = 0
        self.pending_dice = None
        self.pending_valid_moves = []

    def _advance_to_agent_turn(self) -> None:
        """Advance the game until it's the agent's turn."""
        while (
            not self.game.game_over
            and self.game.get_current_player().color != self.agent_color
        ):
            self._simulate_opponent_turn()

    def _roll_dice_for_agent(self) -> None:
        """Roll dice and get valid moves for the agent."""
        player = self.game.get_current_player()
        assert player.color == self.agent_color, "Not agent's turn"
        self.pending_dice = self.game.roll_dice()
        self.pending_valid_moves = self.game.get_valid_moves(
            player, self.pending_dice
        )

    def _build_observation(self) -> np.ndarray:
        """Build the current observation."""
        dice_value = self.pending_dice or 0
        return self.obs_builder.build(self.current_turn, dice_value)

    def _build_reset_info(self) -> Dict[str, Any]:
        """Build info dictionary for reset."""
        info = {"episode": self.episode_count}
        info.update(self.extra_reset_info())
        return info

    def step(self, action: int):
        """Execute one step in the environment."""
        if self.game.game_over:
            return self._build_terminal_observation(), 0.0, True, False, {}

        # Execute the agent's action
        move_result = self._execute_agent_action(action)
        is_illegal = move_result is None

        # Handle opponent turns if no extra turn
        if not (move_result and move_result.extra_turn) and not self.game.game_over:
            self._handle_opponent_turns()

        # Calculate reward and check termination
        reward, reward_breakdown = self._calculate_reward(move_result, is_illegal)
        terminated = self._check_termination(move_result)
        truncated = self._check_truncation()

        # Prepare next state
        if not terminated and not truncated and not self.game.game_over:
            self._prepare_next_agent_turn(move_result)

        # Update episode statistics
        if move_result:
            self.episode_stats.captured_opponents += len(move_result.captured_tokens)

            # Debug logging for captures
            if self.cfg.debug_capture_logging and move_result.captured_tokens:
                logger.debug(
                    f"[CaptureEvent] turn={self.current_turn} dice={self.pending_dice} "
                    f"offensive={len(move_result.captured_tokens)} "
                    f"defensive_inc={self.captured_by_opponents_this_turn} "
                    f"cumulative_off={self.episode_stats.captured_opponents} "
                    f"cumulative_def={self.episode_stats.captured_by_opponents}"
                )

        self.current_turn += 1

        obs = self._build_observation()
        step_info = self._build_step_info(move_result, is_illegal)

        info_dict = step_info.__dict__
        # Attach reward breakdown totals for diagnostic logging
        try:
            info_dict["reward_breakdown"] = reward_breakdown
        except Exception:
            info_dict["reward_breakdown"] = {}

        return obs, reward, terminated, truncated, info_dict

    def _execute_agent_action(self, action: int) -> Optional[MoveResult]:
        """Execute the agent's chosen action and return the result."""
        agent = self.game.get_current_player()
        dice = self.pending_dice
        valid_moves = self.pending_valid_moves

        if not valid_moves:
            # No valid moves available
            return self._create_no_move_result(dice)

        # Validate and execute the action
        action = int(action)
        valid_token_ids = [move.token_id for move in valid_moves]

        if action not in valid_token_ids:
            self.illegal_actions += 1
            action = self.rng.choice(valid_token_ids)  # Choose random valid action

        # Track opportunities before move
        if self.cfg.track_opportunities:
            self._track_opportunities_before_move(valid_moves)

        # Execute the move
        move_result = self.game.execute_move(agent, action, dice)

        # Track opportunities after move
        if self.cfg.track_opportunities and valid_moves:
            self._track_opportunities_after_move(move_result, valid_moves, action)

        return move_result

    def _create_no_move_result(self, dice: int) -> MoveResult:
        """Create a move result for when no valid moves are available."""
        agent = self.game.get_current_player()
        return MoveResult(
            success=True,
            player_color=agent.color,
            token_id=0,
            dice_value=dice,
            old_position=-1,
            new_position=-1,
            captured_tokens=[],
            finished_token=False,
            extra_turn=False,
            error=None,
            game_won=False,
        )

    def _track_opportunities_before_move(self, valid_moves: List[ValidMove]) -> None:
        """Track available opportunities before executing a move."""
        # Capture opportunities
        capture_ops = sum(1 for move in valid_moves if move.captures_opponent)
        self.episode_stats.capture_ops_available += capture_ops

        # Finish opportunities
        finish_ops = sum(
            1
            for move in valid_moves
            if move.target_position == GameConstants.FINISH_POSITION
        )
        self.episode_stats.finish_ops_available += finish_ops

        # Home exit opportunities
        exit_ops = sum(
            1 for move in valid_moves if move.move_type == MoveType.EXIT_HOME
        )
        self.episode_stats.home_exit_ops_available += exit_ops

        # Debug logging
        if self.cfg.debug_capture_logging and self.current_turn < 100:
            self._log_opportunity_debug(valid_moves, finish_ops, exit_ops, capture_ops)

    def _track_opportunities_after_move(
        self, move_result: MoveResult, valid_moves: List[ValidMove], action: int
    ) -> None:
        """Track taken opportunities after executing a move."""
        chosen_move = next(
            (move for move in valid_moves if move.token_id == action), None
        )
        if chosen_move is None:
            return

        # Track capture opportunity taken
        if chosen_move.captures_opponent:
            self.episode_stats.capture_ops_taken += 1

        # Track finish opportunity taken
        finished_token = (
            move_result.finished_token
            or move_result.new_position == GameConstants.FINISH_POSITION
        )
        if finished_token:
            self.episode_stats.finish_ops_taken += 1

        # Track home exit opportunity taken
        if chosen_move.move_type == MoveType.EXIT_HOME:
            self.episode_stats.home_exit_ops_taken += 1

    def _log_opportunity_debug(
        self,
        valid_moves: List[ValidMove],
        finish_ops: int,
        exit_ops: int,
        capture_ops: int,
    ) -> None:
        """Log debug information about opportunities."""
        try:
            move_types = [move.move_type for move in valid_moves]
            logger.debug(
                f"[OppDebug] turn={self.current_turn} dice={self.pending_dice} "
                f"move_types={move_types} fin_by_pos_avail+={finish_ops} "
                f"exit_avail+={exit_ops} cap_avail+={capture_ops}"
            )
        except Exception:
            pass

    def _handle_opponent_turns(self) -> None:
        """Handle all opponent turns until it's the agent's turn again."""
        self.game.next_turn()
        while (
            not self.game.game_over
            and self.game.get_current_player().color != self.agent_color
        ):
            self._simulate_opponent_turn()

    def _simulate_opponent_turn(self) -> None:
        """Simulate a single opponent turn."""
        player = self.game.get_current_player()
        if player.color == self.agent_color:
            return

        dice = self.game.roll_dice()
        valid_moves = self.game.get_valid_moves(player, dice)

        if valid_moves:
            token_id = self._get_opponent_action(player, dice, valid_moves)
            move_result = self.game.execute_move(player, token_id, dice)

            # Track captures on agent
            if move_result.captured_tokens:
                agent_tokens_captured = sum(
                    1
                    for token in move_result.captured_tokens
                    if token.player_color == self.agent_color
                )
                self.captured_by_opponents_this_turn += agent_tokens_captured
                self.episode_stats.captured_by_opponents += agent_tokens_captured

            if not move_result.extra_turn:
                self.game.next_turn()
        else:
            self.game.next_turn()

    def _get_opponent_action(
        self, player: Player, dice: int, valid_moves: List[ValidMove]
    ) -> int:
        """Get the action for an opponent player."""
        try:
            context = self.game.get_ai_decision_context(dice)
            return player.make_strategic_decision(context)
        except Exception:
            return self.rng.choice(valid_moves).token_id

    def _calculate_reward(
        self, move_result: Optional[MoveResult], is_illegal: bool
    ) -> float:
        """Calculate the reward for the current step."""
        if move_result is None:
            # No valid moves case
            move_result = self._create_no_move_result(self.pending_dice)

        return self.reward_calc.compute_with_breakdown(
            res=move_result,
            illegal=is_illegal,
            cfg=self.cfg,
            game_over=self.game.game_over,
            captured_by_opponents=self.captured_by_opponents_this_turn,
            extra_turn=bool(move_result.extra_turn),
            winner=self.game.winner,
            agent_color=self.agent_color,
            home_tokens=self._count_home_tokens(),
        )

    def _count_home_tokens(self) -> int:
        """Count how many of the agent's tokens are still at home."""
        agent_player = self.game.get_player_from_color(self.agent_color)
        return sum(
            1
            for pos in agent_player.player_positions()
            if pos == GameConstants.HOME_POSITION
        )

    def _check_termination(self, move_result: Optional[MoveResult]) -> bool:
        """Check if the episode should terminate."""
        if move_result is None:
            return False
        return move_result.game_won or self.game.game_over

    def _check_truncation(self) -> bool:
        """Check if the episode should be truncated due to max turns."""
        return self.current_turn >= self.cfg.max_turns and not self.game.game_over

    def _prepare_next_agent_turn(self, move_result: Optional[MoveResult]) -> None:
        """Prepare the state for the next agent turn."""
        has_extra_turn = move_result and move_result.extra_turn

        if has_extra_turn:
            self._roll_dice_for_agent()
        else:
            self._advance_to_agent_turn()
            if not self.game.game_over:
                self._roll_dice_for_agent()

        # Reset per-turn counters
        self.captured_by_opponents_this_turn = 0

    def _build_terminal_observation(self) -> np.ndarray:
        """Build observation for terminal states."""
        return self.obs_builder.build(self.current_turn, 0)

    def _build_step_info(
        self, move_result: Optional[MoveResult], is_illegal: bool
    ) -> StepInfo:
        """Build the info dataclass for the step."""
        captured_opponents = len(move_result.captured_tokens) if move_result else 0

        agent_player = self.game.get_player_from_color(self.agent_color)
        finished_tokens = agent_player.get_finished_tokens_count()

        return StepInfo(
            illegal_action=is_illegal,
            illegal_actions_total=self.illegal_actions,
            action_mask=MoveUtils.action_mask(self.pending_valid_moves),
            captured_opponents=captured_opponents,
            captured_by_opponents=self.captured_by_opponents_this_turn,
            episode_captured_opponents=self.episode_stats.captured_opponents,
            episode_captured_by_opponents=self.episode_stats.captured_by_opponents,
            finished_tokens=finished_tokens,
            episode_capture_ops_available=self.episode_stats.capture_ops_available,
            episode_capture_ops_taken=self.episode_stats.capture_ops_taken,
            episode_finish_ops_available=self.episode_stats.finish_ops_available,
            episode_finish_ops_taken=self.episode_stats.finish_ops_taken,
            episode_home_exit_ops_available=self.episode_stats.home_exit_ops_available,
            episode_home_exit_ops_taken=self.episode_stats.home_exit_ops_taken,
        )

    # ---- action masking API for MaskablePPO/SubprocVecEnv -----------------
    def action_masks(self) -> list:
        """Return a boolean mask of valid actions for the current agent turn.

        This implements the environment-side action masking API expected by
        MaskablePPO when using SubprocVecEnv (ActionMasker wrapper cannot be
        used across subprocesses). The mask length must match `action_space.n`.
        """
        try:
            mask = MoveUtils.action_mask(self.pending_valid_moves)
            # ensure a plain Python list of bools
            return [bool(x) for x in mask]
        except Exception:
            # If anything goes wrong, fall back to allowing all actions
            return [True] * int(self.action_space.n)
