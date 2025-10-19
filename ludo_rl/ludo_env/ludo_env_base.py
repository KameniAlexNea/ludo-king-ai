import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ludo_engine.core import LudoGame, Player
from ludo_engine.models import (
    ALL_COLORS,
    GameConstants,
    MoveResult,
    MoveType,
    ValidMove,
)
from ludo_engine.strategies.base import Strategy
from ludo_engine.strategies.strategy import StrategyFactory
from stable_baselines3.common.utils import set_random_seed

from ludo_rl.config import EnvConfig
from ludo_rl.ludo_env.observation import (
    ContinuousObservationBuilder,
    DiscreteObservationBuilder,
    ObservationBuilderBase,
)
from ludo_rl.rewards.reward_adv_calculator import AdvancedRewardCalculator
from ludo_rl.rewards.reward_calculator import SparseRewardCalculator
from ludo_rl.rewards.risk_opportunity import (
    MergedRewardCalculator,
    RiskOpportunityCalculator,
)
from ludo_rl.utils.move_utils import MoveUtils


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
    is_game_over: bool = False  # Whether the game has ended
    agent_won: bool = (
        False  # Whether the agent won (only relevant when is_game_over=True)
    )


class LudoRLEnvBase(gym.Env):
    """Base Ludo RL environment with action masking support."""

    metadata = {"render_modes": ["human", "rgb_array"], "name": "LudoRLEnvBase-v0"}

    def __init__(self, cfg: EnvConfig):
        super().__init__()
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

        # Core game state
        self.agent_color = None
        self.game: Optional[LudoGame] = None
        self.obs_builder: Optional[ObservationBuilderBase] = None

        # Action and observation spaces
        # Observation space size will be set after obs_builder initialization
        self.action_space = spaces.Discrete(
            GameConstants.TOKENS_PER_PLAYER, seed=cfg.seed
        )

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
        reward_class = None
        if self.cfg.reward.reward_type == "sparse":
            reward_class = SparseRewardCalculator
        elif self.cfg.reward.reward_type == "risk_opportunity":
            reward_class = RiskOpportunityCalculator
        elif self.cfg.reward.reward_type == "merged":
            reward_class = MergedRewardCalculator
        elif self.cfg.reward.reward_type == "sparse_adv":
            reward_class = AdvancedRewardCalculator
        else:
            raise ValueError(f"Unknown reward type: {self.cfg.reward.reward_type}")
        self.reward_calc = reward_class()

        # Initialize spaces (will be updated on reset)
        self.observation_space = None

        self.opponents: Optional[List[str]] = None  # can be list of Strategy or names
        self.fixed_num_players: Optional[int] = None  # can be set externally

        self._initialize_game_state()

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
        colors = [
            (p.color, p) for p in self.game.players if p.color != self.agent_color
        ]
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

        # Reset reward calculator's episode state (if it supports it)
        if hasattr(self.reward_calc, "reset_for_new_episode"):
            self.reward_calc.reset_for_new_episode()

        # Start the first agent turn
        self._advance_to_agent_turn()
        self._roll_dice_for_agent()

        obs = self._build_observation()
        info = self._build_reset_info()

        self.episode_count += 1
        return obs, info

    def _setup_random_seed(self, seed: Optional[int]) -> None:
        """Set up random seeds for reproducibility."""
        if seed is None:
            return
        set_random_seed(seed)
        self.rng.seed(seed)

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
        if self.cfg.obs.discrete:
            self.obs_builder = DiscreteObservationBuilder(
                self.cfg, self.game, self.agent_color
            )
        else:
            self.obs_builder = ContinuousObservationBuilder(
                self.cfg, self.game, self.agent_color
            )

        # Update observation space with correct size
        # Support discrete observation encoding if requested
        if self.cfg.obs.discrete:
            # Use structured Dict observation space for discrete observations
            tokens_per_player = GameConstants.TOKENS_PER_PLAYER
            max_opponents = GameConstants.MAX_PLAYERS - 1
            pos_bins = (
                GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE + 1
            )
            self.observation_space = spaces.Dict(
                {
                    "agent_color": spaces.MultiDiscrete(
                        [2] * len(ALL_COLORS)
                    ),  # One-hot binary
                    "agent_progress": spaces.MultiDiscrete(
                        [pos_bins] * tokens_per_player
                    ),
                    "agent_vulnerable": spaces.MultiDiscrete([2] * tokens_per_player),
                    "opponents_positions": spaces.MultiDiscrete(
                        [pos_bins] * tokens_per_player * max_opponents
                    ),
                    "opponents_active": spaces.MultiDiscrete([2] * max_opponents),
                    "dice": spaces.MultiDiscrete([7]),
                }
            )
        else:
            # Use structured Dict observation space for continuous observations
            tokens_per_player = GameConstants.TOKENS_PER_PLAYER
            max_opponents = GameConstants.MAX_PLAYERS - 1
            dice_dim = 6
            self.observation_space = spaces.Dict(
                {
                    "agent_color": spaces.MultiBinary(len(ALL_COLORS)),
                    "agent_progress": spaces.Box(
                        low=0.0, high=1.0, shape=(tokens_per_player,), dtype=np.float32
                    ),
                    "agent_vulnerable": spaces.MultiBinary(tokens_per_player),
                    "opponents_positions": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(tokens_per_player * max_opponents,),
                        dtype=np.float32,
                    ),
                    "opponents_active": spaces.MultiBinary(max_opponents),
                    "dice": spaces.MultiBinary(dice_dim),
                }
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
        self.pending_valid_moves = self.game.get_valid_moves(player, self.pending_dice)

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

        terminated = self._check_termination(move_result)
        truncated = self._check_truncation()

        # Prepare next state
        if not terminated and not truncated and not self.game.game_over:
            self._prepare_next_agent_turn(move_result)

        # Update episode statistics
        if move_result:
            self.episode_stats.captured_opponents += len(move_result.captured_tokens)

        self.current_turn += 1

        obs = self._build_observation()
        step_info = self._build_step_info(move_result, is_illegal)

        # Calculate reward and check termination
        reward, reward_breakdown = self._calculate_reward(
            move_result, is_illegal, step_info
        )

        info_dict = step_info.__dict__
        info_dict["reward_breakdown"] = reward_breakdown

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
            token_id = self._get_opponent_action(player, dice)
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

    def _get_opponent_action(self, player: Player, dice: int) -> int:
        """Get the action for an opponent player."""
        context = self.game.get_ai_decision_context(dice)
        return player.make_strategic_decision(context)

    def _calculate_reward(
        self, move_result: Optional[MoveResult], is_illegal: bool, step_info: StepInfo
    ) -> tuple[float, Dict[str, float]]:
        """Calculate the reward for the current step."""
        return self.reward_calc.compute(
            self.game,
            self.agent_color,
            move_result,
            self.cfg,
            return_breakdown=True,
            is_illegal=is_illegal,
            episode_info=asdict(step_info),
        )

    def _check_termination(self, move_result: MoveResult) -> bool:
        """Check if the episode should terminate."""
        return (
            move_result.game_won
            or self.game.game_over
            or (self.game.winner is not None)
        )

    def _check_truncation(self) -> bool:
        """Check if the episode should be truncated due to max turns."""
        return self.current_turn >= self.cfg.max_turns and not self.game.game_over

    def _prepare_next_agent_turn(self, move_result: MoveResult) -> None:
        """Prepare the state for the next agent turn."""
        if move_result.extra_turn:
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

    def _build_step_info(self, move_result: MoveResult, is_illegal: bool) -> StepInfo:
        """Build the info dataclass for the step."""
        captured_opponents = len(move_result.captured_tokens)

        agent_player = self.game.get_player_from_color(self.agent_color)
        finished_tokens = agent_player.get_finished_tokens_count()

        # Check if game is over and if agent won
        is_game_over = self.game.game_over or (self.game.winner is not None)
        agent_won = False
        if is_game_over and self.game.winner is not None:
            agent_won = self.game.winner.color == self.agent_color

        return StepInfo(
            illegal_action=is_illegal,
            illegal_actions_total=self.illegal_actions,
            action_mask=MoveUtils.action_masks(self.pending_valid_moves),
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
            is_game_over=is_game_over,
            agent_won=agent_won,
        )

    # ---- action masking API for MaskablePPO/SubprocVecEnv -----------------
    def action_masks(self) -> list:
        """Return a boolean mask of valid actions for the current agent turn.

        This implements the environment-side action masking API expected by
        MaskablePPO when using SubprocVecEnv (ActionMasker wrapper cannot be
        used across subprocesses). The mask length must match `action_space.n`.
        """
        return MoveUtils.action_masks(self.pending_valid_moves)

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            # @TODO generate ludo env visual
            return np.zeros((480, 640, 3), dtype=np.uint8)
        elif self.render_mode == "human":
            # Print text representation
            if self.game:
                print(f"Current player: {self.game.get_current_player().color}")
                print(f"Dice: {self.pending_dice}")
                print(f"Turn: {self.current_turn}")
            else:
                print("Game not initialized")
        else:
            return super().render()
