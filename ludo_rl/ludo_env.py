import os
import random
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger

from .ludo_king.config import config as king_config
from .ludo_king.config import reward_config
from .ludo_king.game import Game
from .ludo_king.player import Player
from .ludo_king.reward import (
    compute_draw_reward,
    compute_invalid_action_penalty,
    compute_sparse_rewards,
    compute_terminal_reward,
)
from .ludo_king.simulator import Simulator
from .ludo_king.types import Color, Move
from .strategy.registry import STRATEGY_REGISTRY
from .strategy.registry import available as available_strategies
from .utils.opponent_lineup import OpponentLineupSampler, create_default_sampler

STRATEGY_AVAILABLE = [
    s for s in os.getenv("OPPONENTS", ",".join(available_strategies())).split(",") if s
]
STRATEGY_SELECTION = int(os.getenv("STRATEGY_SELECTION", "0"))


def get_observation_space() -> gym.spaces.Space:
    return spaces.Dict(
        {
            "positions": spaces.Box(
                low=0,
                high=king_config.PATH_LENGTH - 1,
                shape=(king_config.HISTORY_LENGTH, 16),
                dtype=np.int64,
            ),
            "dice_history": spaces.Box(
                low=0, high=6, shape=(king_config.HISTORY_LENGTH,), dtype=np.int64
            ),
            "token_mask": spaces.Box(
                low=0,
                high=1,
                shape=(king_config.HISTORY_LENGTH, 16),
                dtype=np.bool_,
            ),
            "player_history": spaces.Box(
                low=0, high=3, shape=(king_config.HISTORY_LENGTH,), dtype=np.int64
            ),
            "token_colors": spaces.Box(low=0, high=3, shape=(16,), dtype=np.int64),
            "current_dice": spaces.Box(low=1, high=6, shape=(1,), dtype=np.int64),
        }
    )


class LudoEnv(gym.Env):
    """
    A Gymnasium environment for Ludo King.

    Observation Space:
        Token-sequence representation:
        - "positions": (10, 16) int, last 10 atomic moves' positions per token (0..57)
        - "dice_history": (10,) int, dice per frame (0 for pad, 1..6 actual)
        - "token_mask": (10, 16) bool, 1 if frame/token valid, else 0 for padding
        - "player_history": (10,) int, which player (0..3) made each move
        - "token_colors": (16,) int in [0..3], color id per token block
        - "current_dice": (1,) int, dice for the agent's current decision (1..6)

    Action Space:
        Discrete(4), representing the choice of which piece to move (0, 1, 2, or 3).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def action_masks(self) -> np.ndarray:
        """Helper for sb3_contrib.common.masking.ActionMasker.

        Returns cached mask if available, otherwise computes it.
        The cache is invalidated whenever game state changes (reset, step, dice roll).
        """
        if self._cached_action_mask is None:
            self._get_info()  # This populates the cache
        return self._cached_action_mask

    def __init__(
        self, render_mode: Optional[str] = None, use_fixed_opponents: bool = True
    ):
        super().__init__()
        self.agent_index = 0  # We are always Player 0
        self.render_mode = render_mode

        self.max_game_turns = king_config.MAX_TURNS
        self.current_turn = 0

        # Internal game state
        self.game: Game | None = None
        self.current_dice_roll: int = 1
        self.current_player_index: int = 0
        self.move_map: Dict[int, Move] = {}
        self.rng = random.Random()
        # Cache for action mask (recomputed only when state changes)
        self._cached_action_mask: np.ndarray | None = None

        # Opponent strategies
        self.opponents: List[str] = STRATEGY_AVAILABLE
        # 0 = random per seat, 1 = sequential cycling
        self.strategy_selection: int = STRATEGY_SELECTION
        # Track resets to advance sequential selection across episodes
        self._reset_count: int = 0

        # Action Space: Choose one of 4 pieces
        self.action_space = spaces.Discrete(king_config.PIECES_PER_PLAYER)

        # Observation Space: token sequence (last 10 atomic moves)
        self.observation_space = get_observation_space()
        self._fixed_opponents_strategies: list[str] = None
        self.use_fixed_opponents = use_fixed_opponents

        # Create the curriculum-aware opponent sampler
        # Use timesteps (not resets) for curriculum progression in multi-env training
        curriculum_timesteps = king_config.CURRICULUM_TOTAL_TIMESTEPS
        self._opponent_sampler: OpponentLineupSampler = create_default_sampler(
            strategies=self.opponents,
            curriculum_timesteps=curriculum_timesteps,
        )

    def _build_observation(self) -> Dict[str, np.ndarray]:
        assert self.game is not None
        obs = self.sim.get_token_sequence_observation(self.current_dice_roll)
        return obs

    def _roll_dice(self) -> int:
        """Roll dice and invalidate action mask cache."""
        assert self.game is not None
        self._cached_action_mask = None
        self.current_dice_roll = self.game.roll_dice()
        return self.current_dice_roll

    def _get_info(self):
        """Generates the info dict, including the crucial action mask.

        Also updates the cached action mask and move_map.
        """
        assert self.game is not None
        valid_moves = self.game.legal_moves(self.agent_index, self.current_dice_roll)
        # Use bool for the mask as recommended by Gymnasium
        action_mask = np.zeros(king_config.PIECES_PER_PLAYER, dtype=np.bool_)
        self.move_map = {}

        for move in valid_moves:
            piece_id = int(move.piece_id)
            action_mask[piece_id] = True
            if piece_id not in self.move_map:
                self.move_map[piece_id] = move

        # Update cache
        self._cached_action_mask = action_mask

        # Indicate if agent has no valid moves (e.g., rolled non-6 with all pieces in yard)
        no_valid_moves = not action_mask.any()
        return {"action_mask": action_mask, "no_valid_moves": no_valid_moves}

    def _check_game_over(self):
        """
        Checks if the agent has won (terminated)
        or if the game has hit the turn limit (truncated).
        """
        # 1. Check for termination (win condition)
        assert self.game is not None
        if king_config.RANK_ENV:
            player = self.game.players[self.agent_index]
            terminated = player.check_won()
        else:
            # Stop if ANY player wins first
            terminated = any(p.check_won() for p in self.game.players)

        # 2. Check for truncation (turn limit)
        truncated = self.current_turn >= self.max_game_turns

        return terminated, truncated

    def _get_lineup(self, num_opponents: int) -> List[str]:
        """Get opponent lineup using the curriculum-aware sampler."""
        # Respect fixed-opponents caching if enabled
        if self.use_fixed_opponents and self._fixed_opponents_strategies is not None:
            if self._reset_count % king_config.FIXED_OPPONENTS_STEPS != 0:
                return self._fixed_opponents_strategies

        # Sample lineup and log stats periodically
        lineup = self._opponent_sampler.sample_lineup(num_opponents)
        self._opponent_sampler.log_stats(interval=50_000)

        self._fixed_opponents_strategies = lineup
        return lineup

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed, options=options)

        # Re-initialize the game with 2 or 4 players based on config
        if king_config.NUM_PLAYERS == 2:
            color_ids = [int(Color.RED), int(Color.YELLOW)]
        else:
            color_ids = [
                int(Color.RED),
                int(Color.GREEN),
                int(Color.YELLOW),
                int(Color.BLUE),
            ][: king_config.NUM_PLAYERS]
        players = [Player(color=c) for c in color_ids]
        self.game = Game(players=players)
        self.current_player_index = self.agent_index
        self.sim = Simulator.for_game(self.game, agent_index=self.agent_index)

        # Attach opponents
        # Build an opponent lineup for this episode based on selection mode
        num_opponents = len(self.game.players) - 1

        lineup = self._get_lineup(num_opponents)

        opp_seat = 0
        for idx, pl in enumerate(self.game.players):
            for pc in pl.pieces:
                pc.position = 0
            pl.has_finished = False
            if idx != self.agent_index:
                strat_name = lineup[opp_seat].strip() if opp_seat < len(lineup) else ""
                opp_seat += 1
                if strat_name and strat_name in STRATEGY_REGISTRY:
                    cls = STRATEGY_REGISTRY[strat_name]
                    try:
                        pl.strategy = cls.create_instance(self.rng)
                    except NotImplementedError as e:
                        logger.warning(
                            f"Strategy {strat_name} doesn't support create_instance, using default constructor: {e}"
                        )
                        pl.strategy = cls()
                    pl.strategy_name = strat_name  # type: ignore[attr-defined]
                else:
                    # Unknown or empty strategy -> leave None; Player.choose will fallback to random
                    pl.strategy = None
                    pl.strategy_name = "random"  # type: ignore[attr-defined]

        # Reset counters and dice
        self.current_turn = 0
        self._roll_dice()

        obs = self._build_observation()
        info = self._get_info()

        # Advance reset counter for sequential selection
        self._reset_count += 1
        # Advance the curriculum sampler's counter
        self._opponent_sampler.advance()

        # NO WHILE LOOP - return immediately even if agent has no valid moves
        # The agent learns to handle no-move situations (dice didn't roll 6 to exit yard)
        # This preserves MDP structure: initial state might have no valid actions
        return obs, info

    def step(self, action: int):
        """
        Execute one step in the environment.

        MDP-Correct Design:
        - Every call returns after a single agent decision point
        - If agent has no valid moves, return immediately with all-zero mask
        - Agent learns to handle no-move situations naturally
        - No while loops that collapse multiple game states
        """
        assert self.game is not None
        reward = 0.0

        # 1) Validate action and map to a chosen move
        mv = self.move_map.get(int(action))
        if mv is None:
            # Invalid action - small penalty and pass turn
            # Do NOT reset summaries - allow accumulation when agent has no valid moves
            reward += compute_invalid_action_penalty()
            self.current_turn += 1
            self.sim.step_opponents_only(reset_summaries=False)
            # Collect any rewards accumulated during opponent turns (e.g., got_captured)
            reward += self.sim.get_agent_reward()
            self._roll_dice()
            obs = self._build_observation()
            info = self._get_info()
            terminated, truncated = self._check_game_over()
            return obs, reward, terminated, truncated, info

        # 2) Apply agent move and compute sparse rewards from events
        result = self.game.apply_move(mv)
        extra_turn = result.extra_turn and result.events.move_resolved

        # Calculate sparse rewards from move events (env owns reward calculation)
        rewards = compute_sparse_rewards(
            num_players=len(self.game.players),
            mover_index=self.agent_index,
            events=result.events,
        )
        reward += rewards.get(self.agent_index, 0.0)

        # 3) If no extra turn, opponents play until agent's turn
        if not extra_turn:
            self.current_turn += 1
            self.sim.step_opponents_only(reset_summaries=True)
            # Collect any rewards accumulated during opponent turns (e.g., got_captured)
            reward += self.sim.get_agent_reward()

        # 4) Prepare next observation
        self._roll_dice()
        obs = self._build_observation()
        info = self._get_info()

        # 5) Check for termination/truncation
        terminated, truncated = self._check_game_over()
        if terminated:
            if king_config.RANK_ENV:
                rank = sum(p.check_won() for p in self.game.players)
                reward += compute_terminal_reward(len(self.game.players), rank)
                info["final_rank"] = rank
            else:
                agent_won = self.game.players[self.agent_index].check_won()
                if agent_won:
                    reward += reward_config.win
                    info["win"] = True
                else:
                    reward += reward_config.lose
                    info["win"] = False
            return obs, reward, terminated, truncated, info

        if truncated:
            reward += compute_draw_reward()
            info["final_rank"] = 0
            info["TimeLimit.truncated"] = True
            return obs, reward, terminated, truncated, info

        # 6) NO WHILE LOOP - return immediately even if no valid moves
        # The agent will learn to handle no-move situations
        # The observation and mask tell it everything it needs to know
        # This preserves MDP structure: one step = one state transition
        if not np.any(info["action_mask"]):
            # No valid moves available - just return with zero mask
            # Agent will be forced to pick an invalid action, get penalty, and continue
            # This is correct MDP behavior - agent sees no-move state
            pass

        return obs, reward, terminated, truncated, info

    # --- Internal helpers ---

    def set_curriculum_timesteps(self, timesteps: int) -> None:
        """
        Update curriculum progress based on global training timesteps.

        Called by CurriculumSyncCallback to sync all envs to the same
        training progress, ensuring curriculum advances correctly in
        multi-env (vectorized) training.
        """
        self._opponent_sampler.set_global_timesteps(timesteps)

    def render(self):
        if self.render_mode == "rgb_array":
            if self.game is None:
                return None
            try:
                from .ludo_king.render import render_from_game  # lazy import
            except Exception as e:
                logger.warning(f"RGB render not available: {e}")
                return None
            img = render_from_game(self.game, show_ids=False)
            return np.asarray(img)

        return format_env_state(self)

    def close(self):
        pass


def format_env_state(env: "LudoEnv") -> str:
    """Returns a string snapshot of the current game state."""
    lines = [f"--- Turn {env.current_turn}/{env.max_game_turns} ---"]
    if env.game is None:
        return "\n".join(lines)
    for idx, player in enumerate(env.game.players):
        label = "AGENT (P0)" if idx == env.agent_index else f"Opponent (P{idx})"
        positions = [piece.position for piece in player.pieces]
        lines.append(f"   {label}: {positions}")
    return "\n".join(lines)
