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
from .ludo_king.simulator import Simulator
from .ludo_king.types import Color
from .strategy.registry import STRATEGY_REGISTRY
from .strategy.registry import available as available_strategies


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

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def action_masks(self):
        # Helper for sb3_contrib.common.masking.ActionMasker
        return self._get_info()["action_mask"]

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
        self.move_map: Dict[int, object] = {}
        self.rng = random.Random()

        # Opponent strategies
        self.opponents: List[str] = [
            s
            for s in os.getenv("OPPONENTS", ",".join(available_strategies())).split(",")
            if s
        ]
        logger.info(
            f"Configured opponents ({king_config.NUM_PLAYERS} players): {self.opponents}"
        )
        # 0 = random per seat, 1 = sequential cycling
        try:
            self.strategy_selection: int = int(
                os.getenv("STRATEGY_SELECTION", "0") or "0"
            )
        except ValueError as e:
            logger.warning(f"Invalid STRATEGY_SELECTION value, defaulting to 0: {e}")
            self.strategy_selection = 0
        logger.info(f"Strategy selection mode: {self.strategy_selection}")
        # Track resets to advance sequential selection across episodes
        self._reset_count: int = 0

        # Action Space: Choose one of 4 pieces
        self.action_space = spaces.Discrete(king_config.PIECES_PER_PLAYER)

        # Observation Space: token sequence (last 10 atomic moves)
        self.observation_space = spaces.Dict(
            {
                "positions": spaces.Box(
                    low=0,
                    high=king_config.PATH_LENGTH - 1,
                    shape=(10, 16),
                    dtype=np.int64,
                ),
                "dice_history": spaces.Box(low=0, high=6, shape=(10,), dtype=np.int64),
                "token_mask": spaces.Box(low=0, high=1, shape=(10, 16), dtype=np.bool_),
                "player_history": spaces.Box(
                    low=0, high=3, shape=(10,), dtype=np.int64
                ),
                "token_colors": spaces.Box(low=0, high=3, shape=(16,), dtype=np.int64),
                "current_dice": spaces.Box(low=1, high=6, shape=(1,), dtype=np.int64),
            }
        )
        self._fixed_opponents_strategies: list[str] = None
        self.use_fixed_opponents = use_fixed_opponents

    def _build_observation(self) -> Dict[str, np.ndarray]:
        assert self.game is not None
        obs = self.sim.get_token_sequence_observation(self.current_dice_roll)
        return obs

    def _get_info(self):
        """Generates the info dict, including the crucial action mask."""
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

        return {"action_mask": action_mask}

    def _check_game_over(self):
        """
        Checks if the agent has won (terminated)
        or if the game has hit the turn limit (truncated).
        """
        # 1. Check for termination (win condition)
        assert self.game is not None
        player = self.game.players[self.agent_index]
        terminated = player.check_won()

        # 2. Check for truncation (turn limit)
        truncated = self.current_turn >= self.max_game_turns

        return terminated, truncated

    def _get_lineup(self, num_opponents: int) -> List[str]:
        if self.use_fixed_opponents and self._fixed_opponents_strategies is not None:
            if self._reset_count % king_config.FIXED_OPPONENTS_STEPS != 0:
                return self._fixed_opponents_strategies
        if self.strategy_selection == 0:
            # Simple: pick each opponent independently at random (with replacement)
            lineup = [self.rng.choice(self.opponents) for _ in range(num_opponents)]
        else:
            # Sequential cycling through provided opponents across episodes
            start = (self._reset_count * num_opponents) % max(1, len(self.opponents))
            lineup = [
                self.opponents[(start + i) % len(self.opponents)]
                for i in range(num_opponents)
            ]
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
        self.current_dice_roll = self.game.roll_dice()

        obs = self._build_observation()
        info = self._get_info()

        # Handle no valid moves for agent on first turn: opponents play until agent has a move
        # Don't reset summaries - accumulate activity from the start
        while not np.any(info["action_mask"]):
            self.current_turn += 1
            self.sim.step_opponents_only(reset_summaries=False)
            self.current_dice_roll = self.game.roll_dice()
            obs = self._build_observation()
            info = self._get_info()
            if self.current_turn >= self.max_game_turns:
                return obs, info

        # Advance reset counter for sequential selection
        self._reset_count += 1
        return obs, info

    def step(self, action: int):
        assert self.game is not None
        reward = 0.0

        # 1) Validate action and map to a chosen move
        mv = self.move_map.get(int(action))
        if mv is None:
            # Invalid action - penalize and pass turn to opponents
            reward += reward_config.lose
            self.current_turn += 1
            self.sim.step_opponents_only()
            self.current_dice_roll = self.game.roll_dice()
            obs = self._build_observation()
            info = self._get_info()
            terminated, truncated = self._check_game_over()
            return obs, reward, terminated, truncated, info

        # 2) Apply agent move
        result = self.game.apply_move(mv)
        extra_turn = result.extra_turn and result.events.move_resolved

        # Add rewards based on move result
        if result.rewards is not None:
            reward += float(result.rewards.get(self.agent_index, 0.0))
        else:
            # Rewards is None when move failed (e.g., hit blockade)
            # Provide explicit feedback for these events
            if result.events.finished:
                # Finishing a piece
                reward += reward_config.finish
            if result.events.knockouts:
                # Captured opponent pieces
                reward += reward_config.capture * len(result.events.knockouts)
            if result.events.hit_blockade and not result.events.move_resolved:
                # Hit a blockade - move failed
                reward += reward_config.hit_blockade
            if result.events.blockades:
                # Formed a blockade
                reward += reward_config.blockade

        # 3) If no extra turn, opponents play until agent's turn
        # Reset summaries here since this is the agent's turn
        if not extra_turn:
            self.current_turn += 1
            self.sim.step_opponents_only(reset_summaries=True)

        # 4) Prepare next observation
        self.current_dice_roll = self.game.roll_dice()
        obs = self._build_observation()
        info = self._get_info()

        # 5) Check for termination/truncation
        terminated, truncated = self._check_game_over()
        if terminated:
            # Rank: number of players who have won at this point
            rank = sum(p.check_won() for p in self.game.players)
            if rank == 1:
                reward += reward_config.win
            else:
                reward += (
                    reward_config.lose
                    * (len(self.game.players) - rank + 1)
                    / len(self.game.players)
                )
            info["final_rank"] = rank
            return obs, reward, terminated, truncated, info

        if truncated:
            reward += reward_config.draw
            info["final_rank"] = 0
            info["TimeLimit.truncated"] = True
            return obs, reward, terminated, truncated, info

        # 6) If agent has no valid moves for next turn, simulate opponents until it does
        # Don't reset summaries here - accumulate all activity between agent turns
        while not np.any(info["action_mask"]) and not terminated and not truncated:
            reward += reward_config.skipped_turn
            self.current_turn += 1
            if self.current_turn >= self.max_game_turns:
                truncated = True
                break
            self.sim.step_opponents_only(reset_summaries=False)
            self.current_dice_roll = self.game.roll_dice()
            obs = self._build_observation()
            info = self._get_info()
            terminated, truncated = self._check_game_over()

        if truncated:
            reward += reward_config.draw
            info["final_rank"] = 0
            info["TimeLimit.truncated"] = True
            return obs, reward, terminated, truncated, info

        return obs, reward, terminated, truncated, info

    # --- Internal helpers ---

    def render(self):
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
