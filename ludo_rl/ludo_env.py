import os
import random
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .ludo_king.config import config as king_config
from .ludo_king.enums import Color
from .ludo_king.game import Game
from .ludo_king.player import Player
from .ludo_king.reward import reward_config
from .ludo_king.simulator import Simulator
from .strategy.registry import STRATEGY_REGISTRY
from .strategy.registry import available as available_strategies
from loguru import logger

class LudoEnv(gym.Env):
    """
    A Gymnasium environment for Ludo King.

    Observation Space:
        A dictionary with:
        - "board": (10, 58) Box, representing 10 stacked channels.
        - "dice_roll": (1,) Box, representing the dice roll (0-5).

    Action Space:
        Discrete(4), representing the choice of which piece to move (0, 1, 2, or 3).
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def action_masks(self):
        # Helper for sb3_contrib.common.masking.ActionMasker
        return self._get_info()["action_mask"]

    def __init__(self, render_mode: Optional[str] = None):
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
        logger.info(f"Configured opponents ({king_config.NUM_PLAYERS} players): {self.opponents}")
        # 0 = random per seat, 1 = sequential cycling
        try:
            self.strategy_selection: int = int(
                os.getenv("STRATEGY_SELECTION", "0") or "0"
            )
        except ValueError:
            self.strategy_selection = 0
        logger.info(f"Strategy selection mode: {self.strategy_selection}")
        # Track resets to advance sequential selection across episodes
        self._reset_count: int = 0

        # Action Space: Choose one of 4 pieces
        self.action_space = spaces.Discrete(king_config.PIECES_PER_PLAYER)

        # Observation Space: 10 channels x PATH_LENGTH, dice as 0..5
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=-50.0,
                    high=50.0,
                    shape=(10, king_config.PATH_LENGTH),
                    dtype=np.float32,
                ),
                "dice_roll": spaces.Box(low=0, high=5, shape=(1,), dtype=np.int64),
            }
        )

    def _build_observation(self) -> Dict[str, np.ndarray]:
        assert self.game is not None
        agent_color = int(self.game.players[self.agent_index].color)
        board_stack = self.game.board.build_tensor(agent_color).astype(
            np.float32, copy=False
        )
        dice_val = np.array([self.current_dice_roll - 1], dtype=np.int64)
        return {"board": board_stack, "dice_roll": dice_val}

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
        if num_opponents > 0:
            if self.strategy_selection == 0:
                # Simple: pick each opponent independently at random (with replacement)
                lineup = [self.rng.choice(self.opponents) for _ in range(num_opponents)]
            else:
                # Sequential cycling through provided opponents across episodes
                start = (self._reset_count * num_opponents) % max(
                    1, len(self.opponents)
                )
                lineup = [
                    self.opponents[(start + i) % len(self.opponents)]
                    for i in range(num_opponents)
                ]
        else:
            raise ValueError(
                "LudoEnv requires at least 2 players (1 agent + 1 opponent)"
            )

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
                    except NotImplementedError:
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
        while not np.any(info["action_mask"]):
            self.current_turn += 1
            self.sim.step_opponents_only()
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

        # Simple shaping: reward for finish/capture/extra turn
        if result.events.finished:
            reward += reward_config.win
        if result.events.knockouts:
            reward += (
                reward_config.capture if hasattr(reward_config, "capture") else 0.0
            )

        # 3) If no extra turn, opponents play until agent's turn
        if not extra_turn:
            self.current_turn += 1
            self.sim.step_opponents_only()

        # 4) Prepare next observation
        self.current_dice_roll = self.game.roll_dice()
        obs = self._build_observation()
        info = self._get_info()

        # 5) Check for termination/truncation
        terminated, truncated = self._check_game_over()
        if terminated:
            # Rank: number of players who have won at this point
            rank = sum(p.check_won() for p in self.game.players)
            if rank == len(self.game.players):
                reward += reward_config.lose
            else:
                reward += (
                    reward_config.win
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
        while not np.any(info["action_mask"]) and not terminated and not truncated:
            reward += reward_config.skipped_turn
            self.current_turn += 1
            if self.current_turn >= self.max_game_turns:
                truncated = True
                break
            self.sim.step_opponents_only()
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
