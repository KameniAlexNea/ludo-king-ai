"""Minimal gymnasium environment for classic Ludo training."""

from __future__ import annotations

import random
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ludo_engine.core import LudoGame, Player
from ludo_engine.models import (
    ALL_COLORS,
    GameConstants,
    MoveResult,
    PlayerColor,
    ValidMove,
)
from ludo_engine.strategies.strategy import StrategyFactory

from .config import EnvConfig
from .observation import make_observation_builder
from .reward import AdvancedRewardCalculator


def _make_mask(valid_moves: Optional[list[ValidMove]]) -> np.ndarray:
    mask = np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=bool)
    if valid_moves:
        for move in valid_moves:
            mask[move.token_id] = True
    return mask


class LudoRLEnv(gym.Env):
    """Single-process classic Ludo environment aimed at PPO training."""

    metadata = {"render_modes": ["human"], "name": "LudoRLEnvMinimal-v0"}

    def __init__(self, cfg: Optional[EnvConfig] = None):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.rng = random.Random(self.cfg.seed)

        tokens = GameConstants.TOKENS_PER_PLAYER
        opponents = len(ALL_COLORS) - 1
        token_total_max = float(tokens)
        opponent_total_max = float(tokens * opponents)

        self.observation_space = spaces.Dict(
            {
                "agent_color": spaces.Box(
                    low=0.0, high=1.0, shape=(len(ALL_COLORS),), dtype=np.float32
                ),
                "agent_progress": spaces.Box(
                    low=0.0, high=1.0, shape=(tokens,), dtype=np.float32
                ),
                "agent_distance_to_finish": spaces.Box(
                    low=0.0, high=1.0, shape=(tokens,), dtype=np.float32
                ),
                "agent_vulnerable": spaces.Box(
                    low=0.0, high=1.0, shape=(tokens,), dtype=np.float32
                ),
                "agent_safe": spaces.Box(
                    low=0.0, high=1.0, shape=(tokens,), dtype=np.float32
                ),
                "agent_home": spaces.Box(
                    low=0.0, high=1.0, shape=(tokens,), dtype=np.float32
                ),
                "agent_on_board": spaces.Box(
                    low=0.0, high=1.0, shape=(tokens,), dtype=np.float32
                ),
                "agent_capture_available": spaces.Box(
                    low=0.0, high=1.0, shape=(tokens,), dtype=np.float32
                ),
                "agent_finish_available": spaces.Box(
                    low=0.0, high=1.0, shape=(tokens,), dtype=np.float32
                ),
                "agent_threat_level": spaces.Box(
                    low=0.0, high=1.0, shape=(tokens,), dtype=np.float32
                ),
                "agent_tokens_at_home": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "agent_tokens_finished": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "agent_tokens_on_safe": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "agent_total_progress": spaces.Box(
                    low=0.0, high=token_total_max, shape=(1,), dtype=np.float32
                ),
                "opponents_positions": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(tokens * opponents,),
                    dtype=np.float32,
                ),
                "opponents_active": spaces.Box(
                    low=0.0, high=1.0, shape=(opponents,), dtype=np.float32
                ),
                "opponent_total_progress": spaces.Box(
                    low=0.0,
                    high=opponent_total_max,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "opponent_best_progress": spaces.Box(
                    low=0.0, high=token_total_max, shape=(1,), dtype=np.float32
                ),
                "opponent_tokens_at_home": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "opponent_tokens_finished": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "opponent_tokens_on_safe": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "progress_lead": spaces.Box(
                    low=-token_total_max,
                    high=token_total_max,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "agent_rank": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "dice": spaces.Box(
                    low=0.0, high=1.0, shape=(GameConstants.DICE_MAX,), dtype=np.float32
                ),
                "dice_value_norm": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "dice_is_six": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "dice_is_even": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "home_exit_ready": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "capture_any": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "finish_any": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
            }
        )

        self.action_space = spaces.Discrete(tokens)

        self.game: Optional[LudoGame] = None
        self.agent_color: PlayerColor | None = None
        self._obs_builder = None
        self.pending_dice: int = 0
        self.pending_valid_moves: list[ValidMove] = []
        self.action_mask = np.zeros(tokens, dtype=bool)
        self.turn_count = 0
        self._opponent_captures = 0
        self.reward_calc = AdvancedRewardCalculator()

        self.opponent_names = self.cfg.opponent_strategy.split(",")

    # gym API ---------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
        elif self.cfg.seed is not None:
            self.rng.seed(self.cfg.seed)

        self.turn_count = 0
        self._opponent_captures = 0

        self._create_game()
        self.reward_calc.reset_for_new_episode()
        self._ensure_agent_turn()
        self._roll_agent_dice()

        observation = self._current_observation()
        info = {"action_mask": self.action_mask.copy()}
        return observation, info

    def step(self, action: int):
        if self.game is None:
            raise RuntimeError("Environment is not initialised. Call reset() first.")

        terminated = False
        truncated = False
        is_illegal = False

        if not self.pending_valid_moves:
            move_result = self._no_move_result()
        else:
            valid_tokens = {m.token_id for m in self.pending_valid_moves}
            chosen = int(action)
            if chosen not in valid_tokens:
                is_illegal = True
                chosen = self.rng.choice(list(valid_tokens))
            agent_player = self.game.get_current_player()
            move_result = self.game.execute_move(
                agent_player, chosen, self.pending_dice
            )

        self.turn_count += 1
        terminated = self.game.game_over or (self.game.winner is not None)

        if not terminated and move_result.extra_turn:
            self._roll_agent_dice()
        elif not terminated:
            self._play_opponents_until_agent()
            terminated = self.game.game_over or (self.game.winner is not None)
            if not terminated:
                self._roll_agent_dice()

        truncated = not terminated and self.turn_count >= self.cfg.max_turns

        reward, breakdown = self.reward_calc.compute(
            game=self.game,
            agent_color=self.agent_color,
            move_result=move_result,
            cfg=self.cfg,
            is_illegal=is_illegal,
            opponent_captures=self._opponent_captures,
            terminated=terminated,
        )

        observation = self._current_observation()
        info = {
            "action_mask": self.action_mask.copy(),
            "opponent_captures": self._opponent_captures,
            "reward_breakdown": breakdown,
            "illegal_action": is_illegal,
        }
        self._opponent_captures = 0

        return observation, reward, bool(terminated), bool(truncated), info

    # internal helpers ------------------------------------------------------
    def _create_game(self) -> None:
        colors = list(ALL_COLORS)
        if not self.cfg.randomize_agent and self.cfg.fixed_agent_color:
            requested = self.cfg.fixed_agent_color.upper()
            try:
                self.agent_color = next(
                    color for color in colors if color.name == requested
                )
            except StopIteration as exc:  # pragma: no cover - guard clause
                raise ValueError(
                    f"Unknown agent color '{self.cfg.fixed_agent_color}'."
                ) from exc
        elif self.cfg.randomize_agent:
            self.agent_color = self.rng.choice(colors)
        else:
            self.agent_color = colors[0]

        pivot = colors.index(self.agent_color)
        ordered = colors[pivot:] + colors[:pivot]
        self.game = LudoGame(ordered)

        for player in self.game.players:
            if player.color != self.agent_color:
                strategy = StrategyFactory.create_strategy(
                    random.choice(self.opponent_names)
                )
                player.set_strategy(strategy)

        self._obs_builder = make_observation_builder(
            self.cfg, self.game, self.agent_color
        )

    def _ensure_agent_turn(self) -> None:
        if self.game is None:
            return
        while (
            not self.game.game_over
            and self.game.get_current_player().color != self.agent_color
        ):
            self._play_opponent_turn(self.game.get_current_player())

    def _play_opponents_until_agent(self) -> None:
        if self.game is None:
            return
        self.game.next_turn()
        while (
            not self.game.game_over
            and self.game.get_current_player().color != self.agent_color
        ):
            self._play_opponent_turn(self.game.get_current_player())

    def _play_opponent_turn(self, player: Player) -> None:
        if self.game is None:
            return
        if player.color == self.agent_color:
            return

        dice = self.game.roll_dice()
        valid_moves = self.game.get_valid_moves(player, dice)
        if valid_moves:
            decision_context = self.game.get_ai_decision_context(dice)
            chosen_token = player.make_strategic_decision(decision_context)
            result = self.game.execute_move(player, chosen_token, dice)
            captured = sum(
                1
                for token in result.captured_tokens
                if token.player_color == self.agent_color
            )
            self._opponent_captures += captured
            if result.extra_turn and not self.game.game_over:
                return
        self.game.next_turn()

    def _roll_agent_dice(self) -> None:
        if self.game is None:
            return
        agent = self.game.get_current_player()
        if agent.color != self.agent_color:
            raise RuntimeError("Internal error: not the agent's turn.")
        self.pending_dice = self.game.roll_dice()
        self.pending_valid_moves = self.game.get_valid_moves(agent, self.pending_dice)
        self.action_mask = _make_mask(self.pending_valid_moves)

    def _current_observation(self) -> Dict[str, np.ndarray]:
        dice_val = self.pending_dice if self.pending_dice else 0
        return self._obs_builder.build(dice_val)

    def _no_move_result(self) -> MoveResult:
        if self.game is None:
            raise RuntimeError("Game state missing.")
        agent = self.game.get_current_player()
        return MoveResult(
            success=True,
            player_color=agent.color,
            token_id=0,
            dice_value=self.pending_dice,
            old_position=-1,
            new_position=-1,
            captured_tokens=[],
            finished_token=False,
            extra_turn=False,
            error=None,
            game_won=False,
        )

    # compatibility with sb3 ActionMasker ---------------------------------
    def valid_action_mask(self) -> np.ndarray:
        """Return copy of the latest action mask."""
        return self.action_mask.copy()
