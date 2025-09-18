from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ludo_engine.core import LudoGame, PlayerColor
from ludo_engine.models import Colors, GameConstants, MoveResult

from ludo_rl.config import EnvConfig
from ludo_rl.ludo_env.observation import ObservationBuilder
from ludo_rl.utils.move_utils import MoveUtils


class LudoRLEnvBase(gym.Env):
    metadata = {"render_modes": ["human"], "name": "LudoRLEnvBase-v0"}

    def __init__(self, cfg: EnvConfig):
        super().__init__()
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

        self.agent_color = Colors.RED
        self._episode = 0

        self.game = LudoGame(
            [PlayerColor.RED, PlayerColor.GREEN, PlayerColor.YELLOW, PlayerColor.BLUE]
        )
        self.obs_builder = ObservationBuilder(cfg, self.game, self.agent_color)
        self.action_space = spaces.Discrete(GameConstants.TOKENS_PER_PLAYER)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_builder.size,), dtype=np.float32
        )

        self._pending_dice: Optional[int] = None
        self._pending_valid: List = []
        self.turns = 0
        self.illegal_actions = 0

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
        try:
            from ludo_engine.strategies.base import Strategy  # type: ignore
            from ludo_engine.strategies.strategy import StrategyFactory  # type: ignore
        except Exception:
            Strategy = object  # type: ignore
            StrategyFactory = None  # type: ignore

        colors = [c for c in Colors.ALL_COLORS if c != self.agent_color]
        for strat, color in zip(strategies, colors):
            player = self.game.get_player_from_color(color)
            try:
                if Strategy is not object and isinstance(strat, Strategy):
                    player.set_strategy(strat)
                elif StrategyFactory is not None:
                    player.set_strategy(StrategyFactory.create_strategy(strat))
            except Exception:
                pass

    # ---- gym api --------------------------------------------------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        if seed is not None:
            self.rng.seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        if self.cfg.randomize_agent:
            self.agent_color = self.rng.choice(list(Colors.ALL_COLORS))
        self.game = LudoGame(
            [PlayerColor.RED, PlayerColor.GREEN, PlayerColor.YELLOW, PlayerColor.BLUE]
        )
        self.obs_builder = ObservationBuilder(self.cfg, self.game, self.agent_color)

        self._pending_dice = None
        self._pending_valid = []
        self.turns = 0
        self.illegal_actions = 0

        # subclass-specific setup and attach opponent strategies
        self.on_reset_before_attach(options)
        self.attach_opponents(options)

        # advance to agent turn and roll dice
        self._ensure_agent_turn()
        self._pending_dice, self._pending_valid = self._roll_agent_dice()
        obs = self.obs_builder.build(self.turns, self._pending_dice)

        info = {"episode": self._episode}
        info.update(self.extra_reset_info())
        self._episode += 1
        return obs, info

    def _ensure_agent_turn(self):
        while (
            not self.game.game_over
            and self.game.get_current_player().color.value != self.agent_color
        ):
            self._simulate_single_opponent()

    def _roll_agent_dice(self):
        dice = self.game.roll_dice()
        valid = self.game.get_valid_moves(self.game.get_current_player(), dice)
        return dice, valid

    def _simulate_single_opponent(self):
        p = self.game.get_current_player()
        if p.color.value == self.agent_color:
            return
        dice = self.game.roll_dice()
        valid = self.game.get_valid_moves(p, dice)
        if valid:
            try:
                ctx = self.game.get_ai_decision_context(dice)
                token_id = p.make_strategic_decision(ctx)
            except Exception:
                token_id = valid[0].token_id
            res = self.game.execute_move(p, token_id, dice)
            if not res.extra_turn:
                self.game.next_turn()
        else:
            self.game.next_turn()

    def step(self, action: int):
        if self.game.game_over:
            obs = self.obs_builder.build(self.turns, 0)
            return obs, 0.0, True, False, {}

        illegal = False
        agent = self.game.get_current_player()
        if self._pending_dice is None:
            self._pending_dice, self._pending_valid = self._roll_agent_dice()
        dice = self._pending_dice
        valid = self._pending_valid

        if not valid:
            # no moves, lose turn
            res = MoveResult(
                success=True,
                player_color=agent.color.value,
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
            extra = False
        else:
            action = int(action)
            valid_ids = [m.token_id for m in valid]
            tok_id = action
            if action not in valid_ids:
                illegal = True
                self.illegal_actions += 1
                tok_id = self.rng.choice(valid_ids)
            res = self.game.execute_move(agent, tok_id, dice)
            extra = res.extra_turn

        # opponent turns if no extra turn
        if not extra and not self.game.game_over:
            self.game.next_turn()
            while (
                not self.game.game_over
                and self.game.get_current_player().color.value != self.agent_color
            ):
                self._simulate_single_opponent()

        # rewards
        reward = 0.0
        if res.captured_tokens:
            reward += self.cfg.reward.capture * len(res.captured_tokens)
        if res.finished_token:
            reward += self.cfg.reward.finish_token
        if illegal:
            reward += self.cfg.reward.illegal_action
        reward += self.cfg.reward.time_penalty

        terminated = False
        if res.game_won:
            reward += self.cfg.reward.win
            terminated = True
        elif self.game.game_over:
            reward += self.cfg.reward.lose
            terminated = True

        self.turns += 1
        truncated = False
        if self.turns >= self.cfg.max_turns and not terminated:
            truncated = True

        # prepare next dice
        if not terminated and not truncated and not self.game.game_over:
            if extra:
                self._pending_dice, self._pending_valid = self._roll_agent_dice()
            else:
                self._ensure_agent_turn()
                if not self.game.game_over:
                    self._pending_dice, self._pending_valid = self._roll_agent_dice()

        obs = self.obs_builder.build(self.turns, self._pending_dice or 0)
        info = {
            "illegal_action": illegal,
            "illegal_actions_total": self.illegal_actions,
            "action_mask": MoveUtils.action_mask(self._pending_valid),
        }
        return obs, reward, terminated, truncated, info
