from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger
from ludo_engine.core import LudoGame, PlayerColor
from ludo_engine.models import (
    ALL_COLORS,
    GameConstants,
    MoveResult,
    MoveType,
    ValidMove,
)

from ludo_rl.config import EnvConfig
from ludo_rl.ludo_env.observation import ObservationBuilder
from ludo_rl.utils.move_utils import MoveUtils
from ludo_rl.utils.reward_calculator import RewardCalculator


class LudoRLEnvBase(gym.Env):
    metadata = {"render_modes": ["human"], "name": "LudoRLEnvBase-v0"}

    def __init__(self, cfg: EnvConfig):
        super().__init__()
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

        self.agent_color = PlayerColor.RED
        self._episode = 0

        self.game = LudoGame(ALL_COLORS)
        self.obs_builder = ObservationBuilder(cfg, self.game, self.agent_color)
        self.action_space = spaces.Discrete(GameConstants.TOKENS_PER_PLAYER)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_builder.size,), dtype=np.float32
        )

        self._pending_dice: Optional[int] = None
        self._pending_valid: List[ValidMove] = []
        self.turns = 0
        self.illegal_actions = 0
        self._reward_calc = RewardCalculator()
        # Episode-level cumulative capture counters (agent perspective)
        self._episode_captured_opponents = (
            0  # total opponent tokens captured by agent this episode
        )
        self._episode_captured_by_opponents = (
            0  # total agent tokens captured by opponents this episode
        )
        self._captured_by_opponents = 0  # per-agent-turn counter
        # Opportunity instrumentation
        self._episode_capture_opportunities_available = 0
        self._episode_capture_opportunities_taken = 0
        self._episode_finish_opportunities_available = 0
        self._episode_finish_opportunities_taken = 0
        self._episode_home_exit_opportunities_available = 0
        self._episode_home_exit_opportunities_taken = 0

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

        colors = [c for c in ALL_COLORS if c != self.agent_color]
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
            self.agent_color = self.rng.choice(ALL_COLORS)
        self.game = LudoGame(ALL_COLORS)
        self.obs_builder = ObservationBuilder(self.cfg, self.game, self.agent_color)

        self._pending_dice = None
        self._pending_valid = []
        self.turns = 0
        self.illegal_actions = 0
        self._episode_captured_opponents = 0
        self._episode_captured_by_opponents = 0
        self._captured_by_opponents = 0  # per-agent-turn counter
        self._episode_capture_opportunities_available = 0
        self._episode_capture_opportunities_taken = 0
        self._episode_finish_opportunities_available = 0
        self._episode_finish_opportunities_taken = 0
        self._episode_home_exit_opportunities_available = 0
        self._episode_home_exit_opportunities_taken = 0

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
            and self.game.get_current_player().color != self.agent_color
        ):
            self._simulate_single_opponent()

    def _roll_agent_dice(self):
        dice = self.game.roll_dice()
        valid = self.game.get_valid_moves(self.game.get_current_player(), dice)
        return dice, valid

    def _simulate_single_opponent(self):
        p = self.game.get_current_player()
        if p.color == self.agent_color:
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
            # Track captures on the agent caused by opponents
            if res.captured_tokens:
                agent_color = self.agent_color
                for ct in res.captured_tokens:
                    if ct.player_color == agent_color:
                        self._captured_by_opponents += 1
                        self._episode_captured_by_opponents += 1
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
            extra = False
        else:
            action = int(action)
            valid_ids = [m.token_id for m in valid]
            pre_fin_ops = 0
            pre_exit_ops = 0
            if self.cfg.track_opportunities:
                # capture opportunities (each move that would capture at least one opponent)
                cap_ops = sum(1 for m in valid if m.captures_opponent)
                self._episode_capture_opportunities_available += cap_ops
                # finish opportunities: any move whose resulting position equals FINISH_POSITION
                pre_fin_ops = 0
                for m in valid:
                    tgt = m.target_position
                    if tgt == GameConstants.FINISH_POSITION:
                        pre_fin_ops += 1
                self._episode_finish_opportunities_available += pre_fin_ops
                # home exit opportunities still by move_type
                pre_exit_ops = sum(
                    1 for m in valid if m.move_type == MoveType.EXIT_HOME
                )
                self._episode_home_exit_opportunities_available += pre_exit_ops
                if self.cfg.debug_capture_logging and self.turns < 100:
                    try:
                        mt_list = [m.move_type for m in valid]
                        logger.debug(
                            f"[OppDebug] turn={self.turns} dice={dice} move_types={mt_list} fin_by_pos_avail+={pre_fin_ops} exit_avail+={pre_exit_ops} cap_avail+={cap_ops}"
                        )
                    except Exception:
                        pass
            tok_id = action
            if action not in valid_ids:
                illegal = True
                self.illegal_actions += 1
                tok_id = self.rng.choice(valid_ids)
            res = self.game.execute_move(agent, tok_id, dice)
            extra = res.extra_turn
            if self.cfg.track_opportunities and valid:
                chosen = None
                for mv in valid:
                    if mv.token_id == tok_id:
                        chosen = mv
                        break
                if chosen is not None:
                    if chosen.captures_opponent:
                        self._episode_capture_opportunities_taken += 1
                    # Determine if the executed move actually finished a token via position comparison
                    finished_flag = (
                        res.finished_token
                        or res.new_position == GameConstants.FINISH_POSITION
                    )
                    if finished_flag:
                        self._episode_finish_opportunities_taken += 1
                        if (
                            pre_fin_ops == 0
                        ):  # retroactive availability if not pre-counted
                            self._episode_finish_opportunities_available += 1
                    if chosen.move_type == MoveType.EXIT_HOME:
                        self._episode_home_exit_opportunities_taken += 1
                        if pre_exit_ops == 0:
                            self._episode_home_exit_opportunities_available += 1

        # Reset per-full-turn counters (opponent captures on agent since last agent action)
        self._captured_by_opponents = 0

        # opponent turns if no extra turn
        if not extra and not self.game.game_over:
            self.game.next_turn()
            while (
                not self.game.game_over
                and self.game.get_current_player().color != self.agent_color
            ):
                self._simulate_single_opponent()

        # Track offensive captures for agent move
        self._episode_captured_opponents += len(res.captured_tokens)
        # Optional debug logging
        if self.cfg.debug_capture_logging and res.captured_tokens:
            offensive = len(res.captured_tokens)
            defensive = self._captured_by_opponents
            logger.debug(
                f"[CaptureEvent] turn={self.turns} dice={dice} offensive={offensive} defensive_inc={defensive} cumulative_off={self._episode_captured_opponents} cumulative_def={self._episode_captured_by_opponents}"
            )

        # rewards
        terminated = res.game_won or self.game.game_over

        # Get winner for reward calculation
        winner = getattr(self.game, "winner", None) if self.game.game_over else None

        reward = self._reward_calc.compute(
            res=res,
            illegal=illegal,
            cfg=self.cfg,
            game_over=self.game.game_over,
            captured_by_opponents=int(self._captured_by_opponents),
            extra_turn=bool(extra),
            winner=winner,
            agent_color=self.agent_color,
            home_tokens=len(
                [
                    i
                    for i in self.game.get_player_from_color(
                        self.agent_color
                    ).player_positions()
                    if i == GameConstants.HOME_POSITION
                ]
            ),
        )

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
        # Compute simple extra stats for consumers (evaluation)
        captured_opponents = len(res.captured_tokens)
        captured_by_opponents = int(self._captured_by_opponents)
        agent_player = self.game.get_player_from_color(self.agent_color)
        finished_tokens = agent_player.get_finished_tokens_count()
        info = {
            "illegal_action": illegal,
            "illegal_actions_total": self.illegal_actions,
            "action_mask": MoveUtils.action_mask(self._pending_valid),
            "captured_opponents": captured_opponents,
            "captured_by_opponents": captured_by_opponents,
            # cumulative episode-level stats (useful for evaluation)
            "episode_captured_opponents": int(self._episode_captured_opponents),
            "episode_captured_by_opponents": int(self._episode_captured_by_opponents),
            "finished_tokens": finished_tokens,
            # opportunity instrumentation
            "episode_capture_ops_available": int(
                self._episode_capture_opportunities_available
            ),
            "episode_capture_ops_taken": int(self._episode_capture_opportunities_taken),
            "episode_finish_ops_available": int(
                self._episode_finish_opportunities_available
            ),
            "episode_finish_ops_taken": int(self._episode_finish_opportunities_taken),
            "episode_home_exit_ops_available": int(
                self._episode_home_exit_opportunities_available
            ),
            "episode_home_exit_ops_taken": int(
                self._episode_home_exit_opportunities_taken
            ),
        }
        return obs, reward, terminated, truncated, info
