from __future__ import annotations

import random
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ludo_engine.core import LudoGame, PlayerColor
from ludo_engine.models import Colors, GameConstants, MoveResult, ValidMove
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from ludo_rl.config import EnvConfig
from ludo_rl.ludo_env.observation import ObservationBuilder
from ludo_rl.strategies.frozen_policy_strategy import FrozenPolicyStrategy
from ludo_rl.utils.move_utils import MoveUtils


class LudoRLEnvSelfPlay(gym.Env):
    """Self-play environment using a frozen copy of the policy as opponents.

    - At reset, the agent color is selected (optionally randomized).
    - The learning agent controls only that color; other 3 colors use a frozen model.
    - A callback can periodically provide a new frozen model path and obs_rms stats.
    """

    metadata = {"render_modes": ["human"], "name": "LudoRLEnvSelfPlay-v0"}

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
        self._pending_valid: List[ValidMove] = []
        self.turns = 0
        self.illegal_actions = 0

        # Live training model and frozen snapshot used by opponents
        self.model: MaskablePPO = None
        self._frozen_policy: MaskableActorCriticPolicy = None
        self._opponent_builders: Dict[str, ObservationBuilder] = {}

    # ---- Model snapshot management (in-memory) ----
    def set_model(self, model: MaskablePPO) -> None:
        """Inject the live model; env will snapshot its policy on reset for opponents."""
        self.model = model

    def _snapshot_policy(self) -> None:
        try:
            self._frozen_policy = getattr(self.model, "policy", None)
        except Exception:
            self._frozen_policy = None

    # ---- gym api ----
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
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
        # per-opponent builders (perspectives)
        self._opponent_builders = {}
        for c in Colors.ALL_COLORS:
            if c != self.agent_color:
                self._opponent_builders[c] = ObservationBuilder(self.cfg, self.game, c)

        # Snapshot current policy for this episode (used by opponents)
        self._snapshot_policy()
        # Attach strategies for opponents so simulation matches classic env
        for p in self.game.players:
            if p.color.value != self.agent_color:
                ob = self._opponent_builders[p.color.value]
                strat = FrozenPolicyStrategy(
                    policy=self._frozen_policy,
                    obs_builder=ob,
                    deterministic=True,
                )
                try:
                    p.set_strategy(strat)
                except Exception:
                    pass

        self._pending_dice = None
        self._pending_valid = []
        self.turns = 0
        self.illegal_actions = 0

        # advance to agent turn and roll dice
        self._ensure_agent_turn()
        self._pending_dice, self._pending_valid = self._roll_agent_dice()
        obs = self.obs_builder.build(self.turns, self._pending_dice)

        info = {
            "episode": self._episode,
        }
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
