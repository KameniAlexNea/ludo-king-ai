from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ludo_engine.core import LudoGame, PlayerColor
from ludo_engine.models import Colors, GameConstants, MoveResult
from ludo_engine.strategies.strategy import StrategyFactory

from ludo_rl.config import EnvConfig
from ludo_rl.ludo_env.observation import ObservationBuilder
from ludo_rl.utils.move_utils import MoveUtils


class LudoRLEnv(gym.Env):
    metadata = {"render_modes": ["human"], "name": "LudoRLEnv-v0"}

    def __init__(self, cfg: EnvConfig):
        super().__init__()
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

        self.agent_color = Colors.RED
        self._episode = 0
        self._progress: Optional[float] = None

        self.game = LudoGame([PlayerColor.RED, PlayerColor.GREEN, PlayerColor.YELLOW, PlayerColor.BLUE])
        self.obs_builder = ObservationBuilder(cfg, self.game, self.agent_color)
        self.action_space = spaces.Discrete(GameConstants.TOKENS_PER_PLAYER)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_builder.size,), dtype=np.float32)

        self._pending_dice: Optional[int] = None
        self._pending_valid: List = []
        self.turns = 0
        self.illegal_actions = 0

    # ---- opponent sampling ----
    def _sample_opponents(self) -> List[str]:
        """Sample 3 opponent strategies using a single, simple weighted scheme.

        - Every strategy has a base weight (derived from the benchmark order).
        - A level multiplier (by training progress) boosts or dampens categories.
        - We always compute weights for ALL available candidates and sample 3
          without replacement proportional to those weights.
        """
        candidates = list(self.cfg.opponents.candidates)
        if len(candidates) <= 3:
            return candidates

        # Base weights from benchmark ranking (stronger => higher base)
        base_w = {
            "probabilistic_v2": 12.0,
            "probabilistic_v3": 11.0,
            "probabilistic": 10.0,
            "hybrid_prob": 9.0,
            "killer": 8.0,
            "cautious": 7.0,
            "defensive": 6.0,
            "balanced": 5.0,
            "winner": 4.0,
            "optimist": 3.0,
            "random": 2.0,
            "weighted_random": 2.0,
        }

        # Category for level multipliers
        easy = {"random", "weighted_random", "optimist"}
        medium = {"winner", "defensive", "balanced"}
        hard = {"cautious", "killer", "hybrid_prob", "probabilistic"}
        elite = {"probabilistic_v2", "probabilistic_v3"}

        # Progress-based multipliers (simple and monotonic)
        p = 0.0 if self._progress is None else float(self._progress)
        b = self.cfg.curriculum.boundaries
        if p < b[0]:
            mult = {"easy": 1.5, "medium": 1.0, "hard": 0.6, "elite": 0.4}
        elif p < b[1]:
            mult = {"easy": 1.2, "medium": 1.1, "hard": 0.9, "elite": 0.6}
        elif p < b[2]:
            mult = {"easy": 0.8, "medium": 1.0, "hard": 1.2, "elite": 1.4}
        else:
            mult = {"easy": 0.5, "medium": 0.9, "hard": 1.3, "elite": 1.6}

        def cat(name: str) -> str:
            if name in easy: return "easy"
            if name in medium: return "medium"
            if name in hard: return "hard"
            if name in elite: return "elite"
            return "medium"

        # Compute final weights for all available candidates
        weights: List[float] = []
        for s in candidates:
            w0 = base_w.get(s, 1.0)
            w = w0 * mult.get(cat(s), 1.0)
            weights.append(max(1e-6, float(w)))

        # Weighted sample 3 without replacement
        chosen: List[str] = []
        cand = candidates[:]
        wts = weights[:]
        for _ in range(3):
            total = sum(wts)
            r = self.rng.random() * total
            cum = 0.0
            idx = 0
            for i, w in enumerate(wts):
                cum += w
                if r <= cum:
                    idx = i
                    break
            chosen.append(cand.pop(idx))
            wts.pop(idx)
        return chosen

    def _attach_strategies(self, strategies: List[str]):
        colors = [c for c in Colors.ALL_COLORS if c != self.agent_color]
        for name, color in zip(strategies, colors):
            player = self.game.get_player_from_color(color)
            try:
                player.set_strategy(StrategyFactory.create_strategy(name))
            except Exception:
                pass

    # ---- gym api ----
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng.seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        if self.cfg.randomize_agent:
            self.agent_color = self.rng.choice(list(Colors.ALL_COLORS))
        self.game = LudoGame([PlayerColor.RED, PlayerColor.GREEN, PlayerColor.YELLOW, PlayerColor.BLUE])
        self.obs_builder = ObservationBuilder(self.cfg, self.game, self.agent_color)

        self._pending_dice = None
        self._pending_valid = []
        self.turns = 0
        self.illegal_actions = 0

        # attach strategies per episode
        self._attach_strategies(self._sample_opponents())

        # advance to agent turn and roll dice
        self._ensure_agent_turn()
        self._pending_dice, self._pending_valid = self._roll_agent_dice()
        obs = self.obs_builder.build(self.turns, self._pending_dice)

        info = {
            "episode": self._episode,
            "progress": self._progress,
        }
        self._episode += 1
        return obs, info

    def _ensure_agent_turn(self):
        while not self.game.game_over and self.game.get_current_player().color.value != self.agent_color:
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
            res = MoveResult(success=True, player_color=agent.color.value, token_id=0, dice_value=dice,
                             old_position=-1, new_position=-1, captured_tokens=[], finished_token=False,
                             extra_turn=False, error=None, game_won=False)
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
            while not self.game.game_over and self.game.get_current_player().color.value != self.agent_color:
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

    # called by callback to set progress 0..1
    def set_training_progress(self, p: float):
        self._progress = max(0.0, min(1.0, float(p)))
