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

        self.game = LudoGame([PlayerColor.RED, PlayerColor.GREEN, PlayerColor.YELLOW, PlayerColor.BLUE])
        self.obs_builder = ObservationBuilder(cfg, self.game, self.agent_color)
        self.action_space = spaces.Discrete(GameConstants.TOKENS_PER_PLAYER)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_builder.size,), dtype=np.float32)

        self._pending_dice: Optional[int] = None
        self._pending_valid: List = []
        self.turns = 0
        self.illegal_actions = 0

        # Opponent policy: loaded via load_frozen_model
        self._frozen_model = None
        self._frozen_obs_stats = None  # expects object with .mean, .var, and epsilon 1e-8
        self._opponent_builders: Dict[str, ObservationBuilder] = {}

    # ---- Frozen model management ----
    def load_frozen_model(self, path: str, obs_rms: Optional[Any] = None) -> bool:
        """Load a frozen model from disk to be used as opponent policy.

        path: path to a saved SB3 model (MaskablePPO).
        obs_rms: optional RunningMeanStd stats from VecNormalize for observation normalization.
        """
        try:
            from sb3_contrib import MaskablePPO

            self._frozen_model = MaskablePPO.load(path, device="cpu")
            self._frozen_obs_stats = getattr(obs_rms, "__dict__", obs_rms) or obs_rms
            return True
        except Exception:
            self._frozen_model = None
            self._frozen_obs_stats = None
            return False

    def _normalize_for_frozen(self, obs: np.ndarray) -> np.ndarray:
        stats = self._frozen_obs_stats
        if stats is None:
            return obs
        try:
            mean = getattr(stats, "mean", None)
            var = getattr(stats, "var", None)
            eps = getattr(stats, "epsilon", 1e-8) or 1e-8
            if mean is None or var is None:
                return obs
            return (obs - mean) / np.sqrt(var + eps)
        except Exception:
            return obs

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
        # per-opponent builders (perspectives)
        self._opponent_builders = {}
        for c in Colors.ALL_COLORS:
            if c != self.agent_color:
                self._opponent_builders[c] = ObservationBuilder(self.cfg, self.game, c)

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
            # Build observation from this opponent's perspective
            ob = self._opponent_builders.get(p.color.value)
            if ob is None:
                ob = ObservationBuilder(self.cfg, self.game, p.color.value)
                self._opponent_builders[p.color.value] = ob
            opp_obs = ob.build(self.turns, dice)
            opp_obs = self._normalize_for_frozen(opp_obs)
            mask = MoveUtils.action_mask(valid)

            # Use frozen model if available; else random valid
            tok_id = None
            if self._frozen_model is not None:
                try:
                    action, _ = self._frozen_model.predict(opp_obs[None, :], deterministic=False, action_masks=mask)
                    tok_id = int(action)
                    if tok_id not in [m.token_id for m in valid]:
                        tok_id = None
                except Exception:
                    tok_id = None
            if tok_id is None:
                tok_id = random.choice([m.token_id for m in valid])

            res = self.game.execute_move(p, tok_id, dice)
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
