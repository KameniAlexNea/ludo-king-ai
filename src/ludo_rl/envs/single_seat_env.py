from __future__ import annotations

import copy
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sb3_contrib import MaskablePPO

from models.configs.config import EnvConfig
from models.envs.ludo_env_aec.raw_env import raw_env as AECEnv


class SingleSeatSelfPlayEnv(gym.Env):
    """Expose only one agent's turns to SB3 while playing all other seats internally.

    - Wraps the PettingZoo AEC env (raw_env)
    - On reset, optionally randomizes the mapping of seats to colors
    - step(action) is only called for the learner seat; opponent turns are simulated
      internally using other PPO policies or scripted bots.
    - Reward is the shaped reward of the learner seat. Terminal reward is included
      even if the episode ends on an opponent move.
    """

    metadata = {"render_modes": ["human", "ansi"], "name": "LudoSingleSeat-v0"}

    def __init__(
        self,
        cfg: Optional[EnvConfig],
        learner_agent_id: str,  # e.g., "player_0"
        opponents: Optional[Dict[str, MaskablePPO]] = None,  # agent_id -> policy
        scripted_fallback: str = "balanced",
    ) -> None:
        super().__init__()
        self.cfg = copy.deepcopy(cfg) if cfg is not None else EnvConfig()
        # Ensure flattened, multi-agent-compatible observation builder
        self.cfg.multi_agent = True
        self.learner_id = learner_agent_id
        self.scripted_fallback = scripted_fallback
        self._opponents: Dict[str, MaskablePPO] = opponents or {}

        # Underlying AEC env
        self.base_env = AECEnv(self.cfg)
        self.possible_agents = list(self.base_env.possible_agents)
        self.agent_indices = {ag: i for i, ag in enumerate(self.possible_agents)}

        # Multi-input observation to match our MaskablePPO policy
        sample_agent = self.possible_agents[0]
        self.observation_space = spaces.Dict(
            {
                "observation": self.base_env.observation_space(sample_agent),
                "action_mask": spaces.Box(
                    low=0, high=1,
                    shape=(self.base_env.action_space(sample_agent).n,),
                    dtype=np.int8,
                ),
                "agent_index": spaces.Discrete(len(self.possible_agents)),
            }
        )
        self.action_space = self.base_env.action_space(sample_agent)

        self._last_mask: Optional[np.ndarray] = None
        self._last_obs: Optional[dict] = None
        self._terminated = False
        self._truncated = False

    # ---------------------------
    # Public opponent control
    # ---------------------------
    def set_opponents(self, opponents: Dict[str, MaskablePPO]) -> None:
        self._opponents = dict(opponents)

    # ---------------------------
    # Gym API
    # ---------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = None, {}
        self.base_env.reset(seed=seed, options=options)
        # Fast-forward opponent turns until it's learner's turn or end
        self._terminated = False
        self._truncated = False
        self._run_opponents_until_learner_turn()

        if self._terminated or self._truncated:
            # Episode ended before learner turn (rare but possible)
            obs = self._terminal_obs()
            info["terminal_observation"] = obs
            return obs, info

        obs = self._build_obs(self.learner_id)
        info = self.base_env.infos.get(self.learner_id, {}).copy()
        return obs, info

    def step(self, action):
        if self._terminated or self._truncated:
            obs = self._terminal_obs()
            return obs, 0.0, True, False, {"terminal_observation": obs}

        if self.base_env.agent_selection != self.learner_id:
            # Should never happen due to our control flow
            self._run_opponents_until_learner_turn()

        # Learner acts
        self.base_env.step(int(action))
        reward = float(self.base_env.rewards.get(self.learner_id, 0.0))
        self._terminated = any(self.base_env.terminations.values())
        self._truncated = any(self.base_env.truncations.values())

        if not (self._terminated or self._truncated):
            # Run opponents until it's learner's turn again or episode ends
            self._run_opponents_until_learner_turn()
            if self._terminated or self._truncated:
                # Terminal reward is set for all agents
                reward += float(self.base_env.rewards.get(self.learner_id, 0.0))

        if self._terminated or self._truncated:
            obs = self._terminal_obs()
        else:
            obs = self._build_obs(self.learner_id)

        info = self.base_env.infos.get(self.learner_id, {}).copy()
        return obs, reward, self._terminated, self._truncated, info

    def action_masks(self) -> np.ndarray:
        if self._last_mask is None:
            return np.ones(self.observation_space["action_mask"].shape, dtype=np.int8)
        return self._last_mask.copy()

    # ---------------------------
    # Helpers
    # ---------------------------
    def _terminal_obs(self) -> dict:
        obs_space = self.observation_space["observation"]
        mask_shape = self.observation_space["action_mask"].shape
        return {
            "observation": np.zeros(obs_space.shape, dtype=obs_space.dtype),
            "action_mask": np.zeros(mask_shape, dtype=np.int8),
            "agent_index": np.array(self.agent_indices[self.learner_id], dtype=np.int64),
        }

    def _build_obs(self, agent: str) -> dict:
        raw = self.base_env.observe(agent)
        obs_array = raw["observation"].astype(np.float32, copy=False)
        mask = raw["action_mask"].astype(np.int8, copy=False)
        self._last_mask = mask
        out = {
            "observation": obs_array,
            "action_mask": mask,
            "agent_index": np.array(self.agent_indices[agent], dtype=np.int64),
        }
        self._last_obs = out
        return out

    def _run_opponents_until_learner_turn(self) -> None:
        # Continue stepping until it's learner's turn or episode ends
        while True:
            agent = self.base_env.agent_selection
            term = any(self.base_env.terminations.values())
            trunc = any(self.base_env.truncations.values())
            if term or trunc:
                self._terminated, self._truncated = term, trunc
                return
            if agent == self.learner_id:
                return

            # Opponent acts (policy first, scripted fallback otherwise)
            action = self._opponent_action(agent)
            self.base_env.step(int(action))

    def _opponent_action(self, agent: str) -> int:
        # Policy-controlled opponent
        if agent in self._opponents and self._opponents[agent] is not None:
            obs = self._build_obs(agent)
            # Use stochastic actions for diversity
            action, _ = self._opponents[agent].predict(obs, deterministic=False)
            return int(action)

        # Scripted fallback
        # raw_env supports valid_move_tokens and pending_dice indirectly via its internals;
        # for a simple fallback, choose a valid token uniformly at random.
        raw = self.base_env.observe(agent)
        mask = raw["action_mask"].astype(np.int8, copy=False)
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            return 0
        return int(np.random.choice(valid))

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()
