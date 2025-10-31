"""Turn-based wrapper that centralizes control of the AEC Ludo environment."""

from __future__ import annotations

from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ludo_engine.strategies.strategy import Strategy, StrategyFactory
from sb3_contrib import MaskablePPO

from ...configs.config import MultiAgentConfig
from .opponent_pool import OpponentPoolManager
from .raw_env import raw_env


class TurnBasedSelfPlayEnv(gym.Env):
    """Gym-compatible wrapper that coordinates the turn-based PettingZoo env."""

    metadata = {"render_modes": ["human", "ansi"], "name": "LudoTurnBased-v0"}

    def __init__(
        self,
        base_env: raw_env,
        opponent_pool: Optional[OpponentPoolManager] = None,
        ma_cfg: Optional[MultiAgentConfig] = None,
    ):
        super().__init__()
        self.base_env = base_env
        self.possible_agents = list(base_env.possible_agents)
        if not self.possible_agents:
            raise ValueError("Base environment must define possible agents")
        self.agent_indices = {
            agent: idx for idx, agent in enumerate(self.possible_agents)
        }

        sample_agent = self.possible_agents[0]
        self.observation_space = spaces.Dict(
            {
                "observation": base_env.observation_space(sample_agent),
                "action_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(base_env.action_space(sample_agent).n,),
                    dtype=np.int8,
                ),
                "agent_index": spaces.Discrete(len(self.possible_agents)),
            }
        )
        self.action_space = base_env.action_space(sample_agent)

        self._current_agent: Optional[str] = None
        self._last_obs: Optional[dict[str, np.ndarray]] = None
        self._last_action_mask: Optional[np.ndarray] = None

        self.opponent_pool = opponent_pool
        self.ma_cfg = ma_cfg or MultiAgentConfig()
        self.opponent_assignments: Dict[str, str] = {}  # agent -> model_path
        self.scripted_assignments: Dict[
            str, Strategy
        ] = {}  # agent -> strategy_instance
        self.opponent_models: Dict[str, MaskablePPO] = {}

    def _sync_active_agent(self) -> None:
        while self.base_env.agents:
            active = self.base_env.agent_selection
            if not (
                self.base_env.terminations[active] or self.base_env.truncations[active]
            ):
                break
            self.base_env.step(None)
        self._current_agent = (
            self.base_env.agent_selection if self.base_env.agents else None
        )

    def _build_observation(self, agent: str) -> dict[str, np.ndarray]:
        raw_obs = self.base_env.observe(agent)
        obs_array = raw_obs["observation"].astype(np.float32, copy=False)
        mask = raw_obs["action_mask"].astype(np.int8, copy=False)
        self._last_action_mask = mask
        obs = {
            "observation": obs_array,
            "action_mask": mask,
            "agent_index": np.array(self.agent_indices[agent], dtype=np.int64),
        }
        self._last_obs = obs
        return obs

    def action_masks(self) -> np.ndarray:
        if self._last_action_mask is None:
            return np.ones(self.observation_space["action_mask"].shape, dtype=np.int8)
        return self._last_action_mask.copy()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.base_env.reset(seed=seed, options=options)

        self.opponent_assignments = {}
        self.scripted_assignments = {}

        if self.opponent_pool and self.ma_cfg.enable_self_play:
            num_opponents = min(
                self.ma_cfg.self_play_num_opponents, len(self.possible_agents) - 1
            )
            if num_opponents > 0:
                agents_to_assign = np.random.choice(
                    self.possible_agents, num_opponents, replace=False
                )
                for agent in agents_to_assign:
                    # Decide: use fixed scripted opponent or self-play opponent
                    use_scripted = (
                        self.ma_cfg.use_fixed_opponents
                        and np.random.random() < self.ma_cfg.fixed_opponent_ratio
                    )

                    if use_scripted:
                        # Use fixed scripted opponent
                        strategy_name = np.random.choice(
                            self.ma_cfg.fixed_opponent_strategies
                        )
                        strategy = StrategyFactory.create_strategy(strategy_name)
                        self.scripted_assignments[agent] = strategy
                    else:
                        # Use self-play opponent from pool
                        opp = self.opponent_pool.sample_opponent()
                        if opp:
                            self.opponent_assignments[agent] = opp
                            # Load model immediately if not already cached
                            if opp not in self.opponent_models:
                                self.opponent_models[opp] = MaskablePPO.load(opp)
                        else:
                            # Fallback to scripted opponent if pool is empty
                            strategy = StrategyFactory.create_strategy(
                                self.ma_cfg.self_play_opponent_fallback
                            )
                            self.scripted_assignments[agent] = strategy

        self._sync_active_agent()

        if self._current_agent is None:
            raise RuntimeError("Environment reset produced no active agents")

        obs = self._build_observation(self._current_agent)
        info = self.base_env.infos.get(self._current_agent, {}).copy()
        info["active_agent"] = self._current_agent
        return obs, info

    def step(self, action):
        if self._current_agent is None:
            obs = self._build_terminal_obs()
            return obs, 0.0, True, False, {"terminal_observation": obs}

        acting_agent = self._current_agent
        prev_obs = self._last_obs

        if acting_agent in self.opponent_assignments:
            # Use opponent model from pool
            opp_path = self.opponent_assignments[acting_agent]
            opp_model = self.opponent_models[opp_path]
            obs_dict = self._build_observation(acting_agent)
            opp_action, _ = opp_model.predict(obs_dict, deterministic=True)
            self.base_env.step(int(opp_action))
        elif acting_agent in self.scripted_assignments:
            # Use scripted strategy as fallback
            strategy = self.scripted_assignments[acting_agent]
            agent_idx = self.agent_indices[acting_agent]
            player = self.base_env.game.players[agent_idx]
            dice = self.base_env._pending_dice.get(acting_agent, 0)
            valid_moves = self.base_env._pending_valid_moves.get(acting_agent, [])

            if valid_moves:
                decision_context = self.base_env.game.get_ai_decision_context(dice)
                chosen_token = strategy.decide(player, decision_context)
            else:
                chosen_token = 0
            self.base_env.step(chosen_token)
        else:
            # Learning agent controls this position
            self.base_env.step(int(action))

        reward = float(self.base_env.rewards.get(acting_agent, 0.0))

        self._sync_active_agent()

        episode_done = self._current_agent is None
        terminated = episode_done and any(self.base_env.terminations.values())
        truncated = episode_done and any(self.base_env.truncations.values())

        if episode_done:
            obs = prev_obs if prev_obs is not None else self._build_terminal_obs()
            self._current_agent = None
            self._last_action_mask = np.zeros_like(obs["action_mask"], dtype=np.int8)
        else:
            obs = self._build_observation(self._current_agent)

        info = self.base_env.infos.get(acting_agent, {}).copy()
        info["acting_agent"] = acting_agent
        if not episode_done and self._current_agent is not None:
            info["next_agent"] = self._current_agent

        if episode_done:
            info["terminal_observation"] = obs

        return obs, reward, terminated, truncated, info

    def _build_terminal_obs(self) -> dict[str, np.ndarray]:
        obs_space = self.observation_space["observation"]
        mask_shape = self.observation_space["action_mask"].shape
        return {
            "observation": np.zeros(obs_space.shape, dtype=obs_space.dtype),
            "action_mask": np.zeros(mask_shape, dtype=np.int8),
            "agent_index": np.zeros((), dtype=np.int64),
        }

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()
