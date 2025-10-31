from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sb3_contrib import MaskablePPO

from models.configs.config import EnvConfig
from models.envs.ludo_env_aec.raw_env import raw_env as AECEnv

AGENTS = ("player_0", "player_1", "player_2", "player_3")


@dataclass
class MatchResult:
    winner: Optional[str]
    rewards: Dict[str, float]
    length: int


def _build_obs(env: AECEnv, agent: str) -> dict:
    raw = env.observe(agent)
    # Build multi-input observation expected by MaskablePPO
    obs = {
        "observation": raw["observation"].astype(np.float32, copy=False),
        "action_mask": raw["action_mask"].astype(np.int8, copy=False),
        "agent_index": np.array(AGENTS.index(agent), dtype=np.int64),
    }
    return obs


def play_match(
    policies: Dict[str, Optional[MaskablePPO]],
    cfg: EnvConfig,
    *,
    deterministic: bool = True,
) -> MatchResult:
    """Run one full AEC match using provided policies or random fallback.

    policies maps agent_id -> MaskablePPO (or None for random fallback).
    Returns the winner, per-agent terminal rewards, and episode length.
    """
    cfg_copy = copy.deepcopy(cfg)
    # Ensure flattened observations for policy inputs
    cfg_copy.multi_agent = True
    env = AECEnv(cfg_copy)
    env.reset()
    steps = 0

    while env.agents:
        agent = env.agent_selection
        obs = _build_obs(env, agent)
        if policies.get(agent) is not None:
            action, _ = policies[agent].predict(obs, deterministic=deterministic)
            action = int(action)
        else:
            mask = obs["action_mask"]
            valid = np.flatnonzero(mask)
            action = int(np.random.choice(valid) if len(valid) else 0)
        env.step(action)
        steps += 1

    # Terminal rewards populated for all agents
    rewards = {ag: float(env.rewards.get(ag, 0.0)) for ag in AGENTS}
    winner = env.game.winner
    winner_agent = None
    if winner is not None:
        # env has color->agent map
        try:
            winner_agent = env._color_agent_map[winner.color]
        except Exception:
            winner_agent = None

    env.close()
    return MatchResult(winner=winner_agent, rewards=rewards, length=steps)


@dataclass
class EvalSummary:
    name: str
    games: int
    wins: int = 0
    losses: int = 0
    draws: int = 0
    avg_len: float = 0.0

    def update(self, res: MatchResult, my_agent: str):
        self.avg_len += res.length
        if res.winner is None:
            self.draws += 1
        elif res.winner == my_agent:
            self.wins += 1
        else:
            self.losses += 1

    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.games)


def tournament_round_robin(
    policies: Dict[str, MaskablePPO],
    cfg: EnvConfig,
    games_per_pair: int = 10,
) -> Dict[str, EvalSummary]:
    """Simple round-robin tournament across the 4 agents.

    Each pair plays games_per_pair matches; other seats are filled by the
    remaining policies. The summary is per agent id.
    """
    agents = list(policies.keys())
    summaries = {ag: EvalSummary(name=ag, games=0) for ag in agents}

    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            a_i = agents[i]
            a_j = agents[j]
            # For each pair, play games with all 4 policies active
            for g in range(games_per_pair):
                res = play_match(policies, cfg, deterministic=True)
                summaries[a_i].update(res, a_i)
                summaries[a_j].update(res, a_j)
                summaries[a_i].games += 1
                summaries[a_j].games += 1

    # Normalize average length
    for s in summaries.values():
        s.avg_len = s.avg_len / max(1, s.games)
    return summaries
