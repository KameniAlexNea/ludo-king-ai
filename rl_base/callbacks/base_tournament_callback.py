"""Base tournament evaluation callback for PPO models.

Provides common functionality for tournament evaluation callbacks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from ludo.game import LudoGame
from ludo.player import PlayerColor
from ludo.strategy import StrategyFactory


@dataclass
class TournamentMetrics:
    """Common tournament metrics for both classic and self-play variants."""
    wins: int = 0
    losses: int = 0
    ranks: List[int] = None
    captures_for: int = 0
    captures_against: int = 0
    illegal_actions: int = 0
    turns: List[int] = None
    ppo_turns: int = 0
    offensive_captures: int = 0
    defensive_captures: int = 0

    def __post_init__(self):
        if self.ranks is None:
            self.ranks = []
        if self.turns is None:
            self.turns = []

    def aggregate(self) -> Dict[str, float]:
        """Aggregate metrics into a summary dictionary."""
        total_games = max(1, len(self.ranks))
        win_rate = self.wins / total_games
        mean_rank = float(np.mean(self.ranks)) if self.ranks else 0.0
        avg_turns = float(np.mean(self.turns)) if self.turns else 0.0
        total_turns = max(1, sum(self.turns) if self.turns else 0)
        illegal_rate = self.illegal_actions / total_turns
        illegal_rate_ppo = self.illegal_actions / max(1, self.ppo_turns)
        capture_diff = (self.captures_for - self.captures_against) / max(1, total_games)
        offensive_per_game = self.captures_for / max(1, total_games)
        defensive_per_game = self.captures_against / max(1, total_games)
        capture_ratio = (
            self.captures_for / max(1, self.captures_against)
            if self.captures_against
            else float("inf")
            if self.captures_for > 0
            else 0.0
        )
        return {
            "win_rate": win_rate,
            "mean_rank": mean_rank,
            "capture_diff": capture_diff,
            "avg_turns": avg_turns,
            "illegal_rate": illegal_rate,
            "illegal_rate_ppo": illegal_rate_ppo,
            "offensive_captures_per_game": offensive_per_game,
            "defensive_captures_per_game": defensive_per_game,
            "capture_ratio": capture_ratio,
        }


class BaseTournamentCallback(BaseCallback, ABC):
    """Base class for tournament callbacks with common functionality."""

    def __init__(
        self,
        baselines: Sequence[str],
        n_games: int = 120,
        eval_freq: int = 100_000,
        max_turns: int = 1000,
        log_prefix: str = "tournament/",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.baselines = list(dict.fromkeys(baselines))
        self.n_games = n_games
        self.eval_freq = eval_freq
        self.max_turns = max_turns
        self.log_prefix = log_prefix

    def _on_step(self) -> bool:
        if self.num_timesteps == 0:
            return True
        if self.num_timesteps % self.eval_freq != 0:
            return True
        self._run_tournament()
        return True

    @abstractmethod
    def _run_tournament(self):
        """Run the tournament. To be implemented by subclasses."""
        pass

    @abstractmethod
    def _policy_select(self, policy, obs: np.ndarray, action_mask: np.ndarray = None) -> int:
        """Select action using the policy. To be implemented by subclasses."""
        pass

    def _log_metrics(self, metrics_dict: Dict[str, float]):
        """Log metrics to TensorBoard."""
        for key, value in metrics_dict.items():
            self.logger.record(f"{self.log_prefix}{key}", value)

    def _create_game_setup(self):
        """Create basic game setup with players."""
        return LudoGame([
            PlayerColor.RED,
            PlayerColor.GREEN, 
            PlayerColor.YELLOW,
            PlayerColor.BLUE,
        ])

    def _assign_strategies_to_players(self, game: LudoGame, strategies: List[str], ppo_color: PlayerColor):
        """Assign strategies to non-PPO players."""
        strategy_idx = 0
        for player in game.players:
            if player.color != ppo_color:
                if strategy_idx < len(strategies):
                    strat = StrategyFactory.create_strategy(strategies[strategy_idx])
                    player.set_strategy(strat)
                    player.strategy_name = strategies[strategy_idx]
                    strategy_idx += 1
