"""Tournament evaluation callback for classic multi-seat PPO model.

This mirrors the logic used in the single-seat (`ludo_rls`) implementation but
keeps assumptions consistent with the classic environment where the policy is
shared across seats and each seat viewpoint is rotated by re-building
observations with the appropriate perspective color.

Features:
- PPO occupies a single fixed seat (RED) for evaluation consistency
- Remaining 3 seats are filled by combinations of baseline scripted strategies
- Plays N games distributed across all 3-opponent combinations
- Aggregates: win_rate, mean_rank, capture_diff, avg_turns, illegal rates, capture ratios
- Logs metrics to TensorBoard under a configurable prefix

Usage:
    callback = ClassicTournamentCallback(
        baselines=["optimist","balanced","cautious"],
        n_games=240,
        eval_freq=100_000,
        max_turns=500,
        log_prefix="tournament/",
        verbose=1,
    )
    model.learn(..., callback=[checkpoint_cb, eval_cb, callback])

Notes:
- We rebuild observations using the classic `ObservationBuilder` from `ludo_rl`.
- Action selection uses soft sampling (stochastic) with simple rejection sampling
  against action mask.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ludo_rl.envs.builders.observation_builder import ObservationBuilder
from ludo_rl.envs.model import EnvConfig
from rl_base.callbacks.base_tournament_callback import BaseTournamentCallback


def _policy_select(policy, obs: np.ndarray) -> int:
    """Single forward pass action selection (stochastic) mirroring env training path.

    Env training does NOT perform rejection sampling against masks; illegal selections
    incur penalty and are mapped to a fallback valid move internally. We emulate that
    here for parity so tournament metrics reflect true policy distribution quality.
    """
    action, _ = policy.predict(obs[None, :], deterministic=False)
    return int(action)


class ClassicTournamentCallback(BaseTournamentCallback):
    """Tournament callback for classic multi-seat PPO using shared base class."""

    def __init__(
        self,
        baselines: Sequence[str],
        n_games: int = 120,
        eval_freq: int = 100_000,
        max_turns: int = 1000,
        log_prefix: str = "tournament/",
        verbose: int = 0,
    ):
        super().__init__(baselines, n_games, eval_freq, max_turns, log_prefix, verbose)
        # Set up environment configuration and observation builder
        self.env_cfg = EnvConfig(max_turns=max_turns)
        self.obs_builder = None  # Will be created per game

    def _select_ppo_action(
        self, policy, obs: np.ndarray, action_mask: np.ndarray | None = None
    ) -> int:
        """Select action using PPO policy with simple stochastic sampling."""
        return _policy_select(policy, obs)

    def _build_observation(self, turn_counter: int, dice: int) -> np.ndarray:
        """Build observation for PPO player from RED perspective."""
        if self.obs_builder is None:
            raise RuntimeError("obs_builder not initialized for current game")
        return self.obs_builder._build_observation(turn_counter, dice)

    def _setup_game_and_players(self, combo):
        """Set up game and assign strategies, with obs_builder initialization."""
        game, ppo_player = super()._setup_game_and_players(combo)
        # Initialize observation builder for this game
        from ludo.player import PlayerColor

        self.obs_builder = ObservationBuilder(self.env_cfg, game, PlayerColor.RED.value)
        return game, ppo_player

    # Support the deprecated method for backward compatibility
    def _policy_select(
        self, policy, obs: np.ndarray, action_mask: np.ndarray = None
    ) -> int:
        """Deprecated method - delegates to _select_ppo_action."""
        return self._select_ppo_action(policy, obs, action_mask)
