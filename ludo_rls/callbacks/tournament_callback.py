"""Tournament evaluation callback for PPO self-play model.

Runs periodic multi-game tournaments pitting the current PPO policy (shared across all seats)
against scripted baseline strategies using the original environment (`ludo_rl.envs.ludo_env.LudoGymEnv`).

Features:
- Plays N games where PPO occupies each color equally (rotating seating)
- Opponents sampled from a fixed baseline list (configurable)
- Collects metrics: win_rate, mean_rank, capture_diff, avg_turns, illegal_rate
- Logs metrics to TensorBoard via `logger.record` (SB3) with prefix

Usage:
    callback = SelfPlayTournamentCallback(
        make_baseline_env_fn=make_baseline_env,
        n_games=400,
        eval_freq=100_000,
        baselines=["random", "cautious", "killer", "probabilistic"],
        log_prefix="tournament/"
    )
    model.learn(..., callback=[checkpoint_cb, eval_cb, callback])

Note: This callback directly uses the model's policy to select actions for each acting seat sequentially.
We temporarily disable gradient tracking for speed.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ludo.constants import Colors, GameConstants
from ludo_rls.envs.builders.observation_builder import ObservationBuilder
from ludo_rls.envs.model import EnvConfig
from rl_base.callbacks.base_tournament_callback import BaseTournamentCallback


def _soft_action_select(
    policy, obs: np.ndarray, action_mask: np.ndarray | None = None
) -> int:
    """Select action via model policy given raw (non-batched) observation.

    We build a single-batch and call policy.predict with deterministic=False to retain exploration signal.
    If action_mask supplied, we re-sample until a legal action selected (simple rejection sampling) or
    fall back to first legal action.
    """
    obs_batch = obs[None, :]
    action, _ = policy.predict(obs_batch, deterministic=False)
    act = int(action)
    if action_mask is not None and action_mask.sum() > 0 and action_mask[act] == 0:
        # rejection sample a few times
        for _ in range(4):
            action, _ = policy.predict(obs_batch, deterministic=False)
            act = int(action)
            if action_mask[act] == 1:
                return act
        # fallback
        legal_indices = np.nonzero(action_mask)[0]
        return int(legal_indices[0])
    return act


def _ensure_color_feature(obs: np.ndarray, current_color: str) -> np.ndarray:
    """If observation vector is missing color one-hot (detected by length), append it.

    Training self-play env includes 4 extra slots. Baseline env likely shorter.
    We detect by comparing length mod 4 of last features heuristically.
    Simpler: if len(obs) % 4 != 0 and len(obs) <= 40 assume missing and append.
    More robust: check against a known self-play length via policy observation_space if available.
    Here we'll append one-hot if length differs from model.observation_space.shape[0].
    """
    # current policy obs size might include one-hot
    try:
        target_len = int(
            getattr(
                getattr(_ensure_color_feature, "policy_ref", None), "observation_space"
            ).shape[0]
        )  # type: ignore
    except Exception:
        target_len = None
    if target_len is not None and obs.shape[0] == target_len:
        return obs
    # If target len known and we're shorter by exactly 4, append
    if target_len is not None and target_len - obs.shape[0] == 4:
        pass
    elif target_len is None:
        # Heuristic: if not multiple of 4 after removing last 5 scalars assume missing
        pass
    # Build one-hot
    one_hot = np.zeros(4, dtype=obs.dtype)
    for i, c in enumerate(Colors.ALL_COLORS):
        if c == current_color:
            one_hot[i] = 1.0
            break
    return np.concatenate([obs, one_hot], axis=0)


class SelfPlayTournamentCallback(BaseTournamentCallback):
    """Run PPO vs (3) scripted strategies tournaments periodically using shared base class.

    Design:
      - PPO always occupies RED seat (matching training perspective)
      - Opponents: all unique 3-combinations drawn from provided baselines list
      - Distribute total n_games approximately evenly across combinations
      - No random fallback strategies; only provided names are used
      - Metrics aggregated over all simulated games
    """

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
        """Select action using PPO policy with action masking and rejection sampling."""
        return _soft_action_select(policy, obs, action_mask)

    def _build_observation(self, turn_counter: int, dice: int) -> np.ndarray:
        """Build observation for PPO player from RED perspective."""
        if self.obs_builder is None:
            raise RuntimeError("obs_builder not initialized for current game")
        return self.obs_builder._build_observation(turn_counter, dice)

    def _get_action_mask(self, valid_moves: List[dict]) -> np.ndarray | None:
        """Build action mask from valid moves for self-play environment."""
        mask = np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=np.int8)
        if valid_moves:
            for m in valid_moves:
                mask[m["token_id"]] = 1
        return mask

    def _setup_game_and_players(self, combo):
        """Set up game and assign strategies, with obs_builder initialization."""
        game, ppo_player = super()._setup_game_and_players(combo)
        # Initialize observation builder for this game
        from ludo.player import PlayerColor

        self.obs_builder = ObservationBuilder(self.env_cfg, game, PlayerColor.RED.value)
        return game, ppo_player
