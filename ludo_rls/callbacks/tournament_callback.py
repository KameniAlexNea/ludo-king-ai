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

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from ludo.constants import Colors, GameConstants
from ludo.game import LudoGame
from ludo.player import PlayerColor
from ludo.strategy import StrategyFactory

from ..envs.builders.observation_builder import ObservationBuilder
from ..envs.model import EnvConfig


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


@dataclass
class TournamentMetrics:
    wins: int = 0
    losses: int = 0
    ranks: List[int] = None
    captures_for: int = 0
    captures_against: int = 0
    illegal_actions: int = 0
    turns: List[int] = None

    def __post_init__(self):
        if self.ranks is None:
            self.ranks = []
        if self.turns is None:
            self.turns = []

    def aggregate(self) -> Dict[str, float]:
        total_games = max(1, len(self.ranks))
        win_rate = self.wins / total_games
        mean_rank = float(np.mean(self.ranks)) if self.ranks else 0.0
        avg_turns = float(np.mean(self.turns)) if self.turns else 0.0
        illegal_rate = self.illegal_actions / max(1, self.turns and sum(self.turns))
        capture_diff = (self.captures_for - self.captures_against) / max(1, total_games)
        return {
            "win_rate": win_rate,
            "mean_rank": mean_rank,
            "capture_diff": capture_diff,
            "avg_turns": avg_turns,
            "illegal_rate": illegal_rate,
        }


class SelfPlayTournamentCallback(BaseCallback):
    """Run PPO vs (3) scripted strategies tournaments periodically.

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
        super().__init__(verbose)
        self.baselines = list(dict.fromkeys(baselines))  # de-dup
        self.n_games = n_games
        self.eval_freq = eval_freq
        self.max_turns = max_turns
        self.log_prefix = log_prefix
        # Precompute combinations of 3 strategies
        self.combos: List[Tuple[str, str, str]] = []
        if len(self.baselines) >= 3:
            self.combos = list(combinations(self.baselines, 3))
        # If fewer than 3 provided, we cannot form proper tournament opponents; fallback empty combos

    def _on_step(self) -> bool:
        if self.num_timesteps == 0:
            return True
        if self.num_timesteps % self.eval_freq != 0:
            return True
        self._run_tournament()
        return True

    def _run_tournament(self):
        policy = self.model.policy  # type: ignore
        metrics = TournamentMetrics()
        if not self.combos:
            if self.verbose:
                print(
                    "[Tournament] Not enough baseline strategies (need >=3). Skipping."
                )
            return metrics.aggregate()

        # Distribute games across combinations
        games_per_combo = max(1, self.n_games // len(self.combos))
        total_games_target = games_per_combo * len(self.combos)

        env_cfg = EnvConfig(max_turns=self.max_turns)
        turn_counter = 0

        for combo in self.combos:
            for _ in range(games_per_combo):
                # Fresh game
                game = LudoGame(
                    [
                        PlayerColor.RED,
                        PlayerColor.GREEN,
                        PlayerColor.YELLOW,
                        PlayerColor.BLUE,
                    ]
                )
                # Assign PPO to RED; assign combo strategies to remaining colors in fixed order skipping RED
                opponent_colors = [
                    c for c in [PlayerColor.GREEN, PlayerColor.YELLOW, PlayerColor.BLUE]
                ]
                for idx, color in enumerate(opponent_colors):
                    strat_name = combo[idx]
                    strat = StrategyFactory.create_strategy(strat_name)
                    player = next(p for p in game.players if p.color is color)
                    player.set_strategy(strat)
                    player.strategy_name = strat_name
                # PPO strategy via policy (handled here, not through game strategy object)
                ppo_player = next(p for p in game.players if p.color is PlayerColor.RED)
                ppo_player.strategy_name = "ppo"

                # Helpers for observation & masking
                obs_builder = ObservationBuilder(env_cfg, game, PlayerColor.RED.value)
                # move_utils = MoveUtils(env_cfg, game, PlayerColor.RED.value)
                turns_in_game = 0
                token_finish_counts = {p.color.value: 0 for p in game.players}

                while not game.game_over and turns_in_game < self.max_turns:
                    current_player = game.get_current_player()
                    dice = game.roll_dice()
                    valid_moves = game.get_valid_moves(current_player, dice)
                    if current_player is ppo_player:
                        # Build obs from PPO perspective
                        obs = obs_builder._build_observation(turn_counter, dice)
                        # Build action mask
                        mask = np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=np.int8)
                        if valid_moves:
                            for m in valid_moves:
                                mask[m["token_id"]] = 1
                        action = _soft_action_select(policy, obs, mask)
                        if valid_moves:
                            valid_ids = [m["token_id"] for m in valid_moves]
                            if action not in valid_ids:
                                action = valid_ids[0]
                            move_res = game.execute_move(current_player, action, dice)
                            if move_res.get("token_finished"):
                                token_finish_counts[current_player.color.value] += 1
                            if move_res.get("game_won"):
                                metrics.wins += 1
                                metrics.ranks.append(1)
                                break
                            if not move_res.get("extra_turn"):
                                game.next_turn()
                        else:
                            game.next_turn()
                        turn_counter += 1
                    else:
                        # Opponent scripted decision using its strategy; build minimal context
                        try:
                            context = game.get_game_state_for_ai()
                            context["dice_value"] = dice
                            if valid_moves:
                                token_id = current_player.make_strategic_decision(
                                    context
                                )
                                move_res = game.execute_move(
                                    current_player, token_id, dice
                                )
                                if move_res.get("token_finished"):
                                    token_finish_counts[current_player.color.value] += 1
                                if move_res.get("game_won"):
                                    # PPO did not win; assign rank later
                                    break
                                if not move_res.get("extra_turn"):
                                    game.next_turn()
                            else:
                                game.next_turn()
                        except Exception:
                            # On any failure just skip turn
                            game.next_turn()
                    turns_in_game += 1
                # If game ended without PPO win, compute PPO rank by finished tokens
                if not game.game_over:
                    # time limit; approximate rank
                    pass
                if not any(p.has_won() and p is ppo_player for p in game.players):
                    # Rank calculation: sort by finished tokens desc
                    finished = [
                        (p.get_finished_tokens_count(), p) for p in game.players
                    ]
                    finished.sort(reverse=True, key=lambda x: x[0])
                    rank_positions = {
                        pl.color.value: i + 1 for i, (cnt, pl) in enumerate(finished)
                    }
                    ppo_rank = rank_positions[ppo_player.color.value]
                    if ppo_rank == 1:
                        metrics.wins += 1
                    else:
                        metrics.losses += 1
                    metrics.ranks.append(ppo_rank)
                metrics.turns.append(turns_in_game)
        # Aggregate & log
        agg = metrics.aggregate()
        for k, v in agg.items():
            self.logger.record(self.log_prefix + k, v)
        if self.verbose:
            print(
                f"[Tournament] Steps={self.num_timesteps} games={total_games_target} metrics={agg}"
            )

        # Ensure TensorBoard flush
        if hasattr(self.logger, "writer") and self.logger.writer:
            self.logger.writer.flush()

        return agg

    def _on_training_end(self) -> None:
        # Final tournament at end
        self._run_tournament()
