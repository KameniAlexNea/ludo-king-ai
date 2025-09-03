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

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from ludo.game import LudoGame
from ludo.player import PlayerColor
from ludo.strategy import StrategyFactory

from ..envs.builders.observation_builder import ObservationBuilder
from ..envs.model import EnvConfig


def _policy_select(policy, obs: np.ndarray) -> int:
    """Single forward pass action selection (stochastic) mirroring env training path.

    Env training does NOT perform rejection sampling against masks; illegal selections
    incur penalty and are mapped to a fallback valid move internally. We emulate that
    here for parity so tournament metrics reflect true policy distribution quality.
    """
    action, _ = policy.predict(obs[None, :], deterministic=False)
    return int(action)


@dataclass
class TournamentMetrics:
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
            else float("inf") if self.captures_for > 0 else 0.0
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


class ClassicTournamentCallback(BaseCallback):
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
        self.combos: List[Tuple[str, str, str]] = []
        if len(self.baselines) >= 3:
            self.combos = list(combinations(self.baselines, 3))

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
                    "[ClassicTournament] Not enough baseline strategies (need >=3). Skipping."
                )
            return metrics.aggregate()

        games_per_combo = max(1, self.n_games // len(self.combos))
        total_games_target = games_per_combo * len(self.combos)
        env_cfg = EnvConfig(max_turns=self.max_turns)
        turn_counter = 0

        for combo in self.combos:
            for _ in range(games_per_combo):
                game = LudoGame(
                    [
                        PlayerColor.RED,
                        PlayerColor.GREEN,
                        PlayerColor.YELLOW,
                        PlayerColor.BLUE,
                    ]
                )
                opponent_colors = [
                    PlayerColor.GREEN,
                    PlayerColor.YELLOW,
                    PlayerColor.BLUE,
                ]
                for idx, color in enumerate(opponent_colors):
                    strat_name = combo[idx]
                    strat = StrategyFactory.create_strategy(strat_name)
                    player = next(p for p in game.players if p.color is color)
                    player.set_strategy(strat)
                    player.strategy_name = strat_name
                ppo_player = next(p for p in game.players if p.color is PlayerColor.RED)
                ppo_player.strategy_name = "ppo"
                obs_builder = ObservationBuilder(env_cfg, game, PlayerColor.RED.value)

                turns_in_game = 0
                token_finish_counts = {p.color.value: 0 for p in game.players}

                while not game.game_over and turns_in_game < self.max_turns:
                    current_player = game.get_current_player()
                    dice = game.roll_dice()
                    valid_moves = game.get_valid_moves(current_player, dice)
                    if current_player is ppo_player:
                        # Build observation exactly like training env (turn count = PPO decision turns so far)
                        obs = obs_builder._build_observation(turn_counter, dice)
                        action = _policy_select(policy, obs)
                        if valid_moves:
                            valid_ids = [m["token_id"] for m in valid_moves]
                            if action not in valid_ids:
                                # Mirror env: mark illegal then fallback to first valid
                                metrics.illegal_actions += 1
                                action = valid_ids[0]
                            move_res = game.execute_move(current_player, action, dice)
                            captured = move_res.get("captured_tokens", [])
                            if captured:
                                metrics.captures_for += len(captured)
                                metrics.offensive_captures += len(captured)
                            if move_res.get("token_finished"):
                                token_finish_counts[current_player.color.value] += 1
                            if move_res.get("game_won"):
                                metrics.wins += 1
                                metrics.ranks.append(1)
                                break
                            if not move_res.get("extra_turn"):
                                game.next_turn()
                        else:
                            # No valid moves: env would not flag illegal; advance turn
                            game.next_turn()
                        turn_counter += 1
                        metrics.ppo_turns += 1
                    else:
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
                                if move_res.get("captured_tokens"):
                                    for ct in move_res["captured_tokens"]:
                                        if ct["player_color"] == ppo_player.color.value:
                                            metrics.captures_against += 1
                                            metrics.defensive_captures += 1
                                if move_res.get("token_finished"):
                                    token_finish_counts[current_player.color.value] += 1
                                if move_res.get("game_won"):
                                    break
                                if not move_res.get("extra_turn"):
                                    game.next_turn()
                            else:
                                game.next_turn()
                        except Exception:
                            game.next_turn()
                    turns_in_game += 1
                if not any(p.has_won() and p is ppo_player for p in game.players):
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
        agg = metrics.aggregate()
        for k, v in agg.items():
            self.logger.record(self.log_prefix + k, v)
        if self.verbose:
            print(
                f"[ClassicTournament] Steps={self.num_timesteps} games={total_games_target} metrics={agg}"
            )
        if hasattr(self.logger, "writer") and self.logger.writer:
            self.logger.writer.flush()
        return agg

    def _on_training_end(self) -> None:
        self._run_tournament()
