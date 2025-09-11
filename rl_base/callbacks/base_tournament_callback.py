"""Base tournament evaluation callback for PPO models.

Provides common functionality for tournament evaluation callbacks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

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
    """Base class for tournament callbacks with shared tournament logic.

    Subclasses must implement:
    - _select_ppo_action: How to select actions using PPO policy
    - _build_observation: How to build observations for PPO
    - _get_action_mask (optional): How to build action masks
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

    def _on_step(self) -> bool:
        if self.num_timesteps == 0:
            return True
        if self.num_timesteps % self.eval_freq != 0:
            return True
        self._run_tournament()
        return True

    def _on_training_end(self) -> None:
        """Final tournament at training end."""
        self._run_tournament()

    @abstractmethod
    def _select_ppo_action(
        self, policy, obs: np.ndarray, action_mask: np.ndarray | None = None
    ) -> int:
        """Select action using PPO policy. Implementation varies by environment."""
        pass

    @abstractmethod
    def _build_observation(self, turn_counter: int, dice: int) -> np.ndarray:
        """Build observation for PPO. Implementation varies by environment."""
        pass

    def _get_action_mask(self, valid_moves: List[dict]) -> np.ndarray | None:
        """Get action mask for valid moves. Optional - return None if not used."""
        return None

    def _setup_game_and_players(
        self, combo: Tuple[str, str, str]
    ) -> Tuple[LudoGame, object]:
        """Set up game and assign strategies to players."""
        game = LudoGame(
            [
                PlayerColor.RED,
                PlayerColor.GREEN,
                PlayerColor.YELLOW,
                PlayerColor.BLUE,
            ]
        )

        # Assign strategies to opponent colors
        opponent_colors = [PlayerColor.GREEN, PlayerColor.YELLOW, PlayerColor.BLUE]
        for idx, color in enumerate(opponent_colors):
            strat_name = combo[idx]
            strat = StrategyFactory.create_strategy(strat_name)
            player = next(p for p in game.players if p.color is color)
            player.set_strategy(strat)
            player.strategy_name = strat_name

        # PPO player (always RED)
        ppo_player = next(p for p in game.players if p.color is PlayerColor.RED)
        ppo_player.strategy_name = "ppo"

        return game, ppo_player

    def _handle_ppo_turn(
        self,
        policy,
        game: LudoGame,
        ppo_player,
        dice: int,
        valid_moves: List[dict],
        turn_counter: int,
        metrics: TournamentMetrics,
    ) -> bool:
        """Handle PPO player turn. Returns True if game ended."""
        obs = self._build_observation(turn_counter, dice)
        action_mask = self._get_action_mask(valid_moves)
        action = self._select_ppo_action(policy, obs, action_mask)

        if valid_moves:
            valid_ids = [m["token_id"] for m in valid_moves]
            if action not in valid_ids:
                metrics.illegal_actions += 1
                action = valid_ids[0]

            move_res = game.execute_move(ppo_player, action, dice)
            captured = move_res.get("captured_tokens", [])
            if captured:
                metrics.captures_for += len(captured)
                metrics.offensive_captures += len(captured)

            if move_res.get("token_finished"):
                pass  # Could track token finish counts if needed

            if move_res.get("game_won"):
                metrics.wins += 1
                metrics.ranks.append(1)
                return True  # Game ended

            if not move_res.get("extra_turn"):
                game.next_turn()
        else:
            # No valid moves
            game.next_turn()

        metrics.ppo_turns += 1
        return False  # Game continues

    def _handle_opponent_turn(
        self,
        game: LudoGame,
        current_player,
        ppo_player,
        dice: int,
        valid_moves: List[dict],
        metrics: TournamentMetrics,
    ) -> bool:
        """Handle opponent player turn. Returns True if game ended."""
        try:
            context = game.get_ai_decision_context(dice)
            if valid_moves:
                token_id = current_player.make_strategic_decision(context)
                move_res = game.execute_move(current_player, token_id, dice)

                # Track captures against PPO
                if move_res.get("captured_tokens"):
                    for ct in move_res["captured_tokens"]:
                        if ct["player_color"] == ppo_player.color.value:
                            metrics.captures_against += 1
                            metrics.defensive_captures += 1

                if move_res.get("token_finished"):
                    pass  # Could track if needed

                if move_res.get("game_won"):
                    return True  # Game ended, opponent won

                if not move_res.get("extra_turn"):
                    game.next_turn()
            else:
                game.next_turn()
        except Exception:
            # On any failure, skip turn
            game.next_turn()

        return False  # Game continues

    def _compute_final_rank(
        self, game: LudoGame, ppo_player, metrics: TournamentMetrics
    ):
        """Compute PPO rank when game ends without explicit winner."""
        if not any(p.has_won() and p is ppo_player for p in game.players):
            # Rank by finished tokens
            finished = [(p.get_finished_tokens_count(), p) for p in game.players]
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

    def _run_tournament(self):
        """Run tournament with current policy against baseline strategies."""
        policy = self.model.policy  # type: ignore
        metrics = TournamentMetrics()

        if not self.combos:
            if self.verbose:
                callback_name = self.__class__.__name__
                print(
                    f"[{callback_name}] Not enough baseline strategies (need >=3). Skipping."
                )
            return metrics.aggregate()

        games_per_combo = max(1, self.n_games // len(self.combos))
        total_games_target = games_per_combo * len(self.combos)
        turn_counter = 0

        for combo in self.combos:
            for _ in range(games_per_combo):
                game, ppo_player = self._setup_game_and_players(combo)
                turns_in_game = 0

                while not game.game_over and turns_in_game < self.max_turns:
                    current_player = game.get_current_player()
                    dice = game.roll_dice()
                    valid_moves = game.get_valid_moves(current_player, dice)

                    game_ended = False
                    if current_player is ppo_player:
                        game_ended = self._handle_ppo_turn(
                            policy,
                            game,
                            ppo_player,
                            dice,
                            valid_moves,
                            turn_counter,
                            metrics,
                        )
                        turn_counter += 1
                    else:
                        game_ended = self._handle_opponent_turn(
                            game, current_player, ppo_player, dice, valid_moves, metrics
                        )

                    if game_ended:
                        break

                    turns_in_game += 1

                # Compute final rank if game didn't end with explicit winner
                self._compute_final_rank(game, ppo_player, metrics)
                metrics.turns.append(turns_in_game)

        # Log aggregated metrics
        agg = metrics.aggregate()
        for k, v in agg.items():
            self.logger.record(self.log_prefix + k, v)

        if self.verbose:
            callback_name = self.__class__.__name__
            print(
                f"[{callback_name}] Steps={self.num_timesteps} games={total_games_target} metrics={agg}"
            )

        # Ensure TensorBoard flush
        if hasattr(self.logger, "writer") and self.logger.writer:
            self.logger.writer.flush()

        return agg

    # Deprecated methods for backward compatibility
    @abstractmethod
    def _policy_select(
        self, policy, obs: np.ndarray, action_mask: np.ndarray = None
    ) -> int:
        """Deprecated. Use _select_ppo_action instead."""
        return self._select_ppo_action(policy, obs, action_mask)

    def _log_metrics(self, metrics_dict: Dict[str, float]):
        """Deprecated. Metrics are now logged automatically."""
        for key, value in metrics_dict.items():
            self.logger.record(f"{self.log_prefix}{key}", value)

    def _create_game_setup(self):
        """Deprecated. Use _setup_game_and_players instead."""
        return LudoGame(
            [
                PlayerColor.RED,
                PlayerColor.GREEN,
                PlayerColor.YELLOW,
                PlayerColor.BLUE,
            ]
        )

    def _assign_strategies_to_players(
        self, game: LudoGame, strategies: List[str], ppo_color: PlayerColor
    ):
        """Deprecated. Use _setup_game_and_players instead."""
        strategy_idx = 0
        for player in game.players:
            if player.color != ppo_color:
                if strategy_idx < len(strategies):
                    strat = StrategyFactory.create_strategy(strategies[strategy_idx])
                    player.set_strategy(strat)
                    player.strategy_name = strategies[strategy_idx]
                    strategy_idx += 1
