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
from typing import Callable, Dict, List, Sequence
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from ludo.constants import Colors, GameConstants

# Type alias for environment factory returning a fresh baseline (non self-play) env
BaselineEnvFactory = Callable[[str], object]


def _soft_action_select(policy, obs: np.ndarray, action_mask: np.ndarray | None = None) -> int:
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
    def __init__(
        self,
        make_baseline_env_fn: Callable[[List[str]], object],
        baselines: Sequence[str],
        n_games: int = 200,
        eval_freq: int = 100_000,
        log_prefix: str = "tournament/",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.make_env_fn = make_baseline_env_fn
        self.baselines = list(baselines)
        self.n_games = n_games
        self.eval_freq = eval_freq
        self.log_prefix = log_prefix

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
        # We cycle through agent seating positions uniformly
        seats = list(Colors.ALL_COLORS)
        games_per_seat = max(1, self.n_games // len(seats))
        total_games_target = games_per_seat * len(seats)
        for seat_color in seats:
            for _ in range(games_per_seat):
                env = self.make_env_fn([b for b in self.baselines if b])
                obs, _ = env.reset()
                done = False
                turns = 0
                # Force PPO seat color if env supports agent_color attribute
                if hasattr(env, "agent_color"):
                    env.agent_color = seat_color  # type: ignore
                while True:
                    # Determine current acting player color (env API differences between baseline & self-play)
                    current_color = getattr(env.game.get_current_player(), "color").value
                    if current_color == seat_color:
                        mask = None
                        if hasattr(env, "move_utils"):
                            mask = env.move_utils.action_masks(env._pending_valid_moves)  # type: ignore
                        action = _soft_action_select(policy, obs, mask)
                    else:
                        # Let scripted strategy handle its own move (baseline env)
                        action = env.game.get_current_player().strategy.select_action(None)  # type: ignore
                    obs, reward, terminated, truncated, info = env.step(action)
                    turns += 1
                    if info.get("step_breakdown"):
                        comp = info["step_breakdown"]
                        if "capture" in comp and comp["capture"] > 0:
                            metrics.captures_for += 1
                        if "got_captured" in comp and comp["got_captured"] < 0:
                            metrics.captures_against += 1
                        if info.get("illegal_action"):
                            metrics.illegal_actions += 1
                    if terminated or truncated:
                        # Determine rank / winner
                        # Simple approach: if agent won allocate win
                        agent_player = next(p for p in env.game.players if p.color.value == seat_color)
                        if agent_player.has_won():
                            metrics.wins += 1
                            metrics.ranks.append(1)
                        else:
                            # Rough rank proxy: order by finished token count descending (ties average rank)
                            finished = [(p.get_finished_tokens_count(), p) for p in env.game.players]
                            finished.sort(reverse=True, key=lambda x: x[0])
                            rank_positions = {pl.color.value: i + 1 for i, (cnt, pl) in enumerate(finished)}
                            metrics.ranks.append(rank_positions[seat_color])
                            metrics.losses += 1
                        metrics.turns.append(turns)
                        break
        # Aggregate & log
        agg = metrics.aggregate()
        for k, v in agg.items():
            self.logger.record(self.log_prefix + k, v)
        if self.verbose:
            print(f"[Tournament] Steps={self.num_timesteps} games={total_games_target} metrics={agg}")

        # Ensure TensorBoard flush
        if hasattr(self.logger, "writer") and self.logger.writer:
            self.logger.writer.flush()

        return agg

    def _on_training_end(self) -> None:
        # Final tournament at end
        self._run_tournament()
        
