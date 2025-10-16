from typing import Optional, Sequence

from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from ludo_rl.config import EnvConfig
from ludo_rl.utils.opponents import build_opponent_combinations


class SimpleBaselineEvalCallback(MaskableEvalCallback):
    """Periodically evaluate the current policy vs fixed baselines.

    Plays 1v3 games where the agent always sits in one seat and the other 3
    seats are filled by scripted strategies specified in `baselines`. We run
    a number of games and compute win rate and average turns. Results are
    logged to TensorBoard.

    Notes:
    - Uses a separate eval env with VecNormalize sharing obs_rms for parity.
    - Sampling tries different opponent order permutations for diversity.
    - Keeps it simple: no ranks, just win/lose rate.
    """

    def __init__(
        self,
        baselines: Sequence[str],
        n_games: int = 60,
        eval_freq: int = 100_000,
        verbose: int = 0,
        env_cfg: Optional[EnvConfig] = None,
        eval_env: VecNormalize = None,
        best_model_save_path: Optional[str] = None,
        callback_on_new_best=None,
        callback_after_eval=None,
        log_path: Optional[str] = None,
    ):
        self.n_games = int(n_games)
        self.env_cfg = env_cfg or EnvConfig()
        self.setups = self.env_cfg.allowed_player_counts

        if eval_env is None:
            raise ValueError("eval_env must be provided")

        self.eval_env = eval_env

        # Initialize parent with eval_env and other parameters
        super().__init__(
            eval_env=self.eval_env,
            n_eval_episodes=self.n_games,  # Use n_games for evaluation episodes
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=False,
            render=False,
            verbose=verbose,
            warn=True,
            use_masking=True,  # Enable masking for Ludo
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
        )
        # Pre-build a list of opponent combinations per setup size with the correct
        # number of opponents (players-1). Example: 2-player -> 1 opponent, 4-player -> 3 opponents.
        self.baselines_per_setup = []
        for setup in self.setups:
            n_comb = max(0, int(setup) - 1)
            combos = build_opponent_combinations(
                list(baselines), n_games=self.n_games, n_comb=n_comb
            )
            self.baselines_per_setup.append(combos)
        self.executed = 0

    def _on_step(self) -> bool:
        # Evaluate every eval_freq steps
        if not (self.eval_freq > 0 and self.n_calls % self.eval_freq == 0):
            return True
        total_setups = len(self.setups)
        setup_idx = (self.executed // self.n_games) % max(1, total_setups)
        setup = self.setups[setup_idx]
        idx_in_setup = self.executed % self.n_games
        opponents = self.baselines_per_setup[setup_idx][idx_in_setup]

        continue_training = True

        # Set attributes on the eval env for this setup
        self.eval_env.set_attr("opponents", opponents)
        self.eval_env.set_attr("fixed_num_players", setup)

        # Call parent _on_step to do the standard evaluation and logging
        continue_training = super()._on_step()

        self.eval_env.set_attr("opponents", None)
        self.eval_env.set_attr("fixed_num_players", None)
        self.executed += 1
        return continue_training
