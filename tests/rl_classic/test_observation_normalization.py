import os
import unittest

import numpy as np
from ludo_engine.core import LudoGame, PlayerColor

from ludo_rl.ppo_strategy import EnvConfig, PPOStrategy

MODEL_DIR = "models/ppo_training"
MODEL_FILE = "best_model.zip"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)


class TestObservationNormalization(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(MODEL_PATH), "Model file not present")
    def test_obs_normalization_bounds(self):
        strat = PPOStrategy(MODEL_PATH, "best", EnvConfig(), deterministic=True)
        if strat.obs_rms is None:
            self.skipTest("No VecNormalize stats available")
        game = LudoGame(
            [
                PlayerColor.RED,
                PlayerColor.GREEN,
                PlayerColor.YELLOW,
                PlayerColor.BLUE,
            ]
        )
        dice_values = [6, 3, 4, 2, 5, 1]
        collected = []
        for d in dice_values:
            ctx = game.get_ai_decision_context(d)
            # Inject context through strategy internal method by calling decide which builds obs
            # We reproduce builder call directly for measuring normalization effect
            turn_count = (
                ctx["current_situation"]["turn_count"]
                if "current_situation" in ctx
                else ctx["game_info"]["turn_count"]
            )
            # Build observation the same way strategy does
            strat._inject_context_into_dummy_game(ctx)  # type: ignore
            obs = strat.obs_builder._build_observation(turn_count, d)
            if strat.obs_rms is not None:
                obs = (obs - strat.obs_rms.mean) / np.sqrt(strat.obs_rms.var + 1e-8)
                obs = np.clip(obs, -10.0, 10.0)
            collected.append(obs)
        arr = np.stack(collected)
        # Mean should be roughly centered (not strictly zero) but within reasonable bounds
        self.assertTrue(np.all(np.abs(arr.mean(axis=0)) < 5.0))
        # Values should respect clipping range
        self.assertLessEqual(arr.max(), 10.0)
        self.assertGreaterEqual(arr.min(), -10.0)


if __name__ == "__main__":
    unittest.main()
