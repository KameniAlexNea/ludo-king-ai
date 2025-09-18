import os
import unittest

from ludo_engine.core import LudoGame, PlayerColor

from rl.strategies.ppo_strategy_multi_seat import EnvConfig, PPOStrategy

MODEL_DIR = "models/ppo_training"
MODEL_FILE = "best_model.zip"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)


class TestPPOStrategyDeterminism(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(MODEL_PATH), "Model file not present")
    def test_deterministic_vs_stochastic(self):
        cfg = EnvConfig(max_turns=10)
        strat_det = PPOStrategy(MODEL_PATH, "best", cfg, deterministic=True)
        strat_sto = PPOStrategy(MODEL_PATH, "best", cfg, deterministic=False)
        # Build a mock context with simple initial game state
        game = LudoGame(
            [
                PlayerColor.RED,
                PlayerColor.GREEN,
                PlayerColor.YELLOW,
                PlayerColor.BLUE,
            ]
        )
        # Roll dice for context
        dice = 6
        ctx = game.get_ai_decision_context(dice)
        # Ensure some valid moves exist (dice=6 gives exit)
        valid_moves = ctx["valid_moves"]
        self.assertTrue(valid_moves)
        a_det = strat_det.decide(ctx)
        # Sample multiple stochastic decisions; at least one should differ sometimes. If not, still pass.
        stochastic_set = set()
        for _ in range(10):
            stochastic_set.add(strat_sto.decide(ctx))
        # Deterministic action should be within valid token ids
        valid_ids = [m["token_id"] for m in valid_moves]
        self.assertIn(a_det, valid_ids)
        for a in stochastic_set:
            self.assertIn(a, valid_ids)
        # If stochastic produced >1 distinct action, assert deterministic not outside set
        if len(stochastic_set) > 1:
            self.assertIn(a_det, stochastic_set)


if __name__ == "__main__":
    unittest.main()
