import unittest
from pathlib import Path

import torch
from ludo_engine.models import Colors, GameConstants

from ludo_rl.ppo_strategy import EnvConfig, PPOStrategy


class DummyPolicy:
    class Dist:
        class DistInner:
            def __init__(self, probs):
                self.probs = probs
                self.logits = torch.log(probs + 1e-8)

        def __init__(self):
            self.distribution = self.DistInner(
                torch.ones(GameConstants.TOKENS_PER_PLAYER)
                / GameConstants.TOKENS_PER_PLAYER
            )

    def get_distribution(self, obs_tensor):  # pragma: no cover - trivial
        return self.Dist()


class DummyModel:
    def __init__(self):
        self.policy = DummyPolicy()

    @staticmethod
    def load(path):  # pragma: no cover - simple factory
        return DummyModel()


class TestPPOStrategy(unittest.TestCase):
    def test_strategy_decide_valid_action(self):
        # Monkeypatch PPO.load directly in the base_ppo_strategy module for testing
        import rl_base.strategies.base_ppo_strategy as base_ppo_module

        # Provide a wrapper that tolerates extra kwargs like device
        def _dummy_load(path, **kwargs):  # pragma: no cover - simple adapter
            return DummyModel.load(path)

        base_ppo_module.PPO.load = _dummy_load  # type: ignore
        cfg = EnvConfig(agent_color=Colors.RED, max_turns=10)
        model_path = Path("dummy_model.zip")
        model_path.write_bytes(b"")
        strat = PPOStrategy(str(model_path), "dummy", cfg)
        ctx = {
            "players": [
                {
                    "color": Colors.RED,
                    "tokens": [{"position": -1} for _ in range(4)],
                    "finished_tokens": 0,
                },
                {
                    "color": Colors.GREEN,
                    "tokens": [{"position": -1} for _ in range(4)],
                    "finished_tokens": 0,
                },
                {
                    "color": Colors.YELLOW,
                    "tokens": [{"position": -1} for _ in range(4)],
                    "finished_tokens": 0,
                },
                {
                    "color": Colors.BLUE,
                    "tokens": [{"position": -1} for _ in range(4)],
                    "finished_tokens": 0,
                },
            ],
            "game_info": {
                "current_player": Colors.RED,
                "dice_value": 3,
                "turn_count": 0,
            },
            "valid_moves": [{"token_id": 0}],
        }
        action = strat.decide(ctx)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < GameConstants.TOKENS_PER_PLAYER)
        # Cleanup created dummy model file
        try:
            if model_path.exists():
                model_path.unlink()
        except Exception:  # pragma: no cover - best effort cleanup
            pass


if __name__ == "__main__":
    unittest.main()
