import unittest
from pathlib import Path

import torch
from ludo_engine.models import Colors, GameConstants

from ludo_rl.ppo_strategy import EnvConfig, PPOStrategy
from ludo_engine.models import (
    AIDecisionContext,
    CurrentSituation,
    PlayerState,
    OpponentInfo,
    ValidMove,
    StrategicAnalysis,
    TokenInfo,
)

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
        ctx = AIDecisionContext(
            current_situation=CurrentSituation(
                player_color=Colors.RED,
                dice_value=3,
                consecutive_sixes=0,
                turn_count=0,
            ),
            player_state=PlayerState(
                player_id=0,
                color=Colors.RED,
                start_position=0,
                tokens=[
                    TokenInfo(
                        token_id=i,
                        player_color=Colors.RED,
                        state="home",
                        position=-1,
                        is_in_home=True,
                        is_active=False,
                        is_in_home_column=False,
                        is_finished=False,
                    )
                    for i in range(4)
                ],
                tokens_in_home=4,
                active_tokens=0,
                tokens_in_home_column=0,
                finished_tokens=0,
                has_won=False,
                positions_occupied=[],
            ),
            opponents=[
                OpponentInfo(
                    color=color,
                    finished_tokens=0,
                    tokens_active=0,
                    threat_level=0.0,
                    positions_occupied=[],
                )
                for color in [Colors.GREEN, Colors.YELLOW, Colors.BLUE]
            ],
            valid_moves=[
                ValidMove(
                    token_id=0,
                    current_position=-1,
                    current_state="home",
                    target_position=3,
                    move_type="exit_home",
                    is_safe_move=False,
                    captures_opponent=False,
                    captured_tokens=[],
                    strategic_value=0.0,
                    strategic_components={},
                )
            ],
            strategic_analysis=StrategicAnalysis(
                can_capture=False,
                can_finish_token=False,
                can_exit_home=True,
                safe_moves=[],
                risky_moves=[],
                best_strategic_move=None,
            ),
        )
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
