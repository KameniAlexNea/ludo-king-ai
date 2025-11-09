from __future__ import annotations

import random
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from ludo_rl.ludo.config import config, strategy_config
from ludo_rl.strategy import (
    CautiousStrategy,
    DefensiveStrategy,
    FinishLineStrategy,
    HeatSeekerStrategy,
    HoarderStrategy,
    HomebodyStrategy,
    KillerStrategy,
    LLMStrategy,
    ProbabilityStrategy,
    RetaliatorStrategy,
    RLStrategy,
    RusherStrategy,
    SupportStrategy,
)
from ludo_rl.strategy.llm_agent import DEFAULT_SYSTEM_PROMPT
from ludo_rl.strategy.types import MoveOption, StrategyContext


def _make_board() -> np.ndarray:
    return np.zeros((5, config.PATH_LENGTH), dtype=float)


def _make_context(
    moves: list[MoveOption],
    board: np.ndarray | None = None,
    dice_roll: int = 6,
    action_mask: np.ndarray | None = None,
) -> StrategyContext:
    board = board if board is not None else _make_board()
    if action_mask is None:
        action_mask = np.array([True, True, False, False], dtype=bool)

    opponent_distribution = board[
        strategy_config.board_channel_opp_start : strategy_config.board_channel_opp_end
        + 1
    ].sum(axis=0)

    return StrategyContext(
        board=board,
        dice_roll=dice_roll,
        action_mask=action_mask,
        moves=moves,
        my_distribution=board[strategy_config.board_channel_my],
        opponent_distribution=opponent_distribution,
        safe_channel=board[strategy_config.board_channel_safe],
    )


def _move(
    *,
    piece_id: int,
    current: int,
    new: int,
    progress: int,
    distance: int,
    can_capture: bool = False,
    capture_count: int = 0,
    enters_home: bool = False,
    enters_safe_zone: bool = False,
    forms_blockade: bool = False,
    extra_turn: bool = False,
    risk: float = 0.0,
    leaving_safe_zone: bool = False,
) -> MoveOption:
    return MoveOption(
        piece_id=piece_id,
        current_pos=current,
        new_pos=new,
        dice_roll=6,
        progress=progress,
        distance_to_goal=distance,
        can_capture=can_capture,
        capture_count=capture_count,
        enters_home=enters_home,
        enters_safe_zone=enters_safe_zone,
        forms_blockade=forms_blockade,
        extra_turn=extra_turn,
        risk=risk,
        leaving_safe_zone=leaving_safe_zone,
    )


class StrategySelectionTests(unittest.TestCase):
    def test_probability_prefers_safer_move(self) -> None:
        moves = [
            _move(
                piece_id=0,
                current=5,
                new=11,
                progress=6,
                distance=40,
                enters_safe_zone=True,
            ),
            _move(piece_id=1, current=5, new=15, progress=10, distance=36, risk=3.0),
        ]
        ctx = _make_context(moves)
        choice = ProbabilityStrategy().select_move(ctx)
        self.assertEqual(choice.piece_id, 0)

    def test_cautious_stays_safe(self) -> None:
        moves = [
            _move(
                piece_id=0,
                current=12,
                new=16,
                progress=4,
                distance=35,
                enters_safe_zone=True,
            ),
            _move(
                piece_id=1,
                current=18,
                new=22,
                progress=4,
                distance=29,
                risk=2.5,
                leaving_safe_zone=True,
            ),
        ]
        ctx = _make_context(moves)
        choice = CautiousStrategy().select_move(ctx)
        self.assertEqual(choice.piece_id, 0)

    def test_killer_prioritises_capture(self) -> None:
        moves = [
            _move(
                piece_id=0,
                current=8,
                new=13,
                progress=5,
                distance=44,
                can_capture=True,
                capture_count=1,
            ),
            _move(piece_id=1, current=12, new=18, progress=6, distance=39, risk=0.5),
        ]
        ctx = _make_context(moves)
        choice = KillerStrategy().select_move(ctx)
        self.assertEqual(choice.piece_id, 0)

    def test_defensive_prefers_safety(self) -> None:
        moves = [
            _move(
                piece_id=0,
                current=25,
                new=30,
                progress=5,
                distance=27,
                enters_safe_zone=True,
            ),
            _move(
                piece_id=1,
                current=32,
                new=36,
                progress=4,
                distance=21,
                leaving_safe_zone=True,
                risk=1.0,
            ),
        ]
        ctx = _make_context(moves)
        choice = DefensiveStrategy().select_move(ctx)
        self.assertEqual(choice.piece_id, 0)

    def test_finish_line_closes_game(self) -> None:
        moves = [
            _move(
                piece_id=0, current=54, new=57, progress=3, distance=0, enters_home=True
            ),
            _move(
                piece_id=1,
                current=50,
                new=55,
                progress=5,
                distance=2,
                enters_safe_zone=True,
            ),
        ]
        ctx = _make_context(moves)
        choice = FinishLineStrategy().select_move(ctx)
        self.assertEqual(choice.piece_id, 0)

    def test_hoarder_favors_blockade(self) -> None:
        moves = [
            _move(
                piece_id=0,
                current=5,
                new=8,
                progress=3,
                distance=49,
                forms_blockade=True,
            ),
            _move(piece_id=1, current=10, new=15, progress=5, distance=42),
        ]
        ctx = _make_context(moves)
        choice = HoarderStrategy().select_move(ctx)
        self.assertEqual(choice.piece_id, 0)

    def test_homebody_values_safe_home_path(self) -> None:
        moves = [
            _move(
                piece_id=0,
                current=45,
                new=52,
                progress=7,
                distance=5,
                enters_safe_zone=True,
            ),
            _move(
                piece_id=1,
                current=30,
                new=34,
                progress=4,
                distance=23,
                leaving_safe_zone=True,
                risk=1.2,
            ),
        ]
        ctx = _make_context(moves)
        choice = HomebodyStrategy().select_move(ctx)
        self.assertEqual(choice.piece_id, 0)

    def test_heatseeker_moves_toward_opponents(self) -> None:
        board = _make_board()
        board[1, 11] = 1.0  # Opponent near square 11
        board[2, 40] = 1.0
        moves = [
            _move(piece_id=0, current=6, new=10, progress=4, distance=47),
            _move(piece_id=1, current=20, new=26, progress=6, distance=31),
        ]
        ctx = _make_context(moves, board=board)
        choice = HeatSeekerStrategy().select_move(ctx)
        self.assertEqual(choice.piece_id, 0)

    def test_retaliator_prefers_engagement(self) -> None:
        board = _make_board()
        board[1, 18] = 1.0
        moves = [
            _move(
                piece_id=0,
                current=12,
                new=18,
                progress=6,
                distance=39,
                can_capture=True,
                capture_count=1,
            ),
            _move(piece_id=1, current=22, new=26, progress=4, distance=31),
        ]
        ctx = _make_context(moves, board=board)
        choice = RetaliatorStrategy().select_move(ctx)
        self.assertEqual(choice.piece_id, 0)

    def test_rusher_chases_progress(self) -> None:
        moves = [
            _move(
                piece_id=0, current=0, new=11, progress=11, distance=46, extra_turn=True
            ),
            _move(piece_id=1, current=10, new=14, progress=4, distance=33),
        ]
        ctx = _make_context(moves)
        choice = RusherStrategy().select_move(ctx)
        self.assertEqual(choice.piece_id, 0)

    def test_support_levels_team_development(self) -> None:
        board = _make_board()
        board[0, 0] = 1.0  # Yard representation
        board[0, 20] = 1.0
        moves = [
            _move(
                piece_id=0, current=0, new=6, progress=6, distance=51, extra_turn=True
            ),
            _move(piece_id=1, current=20, new=24, progress=4, distance=33),
        ]
        ctx = _make_context(moves, board=board)
        choice = SupportStrategy().select_move(ctx)
        self.assertEqual(choice.piece_id, 0)

    def test_create_instance_builds_strategy(self) -> None:
        strategy_classes = [
            CautiousStrategy,
            DefensiveStrategy,
            FinishLineStrategy,
            HeatSeekerStrategy,
            HoarderStrategy,
            HomebodyStrategy,
            KillerStrategy,
            ProbabilityStrategy,
            RetaliatorStrategy,
            RusherStrategy,
            SupportStrategy,
        ]

        for idx, strategy_cls in enumerate(strategy_classes):
            with self.subTest(strategy=strategy_cls.__name__):
                instance = strategy_cls.create_instance(random.Random(idx))
                self.assertIsInstance(instance, strategy_cls)


class RLStrategyTests(unittest.TestCase):
    def tearDown(self) -> None:
        RLStrategy.config = None

    def test_create_without_configuration_errors(self) -> None:
        RLStrategy.config = None
        with self.assertRaises(NotImplementedError):
            RLStrategy.create_instance()

    def test_select_move_uses_model_prediction(self) -> None:
        class DummyModel:
            def __init__(self) -> None:
                self.policy = SimpleNamespace(set_training_mode=lambda *_: None)

            def predict(self, obs, action_masks=None, deterministic=True):
                legal = np.flatnonzero(action_masks)
                return np.array([int(legal[-1])]), None

        strategy = RLStrategy.configure(model=DummyModel(), deterministic=True)

        moves = [
            _move(piece_id=0, current=0, new=6, progress=6, distance=51),
            _move(piece_id=1, current=0, new=7, progress=7, distance=50),
        ]
        ctx = _make_context(moves, action_mask=np.array([True, True, False, False]))
        choice = strategy.select_move(ctx)
        self.assertIsNotNone(choice)
        self.assertEqual(choice.piece_id, 1)

    def test_configure_from_path_loads_model(self) -> None:
        dummy_model = SimpleNamespace(
            policy=SimpleNamespace(set_training_mode=lambda *_: None)
        )
        with mock.patch(
            "ludo_rl.strategy.rl_agent.MaskablePPO.load", return_value=dummy_model
        ) as loader:
            strategy = RLStrategy.configure_from_path("/tmp/model.zip", device="cpu")
        loader.assert_called_once_with("/tmp/model.zip", device="cpu")
        self.assertIsInstance(strategy, RLStrategy)


class LLMStrategyTests(unittest.TestCase):
    def tearDown(self) -> None:
        LLMStrategy.config = None

    def test_create_without_configuration_errors(self) -> None:
        with self.assertRaises(NotImplementedError):
            LLMStrategy.create_instance()

    def test_select_move_parses_json_response(self) -> None:
        class FakeChatModel:
            def __init__(self) -> None:
                self.invocations = 0

            def invoke(self, messages):
                self.invocations += 1
                return {"content": '{"piece_id": 1, "reason": "capture"}'}

        strategy = LLMStrategy.configure(FakeChatModel())

        moves = [
            _move(piece_id=0, current=0, new=6, progress=6, distance=51),
            _move(piece_id=1, current=0, new=7, progress=7, distance=50),
        ]
        ctx = _make_context(moves, action_mask=np.array([True, True, False, False]))
        choice = strategy.select_move(ctx)
        self.assertEqual(choice.piece_id, 1)

    def test_select_move_retries_and_falls_back(self) -> None:
        class FailingChatModel:
            def __init__(self) -> None:
                self.outputs = iter(
                    [
                        {"content": "I choose piece A"},
                        {"content": "No idea"},
                    ]
                )

            def invoke(self, messages):
                try:
                    return next(self.outputs)
                except StopIteration:
                    raise RuntimeError("exhausted")

        strategy = (
            LLMStrategy.configure(  # two retries => fall back to first legal move
                model=FailingChatModel(),
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                max_retries=1,
            )
        )

        moves = [
            _move(piece_id=2, current=5, new=9, progress=4, distance=48),
            _move(piece_id=3, current=10, new=14, progress=4, distance=43),
        ]
        action_mask = np.array([False, False, True, True])
        ctx = _make_context(moves, action_mask=action_mask)
        choice = strategy.select_move(ctx)
        self.assertEqual(choice.piece_id, 2)

    def test_configure_with_model_name_uses_init_chat_model(self) -> None:
        fake_model = SimpleNamespace(invoke=lambda *_: {"content": '{"piece_id": 0}'})

        with mock.patch(
            "ludo_rl.strategy.llm_agent.init_chat_model", return_value=fake_model
        ) as init_mock:
            strategy = LLMStrategy.configure_with_model_name(
                "gpt-5-nano",
                system_prompt="Test",
                temperature=0.2,
            )

        init_mock.assert_called_once_with("gpt-5-nano", temperature=0.2)
        self.assertIsInstance(strategy, LLMStrategy)


if __name__ == "__main__":
    unittest.main()
