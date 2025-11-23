import os
from typing import List

import pytest

from ludo_rl.ludo_env import LudoEnv
from ludo_rl.ludo_king.config import config as king_config


class LogCapture:
    def __init__(self):
        self.records: List[str] = []

    def __call__(self, message):
        # message is a loguru Message object; stringified includes level/time
        txt = str(message)
        if "Opponent lineup for reset" in txt:
            self.records.append(txt)


@pytest.fixture(autouse=True)
def preserve_env():
    prev = os.environ.get("OPPONENTS")
    try:
        os.environ["OPPONENTS"] = "random,killer,defensive"
        yield
    finally:
        if prev is None:
            os.environ.pop("OPPONENTS", None)
        else:
            os.environ["OPPONENTS"] = prev


def test_logging_only_at_zero_with_fixed_cache(monkeypatch):
    # Arrange: fixed caching enabled, steps not aligned with curriculum
    env = LudoEnv(use_fixed_opponents=True)
    # Small curriculum to make boundaries frequent
    env.curriculum_interval_resets = 5
    # Large fixed step so modulo != 0 for counts 1..N (no recompute)
    monkeypatch.setattr(king_config, "FIXED_OPPONENTS_STEPS", 1000, raising=False)

    from loguru import logger

    sink = LogCapture()
    sink_id = logger.add(sink, level="INFO")
    try:
        # Act: call _get_lineup over a range of reset counts
        for c in range(0, 21):
            env._reset_count = c
            env._get_lineup(num_opponents=3)

        # Assert: only initial (0) produced a log due to cache short-circuit
        assert len(sink.records) == 1, (
            f"expected one log at reset 0, got {len(sink.records)}"
        )
        assert "reset 0" in sink.records[0]
    finally:
        logger.remove(sink_id)


def test_logging_at_curriculum_intervals_without_fixed_cache(monkeypatch):
    # Arrange: disable fixed caching to allow logging at curriculum boundaries
    env = LudoEnv(use_fixed_opponents=False)
    env.curriculum_interval_resets = 5
    # Ensure fixed steps do not interfere
    monkeypatch.setattr(king_config, "FIXED_OPPONENTS_STEPS", 0, raising=False)

    from loguru import logger

    sink = LogCapture()
    sink_id = logger.add(sink, level="INFO")
    try:
        # Act
        for c in range(0, 21):
            env._reset_count = c
            env._get_lineup(num_opponents=3)

        # Assert: current code logs at 0, 5, 10, 15 only
        found = "\n".join(sink.records)
        for mark in (0, 5, 10, 15):
            assert f"reset {mark}" in found, (
                f"missing log at curriculum boundary {mark}"
            )
    finally:
        logger.remove(sink_id)


def test_logging_when_fixed_and_curriculum_align(monkeypatch):
    # Arrange: fixed caching enabled AND aligned with curriculum
    env = LudoEnv(use_fixed_opponents=True)
    env.curriculum_interval_resets = 5
    monkeypatch.setattr(king_config, "FIXED_OPPONENTS_STEPS", 5, raising=False)

    from loguru import logger

    sink = LogCapture()
    sink_id = logger.add(sink, level="INFO")
    try:
        for c in range(0, 21):
            env._reset_count = c
            env._get_lineup(num_opponents=3)

        found = "\n".join(sink.records)
        # Assert: with alignment, logs at 0, 5, 10, 15 (matching code)
        for mark in (0, 5, 10, 15):
            assert f"reset {mark}" in found, f"expected log at aligned boundary {mark}"
    finally:
        logger.remove(sink_id)
