import os

import pytest

from ludo_rl.ludo_env import LudoEnv
from ludo_rl.ludo_king.config import config as king_config


@pytest.fixture(autouse=True)
def preset_env():
    prev = os.environ.get("OPPONENTS")
    try:
        # Keep pool simple and deterministic
        os.environ["OPPONENTS"] = "random,killer,defensive"
        yield
    finally:
        if prev is None:
            os.environ.pop("OPPONENTS", None)
        else:
            os.environ["OPPONENTS"] = prev


def collect_lineup_ids(env: LudoEnv, step: int, max_c: int):
    ids = []
    for c in range(0, max_c + 1):
        env._reset_count = c
        lineup = env._get_lineup(num_opponents=3)
        ids.append(id(lineup))
    return ids


def test_fixed_cache_updates_every_k_steps(monkeypatch):
    env = LudoEnv(use_fixed_opponents=True)
    # Make curriculum very large so it doesn't add candidates and affect behavior
    env.curriculum_interval_resets = 10_000_000
    monkeypatch.setattr(king_config, "FIXED_OPPONENTS_STEPS", 3, raising=False)

    ids = collect_lineup_ids(env, step=3, max_c=10)

    # 0,1,2 same id; 3 new id; 3,4,5 same; 6 new id; etc.
    assert ids[0] == ids[1] == ids[2]
    assert ids[3] != ids[2]
    assert ids[3] == ids[4] == ids[5]
    assert ids[6] != ids[5]
    assert ids[6] == ids[7] == ids[8]
    assert ids[9] != ids[8]
    assert ids[9] == ids[10]


def test_fixed_cache_updates_every_step_when_k_is_one(monkeypatch):
    env = LudoEnv(use_fixed_opponents=True)
    env.curriculum_interval_resets = 10_000_000
    monkeypatch.setattr(king_config, "FIXED_OPPONENTS_STEPS", 1, raising=False)

    ids = collect_lineup_ids(env, step=1, max_c=6)

    # Every reset recomputes, thus different ids each time after first
    assert ids[0] != ids[1] != ids[2] != ids[3] != ids[4] != ids[5] != ids[6]


def test_fixed_cache_does_not_update_between_boundaries(monkeypatch):
    env = LudoEnv(use_fixed_opponents=True)
    env.curriculum_interval_resets = 10_000_000
    monkeypatch.setattr(king_config, "FIXED_OPPONENTS_STEPS", 4, raising=False)

    ids = collect_lineup_ids(env, step=4, max_c=7)

    # 0..3 same id; 4 new id; 5..7 share with 4
    assert len(set(ids[0:4])) == 1
    assert ids[4] != ids[3]
    assert len(set(ids[4:8])) == 1
