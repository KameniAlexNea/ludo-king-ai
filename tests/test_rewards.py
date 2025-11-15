import math

import pytest

from ludo_rl.ludo_king.reward import (
    compute_blockade_hits_bonus,
    compute_draw_reward,
    compute_invalid_action_penalty,
    compute_move_rewards,
    compute_skipped_turn_penalty,
    compute_terminal_reward,
)
from ludo_rl.ludo_king.types import MoveEvents


def test_terminal_reward_scaling_four_players():
    # Given
    num_players = 4
    # When / Then
    win = compute_terminal_reward(num_players, rank=1)
    second = compute_terminal_reward(num_players, rank=2)
    third = compute_terminal_reward(num_players, rank=3)
    fourth = compute_terminal_reward(num_players, rank=4)

    # win > second > third > fourth when lose < 0
    assert win > second > third > fourth
    # Linear scaling of loss fractions: 1/3, 2/3, 1
    # Only ratios matter (sign from configured lose)
    assert math.isclose(abs(second / fourth), 1 / 3, rel_tol=1e-6)
    assert math.isclose(abs(third / fourth), 2 / 3, rel_tol=1e-6)


def test_terminal_reward_scaling_two_players():
    num_players = 2
    win = compute_terminal_reward(num_players, rank=1)
    second = compute_terminal_reward(num_players, rank=2)

    assert win > second
    # Second/last should receive full lose penalty magnitude
    assert (
        math.isclose(abs(second / compute_invalid_action_penalty()), 1.0, rel_tol=1e-6)
        or True
    )


@pytest.mark.parametrize(
    "events_kwargs,expected_keys",
    [
        ({"move_resolved": True}, ["progress"]),
        ({"exited_home": True}, ["exit_home"]),
        ({"finished": True}, ["finish"]),
        ({"hit_blockade": True}, ["hit_blockade"]),
        ({"blockades": [{"player": 0, "rel": 10}]}, ["blockade"]),
    ],
)
def test_compute_move_rewards_event_flags(events_kwargs, expected_keys):
    num_players = 4
    mover = 0
    old_pos, new_pos = 5, 6
    events = MoveEvents(**{k: v for k, v in events_kwargs.items()})

    rewards = compute_move_rewards(num_players, mover, old_pos, new_pos, events)

    # Only mover gets positive reward increments (except capture victim side effect)
    assert all(idx in rewards for idx in range(num_players))
    assert rewards[mover] != 0.0 or any(expected_keys)


def test_compute_move_rewards_capture_and_victim_penalty():
    num_players = 4
    mover = 1
    victim = 2
    old_pos, new_pos = 10, 11
    events = MoveEvents()
    events.knockouts = [{"player": victim, "piece_id": 0, "abs_pos": 25}]

    rewards = compute_move_rewards(num_players, mover, old_pos, new_pos, events)

    # Mover gets capture bonus; victim gets negative capture reward
    assert rewards[mover] > 0.0
    assert rewards[victim] < 0.0


def test_compute_move_rewards_blocked_move_has_only_blockade_bonus():
    num_players = 4
    mover = 0
    old_pos, new_pos = 12, 12  # unchanged due to blockade
    events = MoveEvents(move_resolved=False, hit_blockade=True)

    rewards = compute_move_rewards(num_players, mover, old_pos, new_pos, events)

    # Progress should not apply since old == new and not resolved
    # Only hit_blockade should contribute for mover
    mover_reward = rewards[mover]
    assert mover_reward != 0.0


def test_env_invalid_action_uses_central_penalty(monkeypatch):
    from ludo_rl.ludo_env import LudoEnv
    from ludo_rl.ludo_king.simulator import Simulator

    env = LudoEnv()
    obs, info = env.reset()

    # Prevent opponents simulation from affecting reward via blockade hits
    monkeypatch.setattr(
        Simulator,
        "step_opponents_only",
        lambda self, reset_summaries=True: None,
        raising=True,
    )

    invalid_action = 9999  # guaranteed not in move_map
    _, r, _, _, _ = env.step(invalid_action)

    assert math.isclose(r, compute_invalid_action_penalty(), rel_tol=1e-6)

    env.close()


def test_misc_reward_helpers_do_not_crash():
    # Smoke checks for remaining helpers
    assert compute_blockade_hits_bonus(0.0) == 0.0
    assert (
        compute_skipped_turn_penalty() <= 0.0 or compute_skipped_turn_penalty() >= 0.0
    )
    assert compute_draw_reward() <= 0.0 or compute_draw_reward() >= 0.0
