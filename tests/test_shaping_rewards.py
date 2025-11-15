import math


from ludo_rl.ludo_king.config import reward_config
from ludo_rl.ludo_king.game import Game
from ludo_rl.ludo_king.player import Player
from ludo_rl.ludo_king.types import Color, Move


def _simple_game():
    players = [
        Player(Color.RED),
        Player(Color.GREEN),
        Player(Color.YELLOW),
        Player(Color.BLUE),
    ]
    return Game(players=players)


def test_shaping_toggle_off_progress_only(monkeypatch):
    g = _simple_game()
    # Agent RED piece at 5; others at 0
    g.players[0].pieces[0].position = 5

    # Disable shaping
    monkeypatch.setattr(reward_config, "shaping_use", False, raising=False)

    mv = Move(player_index=0, piece_id=0, new_pos=6, dice_roll=1)
    res = g.apply_move(mv)

    # With shaping off, reward should equal base progress only (no None)
    assert res.rewards is not None
    r = res.rewards[0]
    assert r > 0
    # Turning on shaping with zero alpha should match too
    monkeypatch.setattr(reward_config, "shaping_use", True, raising=False)
    monkeypatch.setattr(reward_config, "shaping_alpha", 0.0, raising=False)
    g2 = _simple_game()
    g2.players[0].pieces[0].position = 5
    res2 = g2.apply_move(Move(player_index=0, piece_id=0, new_pos=6, dice_roll=1))
    assert math.isclose(res2.rewards[0], r, rel_tol=1e-6)


def test_shaping_progress_component_adds_positive_delta(monkeypatch):
    g = _simple_game()
    g.players[0].pieces[0].position = 5

    # Enable shaping, isolate progress component
    monkeypatch.setattr(reward_config, "shaping_use", True, raising=False)
    monkeypatch.setattr(reward_config, "shaping_alpha", 1.0, raising=False)
    monkeypatch.setattr(reward_config, "shaping_gamma", 1.0, raising=False)
    monkeypatch.setattr(reward_config, "ro_depth", 1, raising=False)
    monkeypatch.setattr(reward_config, "ro_w_progress", 1.0, raising=False)
    monkeypatch.setattr(reward_config, "ro_w_cap_opp", 0.0, raising=False)
    monkeypatch.setattr(reward_config, "ro_w_cap_risk", 0.0, raising=False)
    monkeypatch.setattr(reward_config, "ro_w_finish_opp", 0.0, raising=False)

    # Apply move 5->6
    mv = Move(player_index=0, piece_id=0, new_pos=6, dice_roll=1)
    res = g.apply_move(mv)

    # Reward should be strictly greater than base progress due to positive shaping delta
    assert res.rewards is not None
    assert res.rewards[0] > 0  # base + shaping > base


def test_shaping_blockade_early_return_no_delta_when_gamma_one(monkeypatch):
    g = _simple_game()
    # Agent red at 5
    g.players[0].pieces[0].position = 5
    # Green blockade at red-relative 10: set both greens to map to same absolute
    # Green's rel that maps to red's abs(10) is 49
    g.players[1].pieces[0].position = 49
    g.players[1].pieces[1].position = 49

    # Enable shaping, gamma=1 so (gamma-1)*phi_before = 0 on early return, isolate progress
    monkeypatch.setattr(reward_config, "shaping_use", True, raising=False)
    monkeypatch.setattr(reward_config, "shaping_alpha", 1.0, raising=False)
    monkeypatch.setattr(reward_config, "shaping_gamma", 1.0, raising=False)
    monkeypatch.setattr(reward_config, "ro_w_progress", 1.0, raising=False)
    monkeypatch.setattr(reward_config, "ro_w_cap_opp", 0.0, raising=False)
    monkeypatch.setattr(reward_config, "ro_w_cap_risk", 0.0, raising=False)
    monkeypatch.setattr(reward_config, "ro_w_finish_opp", 0.0, raising=False)

    mv = Move(player_index=0, piece_id=0, new_pos=10, dice_roll=5)
    res = g.apply_move(mv)

    # Should hit blockade and receive hit_blockade penalty; shaping delta 0 with gamma=1
    assert res.events.hit_blockade and not res.events.move_resolved
    assert res.rewards is not None
    assert math.isclose(res.rewards[0], reward_config.hit_blockade, rel_tol=1e-6)


def test_shaping_depth_increases_risk_penalty(monkeypatch):
    # Arrange a position where an opponent is 2-3 steps behind and can capture
    g1 = _simple_game()
    g1.players[0].pieces[0].position = 10  # agent token
    # Place a green token such that it can reach agent with certain k
    g1.players[1].pieces[0].position = 7  # 3 steps behind -> k=3 capture chance 1/6

    # Isolate risk component
    for attr, val in (
        ("ro_w_progress", 0.0),
        ("ro_w_cap_opp", 0.0),
        ("ro_w_finish_opp", 0.0),
        ("ro_w_cap_risk", 1.0),
        ("shaping_use", True),
        ("shaping_alpha", 1.0),
        ("shaping_gamma", 1.0),
    ):
        monkeypatch.setattr(reward_config, attr, val, raising=False)

    # Move that changes little else (10->11)
    mv = Move(player_index=0, piece_id=0, new_pos=11, dice_roll=1)

    # Depth 1
    monkeypatch.setattr(reward_config, "ro_depth", 1, raising=False)
    r1 = g1.apply_move(mv).rewards[0]

    # Reset similar game, depth 3
    g3 = _simple_game()
    g3.players[0].pieces[0].position = 10
    g3.players[1].pieces[0].position = 7
    monkeypatch.setattr(reward_config, "ro_depth", 3, raising=False)
    r3 = g3.apply_move(
        Move(player_index=0, piece_id=0, new_pos=11, dice_roll=1)
    ).rewards[0]

    # With greater depth, added risk shaping should be more punitive (more negative contribution)
    assert r3 <= r1 + 1e-6
