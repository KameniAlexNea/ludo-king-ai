from __future__ import annotations

from typing import List, Tuple

from ludo_engine import PlayerColor
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sb3_contrib import MaskablePPO

from ludo_rl.ludo_env.ludo_env import LudoRLEnv
from ludo_rl.utils.move_utils import MoveUtils



def collect_imitation_samples(
    env: LudoRLEnv,
    strategies: List[str],
    steps_budget: int,
    multi_seat: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect (obs, action, mask) triples using scripted strategies.

    multi_seat: if True, rotate each color as 'agent' perspective (rebuilding obs_builder)
    NOTE: Uses internal env attributes; intended only for pretrain bootstrap.
    """
    obs_list: List[np.ndarray] = []
    act_list: List[int] = []
    mask_list: List[np.ndarray] = []
    step_counter = 0
    while step_counter < steps_budget:
        env.reset()
        seat_colors = [env.agent_color] if not multi_seat else [PlayerColor.RED.value, PlayerColor.GREEN.value, PlayerColor.YELLOW.value, PlayerColor.BLUE.value]
        for seat in seat_colors:
            env.agent_color = seat
            env.obs_builder = env.obs_builder.__class__(env.cfg, env.game, seat)
            env._ensure_agent_turn()
            env._pending_dice, env._pending_valid = env._roll_agent_dice()
            if not env._pending_valid:
                continue
            obs = env.obs_builder.build(env.turns, env._pending_dice)
            mask = MoveUtils.action_mask(env._pending_valid)
            player = env.game.get_player_from_color(seat)
            try:
                ctx = env.game.get_ai_decision_context(env._pending_dice)
                tok = player.make_strategic_decision(ctx)
            except Exception:
                tok = env._pending_valid[0].token_id
            obs_list.append(obs)
            act_list.append(tok)
            mask_list.append(mask)
            step_counter += 1
            if step_counter >= steps_budget:
                break
    return (
        np.stack(obs_list, axis=0).astype(np.float32),
        np.array(act_list, dtype=np.int64),
        np.stack(mask_list, axis=0).astype(np.float32),
    )


def imitation_train(
    model: MaskablePPO, dataset: TensorDataset, epochs: int, batch_size: int
) -> None:
    policy = model.policy
    optimizer = policy.optimizer
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    policy.train()
    for _ in range(epochs):
        for batch in loader:
            obs_t, act_t, mask_t = batch
            dist = policy.get_distribution(obs_t)
            log_probs = dist.distribution.log_prob(act_t)
            valid_for_action = mask_t[torch.arange(mask_t.size(0)), act_t]
            loss = -(log_probs * valid_for_action).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
