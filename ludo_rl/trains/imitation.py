from typing import Iterable, List, Tuple

import numpy as np
import torch
from loguru import logger
from ludo_engine import PlayerColor
from ludo_engine.core import LudoGame
from ludo_engine.models import ALL_COLORS
from ludo_engine.strategies.strategy import StrategyFactory
from sb3_contrib import MaskablePPO
from torch.utils.data import DataLoader, TensorDataset

from ludo_rl.ludo_env.ludo_env import LudoRLEnv
from ludo_rl.utils.move_utils import MoveUtils


def collect_imitation_samples(
    env: LudoRLEnv,
    strategies: List[str],
    steps_budget: int,
    multi_seat: bool = False,
    shuffle_strategies: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect (obs, action, mask) triples from scripted self-play.

    Strategy assignment:
      - Each new game, assign strategies (cycled or shuffled) to all 4 players.
      - If multi_seat=True, rotate agent seat across colors; else keep env.agent_color.

    We simulate full games turn-by-turn, recording only turns where the acting player
    matches the current agent seat. Observations are always built from that agent seat's
    perspective using env.obs_builder.
    """

    logger.info(
        f"[Imitation] Starting collection of {steps_budget} samples, multi_seat={multi_seat}"
    )

    obs_buf: List[np.ndarray] = []
    act_buf: List[int] = []
    mask_buf: List[np.ndarray] = []
    max_steps = 250

    # Prepare a cycling iterator of strategy names for assignment
    if not strategies:
        raise ValueError(
            "Must provide at least one strategy name for imitation collection"
        )

    strat_pool = list(strategies)
    strat_index = 0

    def next_four() -> List[str]:
        nonlocal strat_index, strat_pool
        if shuffle_strategies:
            env.rng.shuffle(strat_pool)
            strat_index = 0
        # Cycle or slice first 4 (repeat if fewer provided)
        chosen: List[str] = []
        while len(chosen) < 4:
            chosen.append(strat_pool[strat_index % len(strat_pool)])
            strat_index += 1
        return chosen[:4]

    collected = 0
    agent_color_cycle: Iterable[PlayerColor]
    if multi_seat:
        agent_color_cycle = ALL_COLORS
    else:
        agent_color_cycle = [env.agent_color]

    # We'll cycle through agent colors outer loop if multi_seat
    while collected < steps_budget:
        for agent_col in agent_color_cycle:
            if collected >= steps_budget:
                break
            # New game instance & re-bind into env
            env.agent_color = agent_col
            env.game = LudoGame(ALL_COLORS)
            env.obs_builder = env.obs_builder.__class__(
                env.cfg, env.game, env.agent_color
            )
            # Assign strategies to all players
            assigned = next_four()
            players = env.game.players  # order matches colors above
            for p_obj, strat_name in zip(players, assigned):
                p_obj.set_strategy(StrategyFactory.create_strategy(strat_name))
            # Simulate until enough samples or game over
            turn_index = 0
            while (
                not env.game.game_over
                and collected < steps_budget
                and turn_index < max_steps
            ):
                current_player = env.game.get_current_player()
                dice = env.game.roll_dice()
                valid = env.game.get_valid_moves(current_player, dice)
                if valid:
                    try:
                        ctx = env.game.get_ai_decision_context(dice)
                        token_id = current_player.make_strategic_decision(ctx)
                    except Exception:
                        token_id = valid[0].token_id
                    res = env.game.execute_move(current_player, token_id, dice)
                    # Record only if this was the agent turn
                    if current_player.color == env.agent_color:
                        obs = env.obs_builder.build(turn_index, dice)
                        mask = MoveUtils.action_mask(valid)
                        obs_buf.append(obs)
                        act_buf.append(token_id)
                        mask_buf.append(mask)
                        collected += 1
                        if collected % 10000 == 0:
                            logger.info(
                                f"[Imitation] Collected {collected}/{steps_budget} samples"
                            )
                        if collected >= steps_budget:
                            break
                    if not res.extra_turn:
                        env.game.next_turn()
                else:
                    env.game.next_turn()
                turn_index += 1
            # End game loop
    logger.info(f"[Imitation] Collection completed with {len(obs_buf)} samples")
    # Convert buffers to arrays - handle Dict observations
    if not obs_buf:
        raise RuntimeError(
            "No imitation samples collected; check strategy availability."
        )
    
    # For Dict observations, return as lists (don't stack)
    if isinstance(obs_buf[0], dict):
        return obs_buf, np.array(act_buf, dtype=np.int64), np.stack(mask_buf, axis=0).astype(np.float32)
    else:
        # Original behavior for flat arrays
        return (
            np.stack(obs_buf, axis=0).astype(np.float32),
            np.array(act_buf, dtype=np.int64),
            np.stack(mask_buf, axis=0).astype(np.float32),
        )


def imitation_train(
    model: MaskablePPO, dataset, epochs: int, batch_size: int
) -> None:
    curr_device = model.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[Imitation] Training on device: {device}")
    logger.info(
        f"[Imitation] Starting training for {epochs} epochs with batch_size={batch_size}"
    )
    policy = model.policy
    policy.to(device)
    optimizer = policy.optimizer
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    policy.train()
    loss = None
    for _ in range(epochs):
        logger.info(
            f"[Imitation] Epoch {_ + 1}/{epochs} Loss: {loss.item() if loss is not None else 'N/A'}"
        )
        for batch in loader:
            if isinstance(batch, dict):
                # Handle Dict dataset
                obs_list = batch['obs']
                act_t = batch['act']
                mask_t = batch['mask']
                # Manually collate the obs dict
                obs_t = {}
                for key in obs_list[0].keys():
                    tensors = [torch.from_numpy(obs[key]) for obs in obs_list]
                    obs_t[key] = torch.stack(tensors, dim=0)
            else:
                # Handle TensorDataset
                obs_t, act_t, mask_t = batch
            
            if isinstance(obs_t, dict):
                obs_t = {k: v.to(device) for k, v in obs_t.items()}
            else:
                obs_t = obs_t.to(device) if hasattr(obs_t, 'to') else obs_t
            act_t = act_t.to(device)
            mask_t = mask_t.to(device)
            dist = policy.get_distribution(obs_t)
            log_probs: torch.Tensor = dist.distribution.log_prob(act_t)
            valid_for_action: torch.Tensor = mask_t[
                torch.arange(mask_t.size(0), device=device), act_t
            ]
            loss: torch.Tensor = -(log_probs * valid_for_action).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    logger.info("[Imitation] Training completed")
    model.policy.to(curr_device)
