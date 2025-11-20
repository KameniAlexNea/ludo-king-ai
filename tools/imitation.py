from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from dotenv import load_dotenv
from loguru import logger
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from ludo_rl.extractor import LudoCnnExtractor, LudoTransformerExtractor
from ludo_rl.ludo_env import LudoEnv
from ludo_rl.ludo_king import config as king_config
from ludo_rl.ludo_king.config import net_config
from ludo_rl.ludo_king.player import Player
from ludo_rl.strategy.registry import STRATEGY_REGISTRY
from ludo_rl.strategy.registry import available as available_strategies

load_dotenv()


@dataclass(slots=True)
class SampleBatch:
    positions: np.ndarray
    dice_history: np.ndarray
    token_mask: np.ndarray
    player_history: np.ndarray
    token_colors: np.ndarray
    current_dice: np.ndarray
    action_mask: np.ndarray
    action: np.ndarray
    episode_id: np.ndarray
    step_in_episode: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Iterative imitation learning with LudoEnv: collect N steps from a teacher, "
            "then supervise-train the PPO policy; repeat for K iterations."
        )
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Path to the PPO checkpoint to fine-tune. If empty, initialise a fresh model like train.py.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Directory to store imitation checkpoints (iterative and final).",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default=os.getenv("TEACHER", "homebody"),
        help=(
            "Strategy name to imitate (fixed teacher that controls agent seat during collection)."
        ),
    )
    parser.add_argument(
        "--opponents",
        type=str,
        default=os.getenv("OPPONENTS", ",".join(available_strategies())),
        help="Comma-separated opponent strategy pool (env will select per-episode).",
    )
    parser.add_argument(
        "--collect-steps",
        type=int,
        default=int(os.getenv("COLLECT_STEPS", 65536)),
        help="Number of agent steps to collect per iteration.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=int(os.getenv("IMIT_ITERS", 20)),
        help="Number of collect->train iterations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=int(os.getenv("BC_EPOCHS", 1)),
        help="Behaviour cloning epochs per iteration.",
    )
    parser.add_argument(
        "--bc-batch-size",
        type=int,
        default=int(os.getenv("BC_BATCH_SIZE", 1024)),
        help="Batch size for behaviour cloning updates.",
    )
    parser.add_argument(
        "--bc-lr",
        type=float,
        default=float(os.getenv("BC_LR", 1e-4)),
        help="Learning rate for the behaviour cloning optimiser.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use for imitation updates (cpu / cuda / mps).",
    )
    parser.add_argument(
        "--use-transformer",
        action="store_true",
        help="When starting fresh (no --model-path), initialise a Transformer-based extractor like train.py.",
    )
    parser.add_argument(
        "--dagger-frac-max",
        type=float,
        default=float(os.getenv("DAGGER_FRAC_MAX", 0.2)),
        help="Final DAgger agent-execution fraction (anneals 0 -> dagger-frac-max across iterations).",
    )
    parser.add_argument(
        "--dagger-warmup-iters",
        type=int,
        default=int(os.getenv("DAGGER_WARMUP_ITERS", 0)),
        help="Number of initial iterations to keep DAgger disabled (fraction=0) before annealing.",
    )
    parser.add_argument(
        "--ppo-finetune-last",
        type=int,
        default=int(os.getenv("PPO_FINETUNE_LAST", 5)),
        help="Number of last iterations to switch from BC to PPO fine-tuning.",
    )
    parser.add_argument(
        "--ppo-steps-per-iter",
        type=int,
        default=int(os.getenv("PPO_STEPS_PER_ITER", 65536)),
        help="Total timesteps per PPO fine-tune iteration.",
    )
    return parser.parse_args()


def validate_strategies(opponents: Sequence[str]) -> None:
    unknown = [name for name in opponents if name not in STRATEGY_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown strategy names: {', '.join(unknown)}")


def build_board_stack(env: LudoEnv, player_index: int) -> np.ndarray:
    return env.game.board.build_tensor(player_index)  # type: ignore[attr-defined]


def attach_strategy(player: Player, strategy_name: str, rng: random.Random) -> None:
    cls = STRATEGY_REGISTRY[strategy_name]
    try:
        player.strategy = cls.create_instance(rng)
    except NotImplementedError:
        player.strategy = cls()
    player.strategy_name = strategy_name  # type: ignore[attr-defined]


def decide_with_teacher(env: LudoEnv, teacher: str, rng: random.Random) -> int:
    """Select agent action (piece id 0..3) using the teacher strategy on current env state."""
    dice = int(env.current_dice_roll)
    legal = env.game.legal_moves(0, dice)
    if not legal:
        return 0
    board_stack = build_board_stack(env, player_index=0)
    decision = env.game.players[0].choose(board_stack, dice, legal)
    mv = decision if decision is not None else rng.choice(legal)
    return int(getattr(mv, "piece_id", 0))


def decide_with_agent(
    model: MaskablePPO, obs: dict, mask: np.ndarray, deterministic: bool
) -> int:
    action, _ = model.predict(
        obs, action_masks=mask[None, ...], deterministic=deterministic
    )
    return int(np.asarray(action).item())


def collect_fixed_teacher_steps(
    env: LudoEnv,
    teacher: str,
    steps: int,
    rng: random.Random,
    model: MaskablePPO | None = None,
    dagger_frac: float = 0.0,
    deterministic_agent: bool = True,
    opponents_pool: Sequence[str] | None = None,
) -> SampleBatch:
    pos_list: List[np.ndarray] = []
    dice_hist_list: List[np.ndarray] = []
    token_mask_list: List[np.ndarray] = []
    player_hist_list: List[np.ndarray] = []
    token_colors_list: List[np.ndarray] = []
    curr_dice_list: List[np.ndarray] = []
    action_mask_list: List[np.ndarray] = []
    actions: List[int] = []
    epi_ids: List[int] = []
    step_ids: List[int] = []

    # Sample 3 opponents (or NUM_PLAYERS-1) from pool and fix lineup for this episode
    def _choose_lineup() -> List[str]:
        pool = list(opponents_pool or [])
        pool = [s for s in pool if s != teacher]
        k = max(1, king_config.NUM_PLAYERS - 1)
        if not pool:
            return ["random"] * k
        # Use seeded RNG for determinism when --seed is provided
        return rng.choices(pool, k=k)

    env._fixed_opponents_strategies = _choose_lineup()
    env._reset_count = 1  # force using fixed lineup immediately
    obs, info = env.reset()
    # Attach teacher once per episode
    attach_strategy(env.game.players[0], teacher, rng)
    collected = 0
    episode_counter = 0
    step_in_epi = 0
    while collected < steps:
        mask = info.get("action_mask") if isinstance(info, dict) else None
        if mask is None or not np.any(mask):
            # still store skipped observation? skip and let opponents play
            action = 0
        else:
            # DAgger: choose executor; always label with teacher action
            teacher_action = decide_with_teacher(env, teacher, rng)
            use_agent = model is not None and rng.random() < max(
                0.0, min(1.0, dagger_frac)
            )
            if use_agent:
                action = decide_with_agent(model, obs, mask, deterministic_agent)
            else:
                action = teacher_action

        # Record only when there is a valid mask (agent turn)
        if mask is not None and np.any(mask):
            pos_list.append(np.array(obs["positions"], copy=True))
            dice_hist_list.append(np.array(obs["dice_history"], copy=True))
            token_mask_list.append(np.array(obs["token_mask"], copy=True))
            player_hist_list.append(np.array(obs["player_history"], copy=True))
            token_colors_list.append(np.array(obs["token_colors"], copy=True))
            curr_dice_list.append(np.array(obs["current_dice"], copy=True))
            action_mask_list.append(np.array(mask, copy=True))
            actions.append(int(teacher_action))
            epi_ids.append(episode_counter)
            step_ids.append(step_in_epi)
            collected += 1
            step_in_epi += 1

        obs, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            episode_counter += 1
            step_in_epi = 0
            # New episode -> new opponent lineup
            env._fixed_opponents_strategies = _choose_lineup()
            env._reset_count = 1
            obs, info = env.reset()
            attach_strategy(env.game.players[0], teacher, rng)

    return SampleBatch(
        positions=np.stack(pos_list, axis=0),
        dice_history=np.stack(dice_hist_list, axis=0),
        token_mask=np.stack(token_mask_list, axis=0),
        player_history=np.stack(player_hist_list, axis=0),
        token_colors=np.stack(token_colors_list, axis=0),
        current_dice=np.stack(curr_dice_list, axis=0),
        action_mask=np.stack(action_mask_list, axis=0),
        action=np.asarray(actions, dtype=np.int64),
        episode_id=np.asarray(epi_ids, dtype=np.int64),
        step_in_episode=np.asarray(step_ids, dtype=np.int64),
    )


def save_dataset(dataset: SampleBatch, path: str | None) -> None:
    if path is None:
        return
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        positions=dataset.positions,
        dice_history=dataset.dice_history,
        token_mask=dataset.token_mask,
        player_history=dataset.player_history,
        token_colors=dataset.token_colors,
        current_dice=dataset.current_dice,
        action_mask=dataset.action_mask,
        action=dataset.action,
        episode_id=dataset.episode_id,
        step_in_episode=dataset.step_in_episode,
    )
    logger.info(f"Saved dataset with {dataset.action.shape[0]} samples to {dest}")


def behaviour_clone_step(
    policy: MaskableActorCriticPolicy,
    dataset: SampleBatch,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
) -> float:
    policy.train()
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)
    num_samples = dataset.action.shape[0]

    # Tensors for token-sequence observation
    tens = {
        "positions": torch.tensor(dataset.positions, dtype=torch.long, device=device),
        "dice_history": torch.tensor(
            dataset.dice_history, dtype=torch.long, device=device
        ),
        "token_mask": torch.tensor(dataset.token_mask, dtype=torch.bool, device=device),
        "player_history": torch.tensor(
            dataset.player_history, dtype=torch.long, device=device
        ),
        "token_colors": torch.tensor(
            dataset.token_colors, dtype=torch.long, device=device
        ),
        "current_dice": torch.tensor(
            dataset.current_dice, dtype=torch.long, device=device
        ),
    }
    mask_tensor = torch.tensor(dataset.action_mask, dtype=torch.bool, device=device)
    action_tensor = torch.tensor(dataset.action, dtype=torch.long, device=device)

    indices = np.arange(num_samples)
    last_loss = 0.0
    for epoch in range(max(1, epochs)):
        rng_idx = np.random.permutation(indices)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_idx = rng_idx[start:end]
            obs = {k: v[batch_idx] for k, v in tens.items()}
            mask_batch = mask_tensor[batch_idx]
            action_batch = action_tensor[batch_idx]

            dist = policy.get_distribution(obs, action_masks=mask_batch)
            log_prob = dist.log_prob(action_batch)
            loss = -log_prob.mean()

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimiser.step()

            epoch_loss += float(loss.item())
            batches += 1
        last_loss = epoch_loss / max(1, batches)
        logger.info(f"BC epoch {epoch + 1}/{epochs} - loss {last_loss:.4f}")
    return last_loss


def main() -> None:
    args = parse_args()
    rng = random.Random()

    teacher = args.teacher.strip().lower()
    opponents = [s.strip() for s in args.opponents.split(",") if s.strip()]
    validate_strategies([teacher])
    validate_strategies(opponents)

    # Load or initialise model
    if args.model_path and os.path.exists(args.model_path):
        model = MaskablePPO.load(args.model_path, device=args.device)
        policy = model.policy
        logger.info(f"Loaded base model from {args.model_path}")
    else:
        # Build a minimal VecEnv for policy initialisation (mirrors train.py shapes)
        def _make_env():
            return LudoEnv()

        vec = DummyVecEnv([_make_env])
        vec = VecMonitor(vec)
        vec = VecNormalize(vec, norm_obs=False, norm_reward=True)

        policy_kwargs = dict(
            activation_fn=torch.nn.GELU,
            features_extractor_class=(
                LudoTransformerExtractor if args.use_transformer else LudoCnnExtractor
            ),
            features_extractor_kwargs=dict(features_dim=net_config.embed_dim),
            net_arch=dict(pi=net_config.pi, vf=net_config.vf),
            share_features_extractor=True,
        )
        model = MaskablePPO(
            "MultiInputPolicy",
            vec,
            policy_kwargs=policy_kwargs,
            n_steps=1024,
            batch_size=1024,
            n_epochs=1,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            learning_rate=3e-4,
            device=args.device,
            verbose=0,
        )
        policy = model.policy
        extractor_name = "Transformer" if args.use_transformer else "CNN"
        logger.info(
            f"Initialised a fresh PPO policy ({extractor_name} extractor) for imitation."
        )

    # Build env and set opponents
    env = LudoEnv(use_fixed_opponents=True)
    env.opponents = opponents
    env.strategy_selection = 0  # random pick per seat from pool
    env._reset_count = 0

    logger.info(
        f"Imitation: teacher={teacher}, iters={args.iters}, collect_steps={args.collect_steps}, opponents={opponents}"
    )

    for it in range(1, args.iters + 1):
        remaining_for_bc = max(0, args.iters - args.ppo_finetune_last)
        # Warmup + linear anneal during BC phase only
        warmup = max(0, int(args.dagger_warmup_iters))
        if it <= warmup or remaining_for_bc <= warmup:
            dagger_frac = 0.0
        else:
            # map it in [warmup+1 .. remaining_for_bc] -> [0 .. 1]
            span = max(1, remaining_for_bc - warmup)
            pos = min(it, remaining_for_bc) - warmup
            dagger_frac = args.dagger_frac_max * float(pos) / float(span)
        if it <= remaining_for_bc:
            # Behaviour Cloning phase with DAgger mix
            dataset = collect_fixed_teacher_steps(
                env=env,
                teacher=teacher,
                steps=args.collect_steps,
                rng=rng,
                model=model,
                dagger_frac=dagger_frac,
                deterministic_agent=True,
                opponents_pool=opponents,
            )
            logger.info(
                f"Iter {it}/{args.iters} (BC): dagger={dagger_frac:.3f}, collected {dataset.action.shape[0]} steps."
            )
            loss = behaviour_clone_step(
                policy=policy,
                dataset=dataset,
                device=args.device,
                epochs=args.bc_epochs,
                batch_size=args.bc_batch_size,
                lr=args.bc_lr,
            )
            logger.info(f"Iter {it}: BC loss {loss:.4f}")
        else:
            # PPO fine-tuning phase (optimize full trajectories)
            logger.info(
                f"Iter {it}/{args.iters} (PPO fine-tune): running {args.ppo_steps_per_iter} timesteps."
            )

            # Build a PPO env with same opponents
            def _make_env_ppo():
                e = LudoEnv(use_fixed_opponents=True)
                e.opponents = opponents
                e.strategy_selection = 0
                e._reset_count = 0
                return e

            ppo_env = DummyVecEnv([_make_env_ppo])
            ppo_env = VecMonitor(ppo_env)
            ppo_env = VecNormalize(ppo_env, norm_obs=False, norm_reward=True)
            model.set_env(ppo_env)
            model.learn(total_timesteps=int(args.ppo_steps_per_iter))

        # Save checkpoint each iteration under the provided folder
        save_dir = Path(args.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        iter_path = save_dir / f"imitation_iter{it}.zip"
        model.save(str(iter_path))
        logger.info(f"Saved checkpoint to {iter_path}")

    # Final save in the same folder
    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    final_path = save_dir / "imitation_final.zip"
    model.save(str(final_path))
    logger.info(f"Saved final behaviour-cloned checkpoint to {final_path}")


if __name__ == "__main__":
    main()
