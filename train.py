from __future__ import annotations

import argparse
import copy
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from ludo_rl.callbacks.curriculum import ProgressCallback
from ludo_rl.callbacks.eval_baselines import SimpleBaselineEvalCallback
from ludo_rl.callbacks.annealing import AnnealingCallback
from ludo_rl.config import EnvConfig, TrainConfig
from ludo_rl.ludo_env.ludo_env import LudoRLEnv
from ludo_rl.ludo_env.ludo_env_selfplay import LudoRLEnvSelfPlay
from ludo_rl.utils.move_utils import MoveUtils


def make_env(rank: int, seed: int, base_cfg: EnvConfig, env_type: str = "classic"):
    def _init():
        cfg = copy.deepcopy(base_cfg)
        cfg.seed = seed + rank
        if env_type == "selfplay":
            env = LudoRLEnvSelfPlay(cfg)
        else:
            env = LudoRLEnv(cfg)
        return ActionMasker(env, MoveUtils.get_action_mask_for_env)

    return _init


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=TrainConfig.total_steps)
    p.add_argument("--n-envs", type=int, default=TrainConfig.n_envs)
    p.add_argument("--logdir", type=str, default=TrainConfig.logdir)
    p.add_argument("--model-dir", type=str, default=TrainConfig.model_dir)
    p.add_argument("--eval-freq", type=int, default=TrainConfig.eval_freq)
    p.add_argument("--eval-games", type=int, default=TrainConfig.eval_games)
    p.add_argument(
        "--checkpoint-freq",
        type=int,
        default=100_000,
        help="Checkpoint every N steps; 0 disables",
    )
    p.add_argument(
        "--checkpoint-prefix",
        type=str,
        default="ppo_ludo",
        help="Checkpoint file prefix",
    )
    p.add_argument(
        "--eval-baselines",
        type=str,
        default=TrainConfig.eval_baselines,
        help="Comma-separated list of opponent strategy names",
    )
    p.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    p.add_argument("--n-steps", type=int, default=TrainConfig.n_steps)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--ent-coef", type=float, default=TrainConfig.ent_coef)
    p.add_argument("--max-turns", type=int, default=TrainConfig.max_turns)
    # Imitation / kickstart
    p.add_argument("--imitation-enabled", action="store_true", default=False)
    p.add_argument(
        "--imitation-strategies",
        type=str,
        default=TrainConfig.imitation_strategies,
        help="Comma-separated scripted strategies to imitate",
    )
    p.add_argument("--imitation-steps", type=int, default=TrainConfig.imitation_steps)
    p.add_argument(
        "--imitation-batch-size", type=int, default=TrainConfig.imitation_batch_size
    )
    p.add_argument("--imitation-epochs", type=int, default=TrainConfig.imitation_epochs)
    p.add_argument(
        "--imitation-entropy-boost",
        type=float,
        default=TrainConfig.imitation_entropy_boost,
        help="Temporary entropy bonus added to ent_coef during imitation phase",
    )
    # Annealing overrides
    p.add_argument(
        "--entropy-coef-initial", type=float, default=TrainConfig.entropy_coef_initial
    )
    p.add_argument(
        "--entropy-coef-final", type=float, default=TrainConfig.entropy_coef_final
    )
    p.add_argument(
        "--entropy-anneal-steps", type=int, default=TrainConfig.entropy_anneal_steps
    )
    p.add_argument(
        "--capture-scale-initial",
        type=float,
        default=TrainConfig.capture_scale_initial,
    )
    p.add_argument(
        "--capture-scale-final", type=float, default=TrainConfig.capture_scale_final
    )
    p.add_argument(
        "--capture-scale-anneal-steps",
        type=int,
        default=TrainConfig.capture_scale_anneal_steps,
    )
    # Learning rate annealing
    p.add_argument(
        "--lr-final",
        type=float,
        default=TrainConfig.learning_rate * 0.25,
        help="Final learning rate for linear anneal (initial is --learning-rate)",
    )
    p.add_argument(
        "--env-type",
        type=str,
        default="classic",
        choices=["classic", "selfplay"],
        help="Environment type",
    )
    return p.parse_args()


def _linear_interp(start: float, end: float, frac: float) -> float:
    return start + (end - start) * min(1.0, max(0.0, frac))


def _collect_imitation_samples(
    env: LudoRLEnv,
    strategies: List[str],
    steps_budget: int,
    multi_seat: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect (obs, action, mask) triples using scripted strategies.

    multi_seat: if True, rotate each color as 'agent' perspective (rebuilding obs_builder)
    """
    obs_list: List[np.ndarray] = []
    act_list: List[int] = []
    mask_list: List[np.ndarray] = []
    step_counter = 0
    strat_cycle = 0
    # Simple round-robin over provided strategies for attaching to opponents
    while step_counter < steps_budget:
        # Reset environment with selected opponent quadruple
        opps = [s.strip() for s in strategies]
        env.reset()
        # If multi-seat imitation, iterate each of 4 colors once per env reset
        seat_colors = [env.agent_color] if not multi_seat else list(env.game.colors)
        for seat in seat_colors:
            # Force agent perspective color
            env.agent_color = seat
            # Rebuild observation builder for seat
            env.obs_builder = env.obs_builder.__class__(env.cfg, env.game, seat)
            # Roll dice and get valid moves for that seat turn
            env._ensure_agent_turn()
            env._pending_dice, env._pending_valid = env._roll_agent_dice()
            if not env._pending_valid:
                continue
            obs = env.obs_builder.build(env.turns, env._pending_dice)
            mask = MoveUtils.action_mask(env._pending_valid)
            # Make strategy decision using that player's current attached strategy
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
        strat_cycle += 1
    return (
        np.stack(obs_list, axis=0).astype(np.float32),
        np.array(act_list, dtype=np.int64),
        np.stack(mask_list, axis=0).astype(np.float32),
    )


def _imitation_train(model: MaskablePPO, dataset: TensorDataset, epochs: int, batch_size: int):
    policy = model.policy
    optimizer = policy.optimizer  # reuse underlying optimizer
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    policy.train()
    for _ in range(epochs):
        for batch in loader:
            obs_t, act_t, mask_t = batch
            # Forward through policy (SB3 expects obs dict maybe; here plain tensor)
            dist = policy.get_distribution(obs_t)
            log_probs = dist.distribution.log_prob(act_t)
            # Mask invalid actions: set loss high for actions chosen outside mask (if mask=0)
            # We multiply log_probs by a validity indicator to only reinforce valid actions.
            # valid_mask for taken actions: gather mask entries
            valid_for_action = mask_t[torch.arange(mask_t.size(0)), act_t]
            # Negative log likelihood only for valid actions; invalid ones get zero weight
            loss = -(log_probs * valid_for_action).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    env_cfg = EnvConfig(max_turns=args.max_turns)

    if args.n_envs == 1:
        venv = DummyVecEnv([make_env(0, 42, env_cfg, args.env_type)])
    else:
        venv = SubprocVecEnv(
            [
                make_env(i, 42 + i * 100, env_cfg, args.env_type)
                for i in range(args.n_envs)
            ]
        )

    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_reward=False)

    # Separate eval env with same wrappers (always classic for evaluation vs baselines)
    eval_env = DummyVecEnv([make_env(999, 1337, env_cfg, "classic")])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)
    try:
        eval_env.obs_rms = venv.obs_rms
    except Exception:
        pass

    model = MaskablePPO(
        "MlpPolicy",
        venv,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        tensorboard_log=args.logdir,
        verbose=1,
        device="cpu",
    )

    # When using selfplay, inject the live model into envs so they can snapshot policy at reset
    if args.env_type == "selfplay":
        try:
            venv.env_method("set_model", model)
        except Exception:
            # Some VecEnv types may require accessing the underlying attribute
            pass

    progress_cb = ProgressCallback(total_timesteps=args.total_steps, update_freq=10_000)
    eval_cb = SimpleBaselineEvalCallback(
        baselines=[s.strip() for s in args.eval_baselines.split(",") if s.strip()],
        n_games=args.eval_games,
        eval_freq=args.eval_freq,
        env_cfg=env_cfg,
        verbose=1,
    )
    callbacks = [progress_cb, eval_cb]

    # Annealing callback (entropy + capture scale + learning rate). We'll wrap learning rate logic here by updating optimizer lr.
    train_cfg = TrainConfig(
        total_steps=args.total_steps,
        n_envs=args.n_envs,
        eval_freq=args.eval_freq,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        entropy_coef_initial=args.entropy_coef_initial,
        entropy_coef_final=args.entropy_coef_final,
        entropy_anneal_steps=args.entropy_anneal_steps,
        capture_scale_initial=args.capture_scale_initial,
        capture_scale_final=args.capture_scale_final,
        capture_scale_anneal_steps=args.capture_scale_anneal_steps,
    )
    anneal_cb = AnnealingCallback(train_cfg)
    callbacks.append(anneal_cb)

    # Optional imitation kickstart
    if args.imitation_enabled:
        print("[Imitation] Collecting scripted policy samples...")
        strat_list = [s.strip() for s in args.imitation_strategies.split(",") if s.strip()]
        # Single-seat samples
        base_env_for_imitation = LudoRLEnv(env_cfg)
        obs_s, act_s, mask_s = _collect_imitation_samples(
            base_env_for_imitation, strat_list, steps_budget=args.imitation_steps, multi_seat=False
        )
        # Multi-seat samples
        obs_m, act_m, mask_m = _collect_imitation_samples(
            base_env_for_imitation, strat_list, steps_budget=args.imitation_steps, multi_seat=True
        )
        obs_all = np.concatenate([obs_s, obs_m], axis=0)
        act_all = np.concatenate([act_s, act_m], axis=0)
        mask_all = np.concatenate([mask_s, mask_m], axis=0)
        dataset = TensorDataset(
            torch.from_numpy(obs_all), torch.from_numpy(act_all), torch.from_numpy(mask_all)
        )
        # Temporary entropy boost
        original_ent = model.ent_coef if isinstance(model.ent_coef, float) else args.ent_coef
        boosted_ent = original_ent + args.imitation_entropy_boost
        if isinstance(model.ent_coef, float):
            model.ent_coef = boosted_ent
        print(
            f"[Imitation] Training on {len(dataset)} samples (single+multi-seat) for {args.imitation_epochs} epochs"
        )
        _imitation_train(
            model,
            dataset,
            epochs=args.imitation_epochs,
            batch_size=args.imitation_batch_size,
        )
        # Restore entropy coef (annealing callback will handle future schedule)
        if isinstance(model.ent_coef, float):
            model.ent_coef = original_ent
        print("[Imitation] Completed pretraining phase.")

    # Add checkpointing if enabled
    if args.checkpoint_freq and args.checkpoint_freq > 0:
        ckpt_cb = CheckpointCallback(
            save_freq=args.checkpoint_freq // args.n_envs,
            save_path=args.model_dir,
            name_prefix=args.checkpoint_prefix,
            save_replay_buffer=True,
            save_vecnormalize=True,
            verbose=1,
        )
        callbacks.append(ckpt_cb)

    # Wrap learn with manual LR annealing by hooking into progress_cb logic via custom loop if needed.
    # Simpler: monkey patch progress callback to also adjust LR based on num_timesteps.
    initial_lr = args.learning_rate
    final_lr = args.lr_final

    original_on_step = progress_cb._on_step

    def lr_wrapper():  # closes over model
        frac = model.num_timesteps / float(args.total_steps) if args.total_steps > 0 else 1.0
        new_lr = _linear_interp(initial_lr, final_lr, frac)
        try:
            for g in model.policy.optimizer.param_groups:
                g["lr"] = new_lr
        except Exception:
            pass

    def patched_on_step():
        lr_wrapper()
        return original_on_step()

    progress_cb._on_step = patched_on_step  # type: ignore

    model.learn(total_timesteps=args.total_steps, callback=callbacks)
    model.save(os.path.join(args.model_dir, "maskable_ppo_ludo_rl_final"))


if __name__ == "__main__":
    main()
