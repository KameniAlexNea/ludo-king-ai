from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer
from stable_baselines3.common.utils import explained_variance
from torch.utils.tensorboard import SummaryWriter

from ludo_rl.eval import champion_vs_fixed, tournament_round_robin
from models.configs.config import EnvConfig
from models.envs.ludo_env_aec.raw_env import raw_env as AECEnv

AGENTS = ("player_0", "player_1", "player_2", "player_3")


@dataclass
class PPOHyper:
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


def build_spaces(env: AECEnv) -> Tuple[spaces.Dict, spaces.Discrete]:
    sample_agent = AGENTS[0]
    obs_space = spaces.Dict(
        {
            "observation": env.observation_space(sample_agent),
            "action_mask": spaces.Box(
                low=0,
                high=1,
                shape=(env.action_space(sample_agent).n,),
                dtype=np.int8,
            ),
            "agent_index": spaces.Discrete(len(AGENTS)),
        }
    )
    act_space = env.action_space(sample_agent)
    return obs_space, act_space


def build_obs(env: AECEnv, agent: str) -> dict:
    raw = env.observe(agent)
    return {
        "observation": raw["observation"].astype(np.float32, copy=False),
        "action_mask": raw["action_mask"].astype(np.int8, copy=False),
        "agent_index": np.array(AGENTS.index(agent), dtype=np.int64),
    }


def train_arena_ippo(
    env_cfg: EnvConfig,
    lr: float = 3e-4,
    n_steps: int = 2048,
    total_updates: int = 1000,
    device: str = "cpu",
):
    cfg = copy.deepcopy(env_cfg)
    cfg.multi_agent = True  # flattened obs
    env = AECEnv(cfg)
    env.reset()

    obs_space, act_space = build_spaces(env)

    # Four independent policies
    models: Dict[str, MaskablePPO] = {}
    for ag in AGENTS:
        models[ag] = MaskablePPO(
            "MultiInputPolicy",
            env=None,  # manual collection; init after setting spaces
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=256,
            ent_coef=0.01,
            vf_coef=0.5,
            gamma=0.99,
            gae_lambda=0.95,
            device=device,
            verbose=0,
            tensorboard_log=os.path.join("training", "logs_arena"),
            policy_kwargs={
                "activation_fn": th.nn.Tanh,
                "net_arch": {"pi": [128, 128], "vf": [256, 256]},
            },
            _init_setup_model=False,
        )
        # Set spaces and minimal env metadata then finish model setup
        models[ag].observation_space = obs_space
        models[ag].action_space = act_space
        # Minimal attributes expected by SB3 internals
        models[ag].n_envs = 1  # type: ignore[attr-defined]
        if getattr(models[ag], "rollout_buffer_kwargs", None) is None:
            models[ag].rollout_buffer_kwargs = {}  # type: ignore[attr-defined]
        models[ag]._setup_model()

    buffers: Dict[str, MaskableDictRolloutBuffer] = {}
    for ag in AGENTS:
        buffers[ag] = MaskableDictRolloutBuffer(
            n_steps,
            obs_space,
            act_space,
            device=th.device(device),
            gamma=0.99,
            gae_lambda=0.95,
            n_envs=1,
        )

    # Per-agent last obs and episode starts
    last_obs: Dict[str, dict] = {}
    episode_starts: Dict[str, np.ndarray] = {
        ag: np.ones((1,), dtype=bool) for ag in AGENTS
    }
    dones_flag: bool = False

    def policy_step(agent: str, obs: dict):
        mdl = models[agent]
        with th.no_grad():
            obs_tensor = mdl.policy.obs_to_tensor(obs)[0]
            # action mask for policy forward (batch dimension 1)
            action_mask = th.as_tensor(obs["action_mask"], device=mdl.device).unsqueeze(
                0
            )
            actions, values, log_probs = mdl.policy(
                obs_tensor, action_masks=action_mask
            )
        # actions is shape (1,); extract scalar safely
        return (
            int(actions.cpu().numpy().item()),
            values.squeeze(),
            log_probs.squeeze(),
            action_mask,
        )

    def to_batch(x):
        return np.array([x])

    writer = SummaryWriter(log_dir=os.path.join("training", "logs_arena_tb"))
    for update in range(total_updates):
        # Reset buffers
        for ag in AGENTS:
            buffers[ag].reset()

        # Collect n_steps per agent
        steps_collected = {ag: 0 for ag in AGENTS}
        env.reset()
        dones_flag = False

        while True:
            agent = env.agent_selection
            obs = build_obs(env, agent)
            last_obs[agent] = obs
            action, value, log_prob, action_mask = policy_step(agent, obs)
            env.step(action)

            reward = float(env.rewards.get(agent, 0.0))
            term = any(env.terminations.values())
            trunc = any(env.truncations.values())
            done = term or trunc

            # Only record if this agent still needs steps
            if steps_collected[agent] < n_steps:
                buffers[agent].add(
                    last_obs[agent],
                    np.array([[action]], dtype=np.int64),
                    to_batch(reward),
                    episode_starts[agent],
                    value.unsqueeze(0),
                    log_prob.unsqueeze(0),
                    action_masks=action_mask,
                )
                steps_collected[agent] += 1
                episode_starts[agent] = np.array([done], dtype=bool)

            if done:
                dones_flag = True
                # Reset episode_starts for all agents next episode
                for ag in AGENTS:
                    episode_starts[ag] = np.ones((1,), dtype=bool)

                env.reset()

            # Stop when all agents have n_steps
            if all(steps_collected[ag] >= n_steps for ag in AGENTS):
                break

        # Compute returns & advantages per agent
        for ag in AGENTS:
            mdl = models[ag]
            if last_obs.get(ag) is not None:
                with th.no_grad():
                    last_val = mdl.policy.predict_values(
                        mdl.policy.obs_to_tensor(last_obs[ag])[0]
                    )
            else:
                last_val = th.zeros(1, device=mdl.device)
            buffers[ag].compute_returns_and_advantage(
                last_values=last_val, dones=np.array([dones_flag], dtype=bool)
            )

        # Train per agent
        for ag in AGENTS:
            mdl = models[ag]
            # Copy of sb3 ppo_mask.train but scoped to our buffer
            entropy_losses = []
            pg_losses, value_losses, clip_fractions = [], [], []
            approx_kl_divs = []

            clip_range = mdl.clip_range(mdl._current_progress_remaining)  # type: ignore[arg-type]
            if mdl.clip_range_vf is not None:
                clip_range_vf = mdl.clip_range_vf(mdl._current_progress_remaining)  # type: ignore[arg-type]

            for epoch in range(mdl.n_epochs):
                for rollout_data in buffers[ag].get(mdl.batch_size):
                    actions = rollout_data.actions.long().flatten()
                    values, log_prob, entropy = mdl.policy.evaluate_actions(
                        rollout_data.observations,
                        actions,
                        action_masks=rollout_data.action_masks,
                    )
                    values = values.flatten()
                    advantages = rollout_data.advantages
                    if mdl.normalize_advantage:
                        advantages = (advantages - advantages.mean()) / (
                            advantages.std() + 1e-8
                        )

                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(
                        ratio, 1 - clip_range, 1 + clip_range
                    )
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                    pg_losses.append(policy_loss.item())
                    clip_fraction = th.mean(
                        (th.abs(ratio - 1) > clip_range).float()
                    ).item()
                    clip_fractions.append(clip_fraction)

                    if mdl.clip_range_vf is None:
                        values_pred = values
                    else:
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values,
                            -clip_range_vf,
                            clip_range_vf,
                        )
                    value_loss = th.nn.functional.mse_loss(
                        rollout_data.returns, values_pred
                    )
                    value_losses.append(value_loss.item())

                    if entropy is None:
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)
                    entropy_losses.append(entropy_loss.item())

                    loss = (
                        policy_loss
                        + mdl.ent_coef * entropy_loss
                        + mdl.vf_coef * value_loss
                    )

                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl = (
                            th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        )
                        approx_kl_divs.append(approx_kl)

                    mdl.policy.optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(
                        mdl.policy.parameters(), mdl.max_grad_norm
                    )
                    mdl.policy.optimizer.step()

            mdl._n_updates += mdl.n_epochs
            ev = explained_variance(
                buffers[ag].values.flatten(), buffers[ag].returns.flatten()
            )
            # Basic logging to stdout
            print(
                f"[Arena][{ag}] update={update} loss={np.mean(pg_losses):.3f} value_loss={np.mean(value_losses):.3f} "
                f"entropy={np.mean(entropy_losses):.3f} clip_frac={np.mean(clip_fractions):.3f} ev={ev:.3f}"
            )
            # TensorBoard per-agent training scalars
            writer.add_scalar(
                f"{ag}/train/policy_loss", float(np.mean(pg_losses)), update
            )
            writer.add_scalar(
                f"{ag}/train/value_loss", float(np.mean(value_losses)), update
            )
            writer.add_scalar(
                f"{ag}/train/entropy_loss", float(np.mean(entropy_losses)), update
            )
            writer.add_scalar(
                f"{ag}/train/clip_fraction", float(np.mean(clip_fractions)), update
            )
            writer.add_scalar(f"{ag}/train/explained_variance", float(ev), update)

        # Periodic tournament and champion eval
        if (update + 1) % 5 == 0:
            summaries = tournament_round_robin(models, env_cfg, games_per_pair=5)
            print("\n[Tournament] per-agent win rates:")
            best_ag = None
            best_wr = -1.0
            for ag, s in summaries.items():
                print(
                    f"  {ag}: win_rate={s.win_rate:.3f} games={s.games} avg_len={s.avg_len:.1f}"
                )
                writer.add_scalar(
                    f"tournament/{ag}/win_rate", float(s.win_rate), update
                )
                writer.add_scalar(f"tournament/{ag}/games", float(s.games), update)
                if s.win_rate > best_wr:
                    best_wr = s.win_rate
                    best_ag = ag

            # Champion eval vs fixed opponents
            fixed_opps = ["balanced", "killer", "cautious", "winner"]
            if best_ag is not None:
                print(f"[Champion] {best_ag} vs fixed opponents")
                results = champion_vs_fixed(
                    models[best_ag], env_cfg, fixed_opps, games=25, deterministic=True
                )
                for res in results:
                    writer.add_scalar(
                        f"champion/{best_ag}/{res.opponent}/win_rate",
                        float(res.win_rate),
                        update,
                    )
                    writer.add_scalar(
                        f"champion/{best_ag}/{res.opponent}/avg_reward",
                        float(res.avg_reward),
                        update,
                    )
                    writer.add_scalar(
                        f"champion/{best_ag}/{res.opponent}/avg_length",
                        float(res.avg_length),
                        update,
                    )
                    print(
                        f"  vs {res.opponent}: win_rate={res.win_rate:.3f} avg_reward={res.avg_reward:.1f} avg_len={res.avg_length:.1f}"
                    )

    writer.flush()
    writer.close()
    env.close()
    return models


if __name__ == "__main__":
    cfg = EnvConfig(randomize_agent=True)
    train_arena_ippo(cfg, lr=3e-4, n_steps=1024, total_updates=1000, device="cpu")
