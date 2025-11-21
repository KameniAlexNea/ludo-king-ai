# Ludo King PPO Training Guide (SB3)

This document explains how the Ludo King agent is trained using Stable-Baselines3 (SB3), focusing on the PPO objective, loss components, reward shaping, and the roles of key hyperparameters (gamma, GAE-lambda, vf_coef, ent_coef). It is written as an advanced, practical guide grounded in this codebase.

## Stack Overview

- Algorithm: Proximal Policy Optimization (PPO) from `sb3_contrib` as `MaskablePPO`.
- Policy: `MultiInputPolicy` with a custom token-sequence features extractor (LSTM or Transformer) that outputs a flat feature vector.
- Loss: Clipped policy gradient + value function regression + entropy bonus.
- Schedules: Cosine warmup + decay for learning rate, entropy coefficient, and clip range.
- Rewards: Centralized dense signal with potential-based Risk/Opportunity (RO) shaping and opponent-progress penalties.

Code anchors:
- Training orchestration: `train.py`
- Scheduling utilities: `tools/scheduler.py`
- Reward configuration: `ludo_rl/ludo_king/config.py` (`Reward` dataclass)
- Reward computation and shaping: `ludo_rl/ludo_king/reward.py`
- Game/env glue: `ludo_rl/ludo_king/game.py`, `ludo_rl/ludo_env.py`, `ludo_rl/ludo_king/simulator.py`
- Extractors: `ludo_rl/extractor.py`

## PPO Objective

PPO maximizes a clipped surrogate objective that constrains policy updates away from the behavior policy.

- Probability ratio: $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$.
- Clipped objective:
$$
\mathcal{L}_{\text{clip}}(\theta) = \mathbb{E}_t\left[\min\big( r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\, A_t \big)\right].
$$

SB3 minimizes total loss (so the policy term enters negated):
$$
\mathcal{L}_{\text{total}} = -\mathcal{L}_{\text{clip}}\ +\ c_1\,\underbrace{(R_t - V_\theta(s_t))^2}_{\text{value loss}}\ -\ c_2\,\underbrace{\mathcal{H}[\pi_\theta(\cdot\mid s_t)]}_{\text{entropy}}.
$$
- $\epsilon$ is the clip range (`--clip-range`).
- $c_1$ is `vf_coef`, the value loss weight.
- $c_2$ is `ent_coef`, the entropy bonus weight.

SB3 additionally supports early stopping by monitoring approximate KL (`approx_kl`) against `target_kl`.

## Returns, GAE, and Gamma

Let $\gamma$ be the discount factor (`--gamma`) and $\lambda$ be the GAE parameter (`--gae-lambda`). Define the TD error:
$$
\delta_t = r_t + \gamma\, V(s_{t+1}) - V(s_t).
$$
Generalized Advantage Estimation (GAE) computes advantages as an exponentially weighted sum of TD errors:
$$
A_t = \sum_{l=0}^{T-t-1} (\gamma\,\lambda)^l\, \delta_{t+l}.
$$
The bootstrapped return used for the value loss is:
$$
R_t^{(\lambda)} = A_t + V(s_t).
$$
- Higher $\gamma$ (e.g., 0.99) emphasizes long-horizon outcomes (finishing tokens, winning).
- $\lambda\in[0,1]$ trades bias/variance. Values around 0.95 are a robust default in delayed-reward board games.

## Loss Components and Schedules

- Policy loss: drives improvement using clipped policy gradients on $A_t$.
- Value loss (weighted by `vf_coef`, default 2.0): stabilizes bootstrapped returns and combats value underfitting.
- Entropy bonus (weighted by `ent_coef`, default ~0.05): maintains exploration pressure.

This project uses cosine schedules (with short warmup) for multiple coefficients via `tools/scheduler.py`:
- Learning rate: $\text{lr}(t) \in [\text{lr}_{\min}, \text{lr}_{\max}]$ with warmup to $\text{lr}_{\max}$ then cosine decay.
- Entropy coef: annealed from $\approx 0.3\,\text{ent}_{\max}$ up to $\text{ent}_{\max}$ during warmup, then decays by cosine.
- Clip range: warmups to `clip_max`, then cosine to `\approx 0.6 × clip_max`.

This encourages stable early exploration and progressively conservative updates.

## Action Masking

Illegal Ludo moves are masked using `MaskablePPO`, preventing gradient credit from flowing through invalid actions and improving sample-efficiency and stability.

## Reward Design in Ludo King

Rewards combine sparse event signals with dense, potential-based shaping to provide informative gradients every step.

### Base Event Signals (examples)
- Win/Lose/Draw: large terminal rewards, scaled to avoid exploding value loss.
- Finish token: positive reward per piece reaching home.
- Capture opponent / got captured: positive/negative.
- Blockade formed/hit and illegal attempts: small dense signals and penalties.
- Exit home / safe squares / skipped turn: small shaping for momentum and safety.

See `ludo_rl/ludo_king/config.py` `Reward` dataclass for exact magnitudes.

### Potential-Based RO Shaping

We add a shaping term based on a learned/handcrafted potential $\Phi(s)$ that aggregates:
- Progress towards finish.
- Immediate capture opportunities and capture risk (with small lookahead depth `ro_depth`).
- Opponents near finishing.

The shaping bonus per transition is:
$$
F(s,a,s') = \alpha\,\big( \gamma_{\text{shape}}\, \Phi(s') - \Phi(s) \big),
$$
where $\alpha$ is `shaping_alpha` and $\gamma_{\text{shape}}$ is `shaping_gamma`.

Critically, potential-based shaping preserves the optimal policy in the underlying MDP while providing dense guidance. In code, `reward.py` computes `phi_before` and `phi_after`, then adds the discounted difference scaled by `shaping_alpha`.

### Opponent-Progress Penalties

To make the agent feel pressure before a terminal loss, we add small penalties when opponents:
- Exit a token from home.
- Finish a token.
- Win the game (additional slight negative given to others on that step).

These are accumulated after simulating opponents’ turns so they are visible in the agent’s immediate reward stream (`simulator.py` and `ludo_env.py`).

### Putting It Together

At each environment step for the learning agent, the final reward is:
$$
\begin{aligned}
\tilde r_t &= r_t^{\text{events}} \quad (\text{captures, finishes, safety, invalid, etc.}) \\
&\quad + \underbrace{\alpha\,(\gamma_{\text{shape}}\,\Phi(s_{t+1}) - \Phi(s_t))}_{\text{RO shaping}} \\
&\quad + r_t^{\text{opp-progress}}. 
\end{aligned}
$$
This $\tilde r_t$ feeds GAE and the PPO losses.

## Network Architecture

- Token-sequence features (`ludo_rl/extractor.py`):
  - Embeddings over position, color, piece index, time step, dice history, current dice, and player.
  - Two variants: LSTM-based (`LudoCnnExtractor`) or Transformer-based (`LudoTransformerExtractor`).
  - The extractor outputs a `features_dim = EMBED_DIM` vector.
- Policy/Value heads: MLPs defined by `PI` and `VF` layer sizes (`net_arch` in `train.py`).
- Key knobs in `NetworkConfig`:
  - `TOKEN_EMBED_DIM` controls embedding/hidden width inside the extractor.
  - `EMBED_DIM` controls the final features dimension returned to the policy.
  - `PI` and `VF` define actor/critic MLP widths/depths.

This separation lets you scale extractor width and head capacity independently.

## Worked Ludo Examples

1) Small progress with risk reduction
- Move a token from 5→6, leaving an opponent’s potential capture window. Base event: tiny progress reward.
- RO: risk term decreases (good), progress term increases slightly.
- Net: $\tilde r_t$ positive but modest; $A_t$ becomes positive if the value underestimated this safety gain.

2) Capture and form a blockade
- Base event: capture reward + blockade formation bonus.
- RO: future capture risk reduced (positive), opponent opportunity reduced.
- Net: $\tilde r_t$ noticeably positive; large $A_t$ pushes policy towards similar tactics in similar contexts.

3) Opponent finishes a piece between your moves
- Base event: none on your turn, but opponent-progress penalty applied before your next decision.
- Net: $\tilde r_t < 0$, shifting $A_t$ negative and nudging the policy to adopt urgency patterns that avoid falling behind.

4) Illegal move attempt into opponent blockade (when not masked)
- Base event: penalty for hitting blockade; with masking, such actions are excluded and do not receive credit.
- RO: no positive shaping; value target decreases slightly; policy learns to avoid these states/actions.

## Reading Training Logs (What to Watch)

- `approx_kl` near but under `target_kl`: healthy updates; too low can mean timid learning, too high can destabilize.
- `entropy_loss` magnitude shrinking over time: exploration annealing as the policy sharpens.
- `explained_variance` ~0.80–0.85: value net tracking returns; sustained <0.7 suggests increasing `vf_coef` or capacity.
- `value_loss` spikes: often from overly large terminal scales or poorly tuned `vf_coef`.
- `ep_rew_mean` trends: noisy but should improve across iterations with steady EV.

## Hyperparameter Roles and Practical Tuning

- `gamma` (default 0.99):
  - Higher emphasizes long-term outcomes like finishing and winning.
  - If learning stalls on near-term tactics (exiting home, safe squares), slightly lower gamma can help; otherwise keep 0.99.

- `gae_lambda` (default 0.95):
  - Lower reduces variance but increases bias; in board games 0.95 is a strong default.

- `vf_coef` (default 2.0):
  - Increase if `explained_variance` is low or value loss dominates policy changes.
  - Decrease if the value head overfits/stiffens policy improvements.

- `ent_coef` (default ~0.05 with cosine schedule):
  - Higher encourages exploration (useful early with diverse opponents).
  - Anneal down over time (scheduler) to consolidate gains.

- `clip_range` (e.g., 0.20–0.25 with cosine to ~60%):
  - Smaller clamps prevent destructive updates but can slow progress;
  - Larger allows faster learning but risks instability; use `target_kl` as guardrails.

- Learning rate (cosine schedule):
  - Warmup then decay improves early stability and later fine-tuning.
  - Typical peak LR in this repo ranges 5e-4 to 1e-3; if KL is tiny and EV stalls, modestly increase LR; if KL spikes, reduce.

## Why Potential-Based Shaping Works Here

Ludo has sparse, delayed outcomes (winning) and many near-symmetric actions. Potential-based shaping:
- Provides a dense gradient aligned with intuitive tactics (progress, captures, safety).
- Preserves optimal policies ($\gamma_{\text{shape}}\,\Phi(s') - \Phi(s)$ form).
- Combats credit assignment issues caused by long horizons and opponent interleaving.

Opponents’ progress penalties act as “urgency signals,” letting the agent sense being behind before a terminal loss hits, which improves policy reactivity.

## Reproducing and Modifying Training

- Key CLI flags: `--n-steps`, `--batch-size`, `--n-epochs`, `--gamma`, `--gae-lambda`, `--clip-range`, `--learning-rate`, `--target-kl`, `--ent-coef`, `--vf-coef`, `--use-transformer`.
- Network config via env vars: `EMBED_DIM`, `TOKEN_EMBED_DIM`, `PI`, `VF`.
- Reward shaping via env vars: `SHAPING_USE`, `SHAPING_ALPHA`, `SHAPING_GAMMA`, `RO_DEPTH`, component weights, and opponent penalties.

Example (medium-capacity nets, exploratory yet safe updates):
```
export EMBED_DIM=128
export TOKEN_EMBED_DIM=16
export PI=128,64
export VF=128,128

nohup python train.py \
  --n-steps 4096 \
  --n-epochs 6 \
  --batch-size 2048 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --ent-coef 0.05 \
  --vf-coef 2.0 \
  --target-kl 0.012 \
  --device cuda \
  --learning-rate 5e-4 \
  --clip-range 0.20 \
  &> nohup.out &
```

## Troubleshooting Patterns

- High `approx_kl` spikes: reduce LR and/or `clip_range`, lower `target_kl`.
- Low `explained_variance` (<0.7): increase `vf_coef`, widen `VF` net, or slightly reduce terminal reward scales.
- Entropy collapses too early: raise `ent_coef` or extend the scheduler warmup.
- Value loss dominates policy loss for long: re-check reward magnitudes; consider normalizing/clinching terminal scales.
- Poor responsiveness to opponents finishing tokens: verify opponent-progress penalties are enabled and accumulated in env steps.

## References

- PPO: Schulman et al. “Proximal Policy Optimization Algorithms.”
- GAE: Schulman et al. “High-Dimensional Continuous Control Using Generalized Advantage Estimation.”
- Potential-based shaping: Ng, Harada, Russell. “Policy Invariance Under Reward Transformations.”
