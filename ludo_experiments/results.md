## Tournament Results

### 1. PPO vs Other Strategies

| Rank | Model            | Wins | Games | Win Rate % | Avg Turns | Medal |
|------|------------------|------|-------|-----------:|----------:|:------|
| 1    | PPO_LUDO_FINAL   | 72   | 200   | 36.0       | 127.5     | 🥇 |
| 2    | BALANCED         | 52   | 200   | 26.0       | 127.8     | 🥈 |
| 3    | OPTIMIST         | 51   | 200   | 25.5       | 128.3     | 🥉 |
| 4    | PROBABILISTIC_V3 | 25   | 200   | 12.5       | 128.2     |  |

### 2. PPO Self-Improvement (PPO vs Frozen PPO)

| Rank | Model            | Wins | Games | Win Rate % | Avg Turns | Medal |
|------|------------------|------|-------|-----------:|----------:|:------|
| 1    | OPTIMIST         | 62   | 200   | 31.0       | 127.3     | 🥇 |
| 2    | BALANCED         | 61   | 200   | 30.5       | 129.0     | 🥈 |
| 3    | PPO_LUDO_FINAL   | 55   | 200   | 27.5       | 127.6     | 🥉 |
| 4    | PROBABILISTIC_V3 | 22   | 200   | 11.0       | 128.0     |  |

### Observations
* Heuristic strategies (Balanced / Optimist) still outperform current PPO snapshot.
* PPO maintains competitive average turns (slightly fewer turns when not winning – possibly earlier eliminations / lower finishing performance).
* Probabilistic_V3 underperforms relative to others; may offer exploitable patterns for PPO curriculum.

### Next Improvement Ideas
1. Curriculum: start with weaker opponents (Probabilistic_V3 + Random) then introduce Balanced/Optimist.
2. Snapshot Lag: freeze opponent PPO every N updates to stabilize target distribution.
3. Reward Tuning: modest boost to capture/finish shaping or slight reduction of terminal magnitude gap.
4. Action Mask Entropy: encourage exploration among legal moves by renormalizing masked policy logits.
5. Evaluation Split: Track separate metrics for early-game (<=50 turns) vs late-game to locate weakness.