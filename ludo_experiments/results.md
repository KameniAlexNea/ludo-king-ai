## Tournament Results

### 1. PPO vs Other Strategies

| Rank | Model            | Wins | Games | Win Rate % | Avg Turns | Medal |
|------|------------------|------|-------|-----------:|----------:|:------|
| 1    | BALANCED         | 65   | 200   | 32.5       | 130.1     | 🥇 |
| 2    | OPTIMIST         | 57   | 200   | 28.5       | 128.8     | 🥈 |
| 3    | PPO_LUDO_FINAL   | 51   | 200   | 25.5       | 128.3     | 🥉 |
| 4    | PROBABILISTIC_V3 | 27   | 200   | 13.5       | 129.0     |  |

### 2. PPO Self-Improvement (PPO vs Frozen PPO)

| Rank | Model            | Wins | Games | Win Rate % | Avg Turns | Medal |
|------|------------------|------|-------|-----------:|----------:|:------|
| 1    | BALANCED         | 63   | 200   | 31.5       | 128.8     | 🥇 |
| 2    | OPTIMIST         | 62   | 200   | 31.0       | 128.0     | 🥈 |
| 3    | PPO_LUDO_FINAL   | 42   | 200   | 21.0       | 126.7     | 🥉 |
| 4    | PROBABILISTIC_V3 | 33   | 200   | 16.5       | 128.1     |  |

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