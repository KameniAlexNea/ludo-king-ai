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
| 1    | OPTIMIST         | 63   | 200   | 31.5       | 130.2     | 🥇 |
| 2    | PPO_LUDO_FINAL   | 60   | 200   | 30.0       | 129.5     | 🥈 |
| 3    | BALANCED         | 52   | 200   | 26.0       | 130.3     | 🥉 |
| 4    | PROBABILISTIC_V3 | 25   | 200   | 12.5       | 131.0     |  |

### Observations
* PPO now leads the mixed baseline tournament (Section 1), surpassing all scripted heuristics after normalization parity.
* In the self‑improvement setting PPO is a close second (–1.5 pp from top), indicating the gap to strongest heuristics has narrowed substantially.
* PPO achieves slightly lower average turns than top heuristics in its wins, suggesting improved conversion efficiency.
* Probabilistic_V3 remains a weak baseline and is still a good curriculum starter opponent.

### Next Improvement Ideas
1. Curriculum refinement: phase out Probabilistic_V3 earlier; introduce harder mixes once PPO >32% win rate.
2. Snapshot Lag: periodic frozen self opponent to reduce non‑stationarity spikes.
3. Late‑game shaping: small bonus for finishing final token sooner to reduce stall turns.
4. Masked sampling temperature (eval option) to probe robustness vs greedy policy overfit.
5. Phase metrics: break down win rate & capture ratio for early (≤50 turns) vs late (>50) segments.