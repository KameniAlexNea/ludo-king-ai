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
| 1    | PPO_LUDO_FINAL   | 77   | 200   | 38.5       | 131.3     | 🥇 |
| 2    | BALANCED         | 53   | 200   | 26.5       | 131.3     | 🥈 |
| 3    | OPTIMIST         | 44   | 200   | 22.0       | 131.9     | 🥉 |
| 4    | PROBABILISTIC_V3 | 26   | 200   | 13.0       | 131.9     |  |

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

---

### 3. Scripted AI Benchmark (No RL / PPO)

Standalone heuristic strategies head‑to‑head (480 games total, mixed fours). Win Rate % = Wins / Games.

| Rank | Strategy          | Wins | Games | Win Rate % | Avg Turns | Medal |
|------|-------------------|------|-------|-----------:|----------:|:------|
| 1    | BALANCED          | 176  | 480   | 36.7       | 124.5     | 🥇 |
| 2    | PROBABILISTIC_V2  | 163  | 480   | 34.0       | 125.9     | 🥈 |
| 3    | HYBRID_PROB       | 154  | 480   | 32.1       | 126.8     | 🥉 |
| 4    | PROBABILISTIC_V3  | 152  | 480   | 31.7       | 126.6     |  |
| 5    | PROBABILISTIC     | 151  | 480   | 31.5       | 128.0     |  |
| 6    | DEFENSIVE         | 144  | 480   | 30.0       | 115.9     |  |
| 7    | CAUTIOUS          | 138  | 480   | 28.7       | 116.9     |  |
| 8    | WINNER            | 135  | 480   | 28.1       | 115.9     |  |
| 9    | OPTIMIST          | 41   | 480   | 8.5        | 127.2     |  |
| 10   | RANDOM            | 38   | 480   | 7.9        | 113.7     |  |
| 11   | KILLER            | 24   | 480   | 5.0        | 131.7     |  |

Key Notes:
* Balanced leads—moderate risk handling beating more complex probabilistic variants.
* Probabilistic_V2 outperforms other probabilistic versions; Hybrid needs tuning to surpass it.
* Killer over-indexes on captures (very low finish conversion efficiency).
* Defensive / Cautious cluster mid-table with faster average turns (earlier game resolutions, often not their wins).
* Optimist aggression not translating to finishes; high average turns suggests stalled conversions.