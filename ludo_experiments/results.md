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

### 3. Scripted AI Benchmark (Updated)

Final tournament standings after refactor (mixed fours; 660 games aggregate per strategy). Win Rate % = Wins / Games.

| Rank | Strategy          | Wins | Games | Win Rate % | Avg Turns | Medal |
|------|-------------------|------|-------|-----------:|----------:|:------|
| 1    | CAUTIOUS          | 216  | 660   | 32.7       | 124.8     | 🥇 |
| 2    | BALANCED          | 215  | 660   | 32.6       | 122.9     | 🥈 |
| 3    | KILLER            | 202  | 660   | 30.6       | 123.6     | 🥉 |
| 4    | PROBABILISTIC     | 202  | 660   | 30.6       | 122.6     |  |
| 5    | PROBABILISTIC_V3  | 200  | 660   | 30.3       | 123.4     |  |
| 6    | HYBRID_PROB       | 200  | 660   | 30.3       | 125.5     |  |
| 7    | PROBABILISTIC_V2  | 194  | 660   | 29.4       | 123.4     |  |
| 8    | WINNER            | 184  | 660   | 27.9       | 115.2     |  |
| 9    | DEFENSIVE         | 167  | 660   | 25.3       | 113.8     |  |
| 10   | WEIGHTED_RANDOM   | 75   | 660   | 11.4       | 117.0     |  |
| 11   | OPTIMIST          | 63   | 660   | 9.5        | 127.6     |  |
| 12   | RANDOM            | 62   | 660   | 9.4        | 113.6     |  |

Key Notes (Updated):
* Cautious climbs to #1 post threat unification; exploits volatile opponents though still loses narrowly to Balanced head‑to‑head.
* Balanced retains near‑top efficiency with slightly faster average turns than Cautious.
* Killer reformed: major jump in win rate after safer capture logic.
* Probabilistic variants now tightly clustered (30–30.6%), differentiation reduced—tuning opportunity.
* Defensive / Winner show low average turns but reduced conversion—over-conservative filtering suspected.
* Optimist still underperforming; aggression not converting to finishes—needs adaptive risk modulation.