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

Final tournament standings after refactor (mixed fours; 1650 games aggregate per strategy). Win Rate % = Wins / Games.

| Rank | Strategy          | Wins | Games | Win Rate % | Avg Turns | Medal |
|------|-------------------|------|-------|-----------:|----------:|:------|
| 1    | BALANCED          | 548  | 1650  | 33.2       | 125.1     | 🥇 |
| 2    | CAUTIOUS          | 534  | 1650  | 32.4       | 124.4     | 🥈 |
| 3    | PROBABILISTIC     | 522  | 1650  | 31.6       | 124.2     | 🥉 |
| 4    | KILLER            | 494  | 1650  | 29.9       | 124.4     |  |
| 5    | HYBRID_PROB       | 488  | 1650  | 29.6       | 124.0     |  |
| 6    | PROBABILISTIC_V2  | 484  | 1650  | 29.3       | 124.6     |  |
| 7    | PROBABILISTIC_V3  | 475  | 1650  | 28.8       | 124.1     |  |
| 8    | WINNER            | 441  | 1650  | 26.7       | 113.9     |  |
| 9    | DEFENSIVE         | 433  | 1650  | 26.2       | 114.8     |  |
| 10   | WEIGHTED_RANDOM   | 223  | 1650  | 13.5       | 115.5     |  |
| 11   | RANDOM            | 162  | 1650  | 9.8        | 114.3     |  |
| 12   | OPTIMIST          | 141  | 1650  | 8.5        | 125.9     |  |

📊 DETAILED PERFORMANCE ANALYSIS 📊
======================================================================

| Strategy          | Captures | Finished | Efficiency |
|-------------------|----------|----------|------------|
| KILLER            | 11349    | 1976     | 1.20       |
| WINNER            | 5992     | 1764     | 1.07       |
| OPTIMIST          | 10389    | 569      | 0.34       |
| DEFENSIVE         | 5872     | 1732     | 1.05       |
| BALANCED          | 10941    | 2197     | 1.33       |
| PROBABILISTIC     | 10995    | 2088     | 1.27       |
| PROBABILISTIC_V3  | 11271    | 1900     | 1.15       |
| PROBABILISTIC_V2  | 11091    | 1936     | 1.17       |
| HYBRID_PROB       | 11056    | 1952     | 1.18       |
| RANDOM            | 5304     | 648      | 0.39       |
| WEIGHTED_RANDOM   | 5418     | 892      | 0.54       |
| CAUTIOUS          | 11123    | 2136     | 1.29       |

### Key Insights:
* **BALANCED** dominates head-to-head matchups with exceptional performance vs OPTIMIST (41%), WEIGHTED_RANDOM (40%), and RANDOM (34%)
* **CAUTIOUS** excels against OPTIMIST (37%) and weaker strategies but shows vulnerability to BALANCED (28%) and PROBABILISTIC (28%)
* **PROBABILISTIC variants** cluster tightly (26-32% range), with V3 slightly trailing V2 in most matchups
* **KILLER** maintains strong aggression vs weak opponents (OPTIMIST: 32%, RANDOM: 33%) but struggles against BALANCED (27%)
* **PROBABILISTIC** shows balanced performance across the board (28-37% range) with strong results vs OPTIMIST (36%) and RANDOM (37%)
* **HYBRID_PROB** performs consistently well (27-33% range) with particular strength vs OPTIMIST (33%) and RANDOM (34%)
* **DEFENSIVE** holds middle ground (22-35% range) with solid performance vs OPTIMIST (32%) but weakness vs BALANCED (22%)
* **WINNER** shows similar middle-tier performance (21-35% range) with strength vs weaker opponents but struggles vs BALANCED (21%)
* **OPTIMIST** severely underperforms across all matchups (7-10% range) - requires major strategic overhaul
* **WEIGHTED_RANDOM** and **RANDOM** remain consistently weak (10-15% and 8-13% ranges respectively) against all opponents

### Head-to-Head Strategic Analysis:

#### 🏆 **Top Strategies vs Top Strategies:**
* **BALANCED vs CAUTIOUS**: BALANCED holds narrow edge (28-31% range) - very competitive matchup between top performers
* **BALANCED vs PROBABILISTIC**: Nearly even matchup (29-29%) - BALANCED's slight advantage in consistency
* **BALANCED vs KILLER**: BALANCED dominates (34-27%) - clear superiority in top-tier competition
* **CAUTIOUS vs PROBABILISTIC**: CAUTIOUS leads slightly (32-28%) - CAUTIOUS's targeted aggression effective
* **CAUTIOUS vs KILLER**: Very close matchup (28-28%) - aggressive styles cancel out
* **PROBABILISTIC vs KILLER**: PROBABILISTIC leads (32-29%) - probabilistic approach edges out pure aggression

#### 🗑️ **Bottom Strategies vs Bottom Strategies:**
* **OPTIMIST vs RANDOM**: OPTIMIST barely leads (9-13%) - both extremely weak, minimal differentiation
* **OPTIMIST vs WEIGHTED_RANDOM**: OPTIMIST slightly ahead (10-15%) - WEIGHTED_RANDOM shows marginal improvement over pure random
* **RANDOM vs WEIGHTED_RANDOM**: Very close matchup (12-15%) - WEIGHTED_RANDOM's heuristics provide small but consistent edge

#### 🎯 **Top Strategies vs Bottom Strategies:**
* **BALANCED vs OPTIMIST**: BALANCED dominates (41-8%) - overwhelming superiority against weakest opponent
* **BALANCED vs RANDOM**: BALANCED crushes (34-10%) - clear exploitation of random play weaknesses
* **BALANCED vs WEIGHTED_RANDOM**: BALANCED leads decisively (40-13%) - strategic depth overwhelms heuristic improvements
* **CAUTIOUS vs OPTIMIST**: CAUTIOUS excels (37-7%) - perfect matchup for CAUTIOUS's conservative aggression
* **CAUTIOUS vs RANDOM**: CAUTIOUS dominates (34-10%) - systematic play overwhelms randomness
* **CAUTIOUS vs WEIGHTED_RANDOM**: CAUTIOUS leads strongly (35-14%) - targeted playstyle exploits weighted random weaknesses
* **PROBABILISTIC vs OPTIMIST**: PROBABILISTIC overwhelms (36-8%) - probabilistic calculation maximizes advantage
* **PROBABILISTIC vs RANDOM**: PROBABILISTIC crushes (37-10%) - calculated risk-taking exploits random behavior
* **PROBABILISTIC vs WEIGHTED_RANDOM**: PROBABILISTIC leads convincingly (34-12%) - strategic depth prevails
* **KILLER vs OPTIMIST**: KILLER performs well (32-9%) - aggressive style effective against passive opponent
* **KILLER vs RANDOM**: KILLER dominates (33-9%) - capture-focused strategy exploits random movement patterns
* **KILLER vs WEIGHTED_RANDOM**: KILLER leads solidly (32-15%) - aggressive playstyle overcomes weighted random heuristics

**Important note:**  
4 players are selected (combination of all strategies) and 10 match-up are organized between them.

# Tournament Configuration  
MAX_TURNS_PER_GAME=1000  
GAMES_PER_MATCHUP=10