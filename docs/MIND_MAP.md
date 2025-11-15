## Project Layout

```
ludo_rl
├─ __init__.py → loads .env so simulator/env can read opponent strategy settings
├─ ludo_env.LudoEnv (Gymnasium Env)
│  ├─ wraps ludo_king.simulator.Simulator to expose observation dict & masked Discrete(4) actions
│  ├─ handles invalid moves, turn limit, reward shaping, and rendering snapshots
│  └─ converts Game.board into a 10‑channel board tensor + dice_roll token
├─ ludo_king.simulator.Simulator
│  ├─ owns Game, tracks agent_index
│  ├─ runs opponents’ turns (respecting extra rolls)
│  └─ can be driven by environment strategies (registry) or random fallback
├─ ludo_king.game.Game
│  ├─ instantiates 2 or 4 Player objects and provides dice + rule enforcement
│  ├─ enforces entry, home column, safe squares, captures, blockades, extra turns
│  └─ builds per‑agent board tensors via Board.build_tensor
├─ ludo_king.board.Board
│  ├─ absolute↔relative mapping, safe squares and channel construction
│  └─ counts pieces per player/channel for tensor features
├─ ludo_king.player.Player
│  ├─ keeps Piece state, win detection
│  ├─ chooses moves via strategies (lazy instantiation by name)
│  └─ falls back to random legal move if requested heuristic is unknown
├─ strategy package
│  ├─ features.build_move_options turns env observation into StrategyContext
│  ├─ BaseStrategy + concrete heuristics (defensive, killer, etc.) score MoveOption
│  └─ registry.create/available expose factories to simulator & players
├─ extractor.LudoCnnExtractor / LudoTransformerExtractor
│  ├─ convert observation dict into feature vectors for MaskablePPO
│  └─ fuse CNN/Transformer encodings with per‑piece embeddings and dice token
├─ tools (arguments, scheduler, evaluate, tournaments, imitation)
│  ├─ evaluate.py — supports 2‑player (opposite seats) and 4‑player lineups
│  ├─ tournament.py — strategy league for 2 or 4 players
│  └─ llm_vs_models.py — LLM/RL/Static mixed matches for 2 or 4 players
└─ train.py
   ├─ parses CLI args, configures MaskablePPO w/ custom extractor
   └─ runs vectorized envs, callbacks (checkpoints, entropy annealing, profiler)
```

- LudoEnv mediates RL interaction: builds masked actions, enforces rewards, loops until player or opponents advance, and emits 10‑channel observations.
- Simulator orchestrates turns: applies agent move (in env), simulates opponents with heuristic strategies, and ensures extra‑turn logic.
- Core rules live in ludo_king Game + Board + Piece/Player; reward.compute_move_rewards still produces shaped returns for PPO.
- Strategy module supplies configurable heuristics; features.build_move_options transforms env data into StrategyContext so Player.choose can score moves consistently.
- extractor.py houses CNN/Transformer feature pipelines that embed board channels, per‑piece context, and dice roll before feeding MaskablePPO during training (train.py).
