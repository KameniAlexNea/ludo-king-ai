# Test Suite Overview

Structured unittest-based suite (no pytest) covering core engine, heuristic strategies, and both RL environments (classic multi-seat and single-seat Option B).

## Layout
- engine/: Core `LudoGame` mechanics (movement, captures, home column, win conditions, special rules like consecutive sixes)
- strategies/: Heuristic strategy decision validity
- rl_classic/: Classic multi-seat environment API, rewards, PPO wrapper behaviors, normalization
- rl_single_seat/: Single-seat environment behavior (training color randomization, masking, rewards)

## Key Behaviors Covered
- Token lifecycle: home -> active -> home column -> finished (including exact-finish requirement)
- Move validation: exit on 6, illegal three-sixes forfeiture
- Capture flow: opponent token reset + extra turn flag
- Reward shaping: presence of step breakdown & components, terminal rewards on win/draw
- Action masking & illegal handling: classic fallback vs single-seat masked autocorrect
- Extra turns: via 6, capture, finish chains (both env variants)
- Observation integrity: size, normalization bounds when VecNormalize stats present
- PPO strategy wrapper: deterministic vs stochastic action selection, mask application

## Edge Cases Included
- Step call after environment termination (planned) ensuring stable output
- Illegal action fallback (classic) vs masked autocorrect (single-seat)
- Forced terminal scenario by marking tokens finished (single-seat planned)

## Running Tests
Standard discovery:
```
python -m unittest discover -v
```
Or target a subset:
```
python -m unittest tests.engine.test_home_column_and_sixes
```

## Future Additions
- Opponent simulator statistical invariants (classic env)
- Performance regression guards (timing thresholds) under a separate slow marker (opt-in)
- Coverage badge generation (if CI introduced)
