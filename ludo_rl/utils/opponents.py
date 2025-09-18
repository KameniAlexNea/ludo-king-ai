def sample_opponents(opponent_names: list[str], progress: float, boundaries: list[float], rgn) -> list[str]:
    """Sample 3 opponent strategies using a single, simple weighted scheme.

    - Every strategy has a base weight (derived from the benchmark order).
    - A level multiplier (by training progress) boosts or dampens categories.
    - We always compute weights for ALL available candidates and sample 3
        without replacement proportional to those weights.
    """
    candidates = list(opponent_names)
    if len(candidates) <= 3:
        return candidates

    # Base weights from benchmark ranking (stronger => higher base)
    base_w = {
        "probabilistic_v2": 12.0,
        "probabilistic_v3": 11.0,
        "probabilistic": 10.0,
        "hybrid_prob": 9.0,
        "killer": 8.0,
        "cautious": 7.0,
        "defensive": 6.0,
        "balanced": 5.0,
        "winner": 4.0,
        "optimist": 3.0,
        "random": 2.0,
        "weighted_random": 2.0,
    }

    # Category for level multipliers
    easy = {"random", "weighted_random", "optimist"}
    medium = {"winner", "defensive", "balanced"}
    hard = {"cautious", "killer", "hybrid_prob", "probabilistic"}
    elite = {"probabilistic_v2", "probabilistic_v3"}

    # Progress-based multipliers (simple and monotonic)
    p = 0.0 if progress is None else float(progress)
    b = boundaries
    if p < b[0]:
        mult = {"easy": 1.5, "medium": 1.0, "hard": 0.6, "elite": 0.4}
    elif p < b[1]:
        mult = {"easy": 1.2, "medium": 1.1, "hard": 0.9, "elite": 0.6}
    elif p < b[2]:
        mult = {"easy": 0.8, "medium": 1.0, "hard": 1.2, "elite": 1.4}
    else:
        mult = {"easy": 0.5, "medium": 0.9, "hard": 1.3, "elite": 1.6}

    def cat(name: str) -> str:
        if name in easy: return "easy"
        if name in medium: return "medium"
        if name in hard: return "hard"
        if name in elite: return "elite"
        return "medium"

    # Compute final weights for all available candidates
    weights: list[float] = []
    for s in candidates:
        w0 = base_w.get(s, 1.0)
        w = w0 * mult.get(cat(s), 1.0)
        weights.append(max(1e-6, float(w)))

    # Weighted sample 3 without replacement
    chosen: list[str] = []
    cand = candidates[:]
    wts = weights[:]
    for _ in range(3):
        total = sum(wts)
        r = rgn.random() * total
        cum = 0.0
        idx = 0
        for i, w in enumerate(wts):
            cum += w
            if r <= cum:
                idx = i
                break
        chosen.append(cand.pop(idx))
        wts.pop(idx)
    return chosen

def build_opponent_triplets(baselines: list[str], n_games: int) -> list[list[str]]:
    """Build a list of opponent triplets for evaluation games.

    Generates permutations of the provided baselines to create diverse 3-opponent combinations,
    then repeats or truncates to reach the desired number of games.
    """
    import itertools

    uniq = list(dict.fromkeys(baselines))  # deduplicate keeping order
    triplets = []
    if len(uniq) >= 3:
        for comb in itertools.combinations(uniq, 3):
            for perm in itertools.permutations(comb, 3):
                triplets.append(list(perm))
    else:
        # If fewer than 3 provided, pad with repeats to reach 3
        pad = (uniq * 3)[:3]
        triplets = [pad]

    # Repeat or truncate to reach n_games size
    if len(triplets) == 0:
        triplets = [(["random", "random", "random"])]
    while len(triplets) < n_games:
        triplets.extend(triplets)
    triplets = triplets[: n_games]
    return triplets