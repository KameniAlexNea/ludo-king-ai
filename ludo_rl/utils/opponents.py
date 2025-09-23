import itertools
import random
from typing import List, Optional, Union

import numpy as np


def sample_opponents(
    opponent_names: List[str],
    progress: Optional[float],
    boundaries: List[float],
    rgn: Union[random.Random, np.random.Generator],
) -> List[str]:
    """Sample 3 opponent strategies using a weighted scheme.

    - Each strategy has a base weight (derived from benchmark order).
    - Multipliers adjust weights based on training progress.
    - Always compute weights for all candidates and sample 3 without replacement.
    """
    candidates = list(opponent_names)
    if len(candidates) <= 3:
        return candidates

    # Base weights from benchmark ranking
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

    # Categories for multipliers
    easy = {"random", "weighted_random", "optimist"}
    medium = {"winner", "defensive", "balanced"}
    hard = {"cautious", "killer", "hybrid_prob", "probabilistic"}
    elite = {"probabilistic_v2", "probabilistic_v3"}

    # Progress multipliers
    p = 0.0 if progress is None else float(progress)
    b = boundaries
    if p < b[0]:
        mult = {"easy": 3.0, "medium": 1.0, "hard": 0.3, "elite": 0.1}
    elif p < b[1]:
        mult = {"easy": 2.0, "medium": 1.0, "hard": 0.6, "elite": 0.4}
    elif p < b[2]:
        mult = {"easy": 0.6, "medium": 1.0, "hard": 1.4, "elite": 2.0}
    else:
        mult = {"easy": 0.3, "medium": 0.8, "hard": 1.5, "elite": 2.5}

    def cat(name: str) -> str:
        if name in easy:
            return "easy"
        if name in medium:
            return "medium"
        if name in hard:
            return "hard"
        if name in elite:
            return "elite"
        return "medium"

    # Compute final weights
    weights: List[float] = []
    for s in candidates:
        w0 = base_w.get(s, 1.0)
        w = w0 * mult.get(cat(s), 1.0)
        weights.append(max(1e-6, float(w)))

    # Weighted sample 3 without replacement
    chosen: List[str] = []
    cand = candidates[:]
    wts = weights[:]
    for _ in range(3):
        total = sum(wts)
        r = rgn.random() * total if hasattr(rgn, "random") else rgn.uniform(0, total)
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


def build_opponent_triplets(baselines: List[str], n_games: int) -> List[List[str]]:
    """Build a list of opponent triplets for evaluation games.

    Generates permutations of the provided baselines to create diverse 3-opponent combinations,
    then repeats or truncates to reach the desired number of games.
    """
    from itertools import cycle, islice

    uniq = list(dict.fromkeys(baselines))  # deduplicate, keep order
    triplets: List[List[str]] = []

    if len(uniq) >= 3:
        for comb in itertools.combinations(uniq, 3):
            for perm in itertools.permutations(comb, 3):
                triplets.append(list(perm))
    elif len(uniq) > 0:
        # If fewer than 3, pad with repeats
        pad = (uniq * 3)[:3]
        triplets = [pad]
    else:
        # Fallback if nothing provided
        triplets = [["random", "random", "random"]]

    # Cycle through triplets to reach exactly n_games
    triplets = list(islice(cycle(triplets), n_games))
    return triplets
