"""Ranking metrics with multiple relevant items, concentration and bootstrap CIs."""
import math

import numpy as np


def mrr(ranked, positives):
    for i, c in enumerate(ranked, 1):
        if c in positives:
            return 1.0 / i
    return 0.0


def hit_at_k(ranked, positives, k):
    return 1.0 if any(c in positives for c in ranked[:k]) else 0.0


def ndcg_at_k(ranked, positives, k):
    dcg = sum(1.0 / math.log2(i + 1) for i, c in enumerate(ranked[:k], 1) if c in positives)
    n_rel = min(len(positives), k)
    if n_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg


def gini(values):
    x = np.sort(np.asarray(values, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return math.nan
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def bootstrap_ci(values, n_boot=1000, seed=42, alpha=0.05):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def gini_over_universe(counts, universe):
    """Gini of `counts` (dict candidate -> count) over every candidate in `universe`,
    counting zero for candidates that never received a recommendation."""
    return gini([counts.get(c, 0) for c in universe])
