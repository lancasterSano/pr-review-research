"""Temporal split per repository: warm-up, then 60/20/20 train/val/test by creation time."""
import numpy as np


def assign_splits(created, warmup_days=90, fractions=(0.6, 0.2, 0.2)):
    t = np.array([c.timestamp() for c in created])
    order = np.argsort(t, kind="stable")
    start = t[order[0]] + warmup_days * 86400.0
    labels = np.empty(len(t), dtype=object)
    eligible = [i for i in order if t[i] >= start]
    for i in order:
        if t[i] < start:
            labels[i] = "warmup"
    n = len(eligible)
    n_train = int(round(n * fractions[0]))
    n_val = int(round(n * fractions[1]))
    for pos, i in enumerate(eligible):
        labels[i] = "train" if pos < n_train else ("val" if pos < n_train + n_val else "test")
    return labels.tolist()
