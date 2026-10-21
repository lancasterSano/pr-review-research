#!/usr/bin/env python3
"""Cache the SHAP values needed to redraw Figures 4 and 5.

Figure 4 (T1, review iterations) is unchanged in substance: it uses the same
code path as the published run (week4_modeling.py, CatBoost, seed 42), and is
only re-rendered here with larger fonts (reviewer 1, item 14.4.1).

Figure 5 (T2, staleness) is a *correction*, not a restyle. The published figure
came from the snapshot label, where `open_pr_count` -- for open PRs the
chronological rank of the PR, hence a proxy for age -- carried 39.4% of the
importance. That model was discarded. This script reproduces the code path of
taskD2_shap_new_t2.py exactly (same rows, same features, same CatBoost
configuration, same seed) and additionally stores the SHAP matrix so the figure
can be restyled without refitting.

Fitting is the expensive part, so everything is written to a single npz that
the drawing script reads. Nothing here overwrites a published artefact.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import FEATURE_COLS, PROCESSED_DIR, OUT_DIR  # noqa: E402

ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "predict"))

csv.field_size_limit(10 ** 9)

OUT_NPZ = os.path.join(OUT_DIR, "E-shap-values.npz")

# Published reference: the top-5 SHAP shares of the trace-based T2 model, as
# recorded in D-shap-new-t2.json by taskD2. Recomputing must land on the same
# numbers or the cache is not the model the article describes.
REF_T2_SHARES = {
    "author_commit_frequency": 17.86,
    "author_pr_count": 11.01,
    "num_commits": 9.58,
    "merge_velocity_7d": 9.36,
    "desc_length": 9.28,
}


def fail(msg):
    print("ERROR:", msg)
    sys.exit(1)


def tf(v):
    if v in ("", "None", "none", None):
        return np.nan
    if v in ("True", "true"):
        return 1.0
    if v in ("False", "false"):
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def make_model():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05, verbose=0,
        random_seed=42, auto_class_weights="Balanced", thread_count=4,
    )


def beeswarm_inputs(model, X, n_sample, seed=42):
    import shap
    idx = np.random.RandomState(seed).choice(len(X), min(n_sample, len(X)),
                                             replace=False)
    X_s = X[idx]
    sv = shap.TreeExplainer(model).shap_values(X_s)
    if isinstance(sv, list):
        sv = sv[1]
    return X_s, sv


# ---------------------------------------------------------------------------
# T1 -- published code path, identical to generate_model_figures.py
# ---------------------------------------------------------------------------
import week4_modeling as w4  # noqa: E402

if list(w4.FEATURE_COLS) != list(FEATURE_COLS):
    fail("FEATURE_COLS drifted between week4_modeling.py and revision/common.py")

merged_path = os.path.join(PROCESSED_DIR, "all_merged_features.csv")
X1, y1, _ = w4.load_data(merged_path, w4.FEATURE_COLS, w4.dp2_target)
print(f"T1: n={len(y1)} positives={int(y1.sum())} ({y1.mean():.4f})")
m1 = make_model()
m1.fit(X1, y1)
X1_s, sv1 = beeswarm_inputs(m1, X1, 2000)
print("T1 SHAP done", sv1.shape)

# ---------------------------------------------------------------------------
# T2 -- trace-based label (Definition 3), identical to taskD2_shap_new_t2.py
# ---------------------------------------------------------------------------
lab = {}
with open(os.path.join(OUT_DIR, "D-definition3-labels.csv"), newline="",
          encoding="utf-8") as f:
    for r in csv.DictReader(f):
        lab[(r["repo"], r["pr_number"])] = int(r["stale_delta_30"])

rows = [r for r in csv.DictReader(open(merged_path, newline="", encoding="utf-8"))
        if (r["repo"], str(r["pr_number"])) in lab]

X2 = np.array([[tf(r.get(c, "")) for c in FEATURE_COLS] for r in rows],
              dtype=np.float64)
for j in range(X2.shape[1]):
    col = X2[:, j]
    m = np.isnan(col)
    if m.any():
        col[m] = np.nanmedian(col)
y2 = np.array([lab[(r["repo"], str(r["pr_number"]))] for r in rows], dtype=np.int32)
print(f"T2: n={len(y2)} positives={int(y2.sum())} ({y2.mean():.4f})")

m2 = make_model()
m2.fit(X2, y2)
X2_s, sv2 = beeswarm_inputs(m2, X2, 4000)
print("T2 SHAP done", sv2.shape)

# Verify against the recorded shares before anything downstream uses this.
share = 100 * np.abs(sv2).mean(axis=0) / np.abs(sv2).mean(axis=0).sum()
bad = []
for feat, ref in REF_T2_SHARES.items():
    got = share[FEATURE_COLS.index(feat)]
    ok = abs(got - ref) < 0.05
    print(f"  [{'PASS' if ok else 'FAIL'}] {feat:26s} {got:6.2f} (ref {ref})")
    if not ok:
        bad.append(feat)
if bad:
    fail(f"T2 SHAP shares do not reproduce D-shap-new-t2.json: {bad}")

np.savez_compressed(
    OUT_NPZ,
    feature_cols=np.array(FEATURE_COLS),
    t1_X=X1_s, t1_shap=sv1,
    t2_X=X2_s, t2_shap=sv2,
    t2_n=np.array([len(y2)]), t2_pos=np.array([int(y2.sum())]),
)
print("wrote", OUT_NPZ)
