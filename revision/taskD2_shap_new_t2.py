#!/usr/bin/env python3
"""SHAP importances for the trace-based T2 model (Option C), to replace the
published staleness interpretation, which attributed 39.4% of importance to
`open_pr_count` -- a column that, for open PRs, was the chronological rank."""
import csv, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FEATURE_COLS, PROCESSED_DIR, OUT_DIR
csv.field_size_limit(10 ** 9)

lab = {}
with open(os.path.join(OUT_DIR, "D-definition3-labels.csv"), newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        lab[(r["repo"], r["pr_number"])] = int(r["stale_delta_30"])

rows = [r for r in csv.DictReader(open(os.path.join(PROCESSED_DIR, "all_merged_features.csv"),
                                        newline="", encoding="utf-8"))
        if (r["repo"], str(r["pr_number"])) in lab]

def tf(v):
    if v in ("", "None", "none", None): return np.nan
    if v in ("True", "true"): return 1.0
    if v in ("False", "false"): return 0.0
    try: return float(v)
    except (TypeError, ValueError): return np.nan

X = np.array([[tf(r.get(c, "")) for c in FEATURE_COLS] for r in rows], dtype=np.float64)
for j in range(X.shape[1]):
    col = X[:, j]; m = np.isnan(col)
    if m.any(): col[m] = np.nanmedian(col)
y = np.array([lab[(r["repo"], str(r["pr_number"]))] for r in rows], dtype=np.int32)
print("n =", len(y), "positives =", y.sum(), f"({y.mean():.4f})")

from catboost import CatBoostClassifier
import shap
model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, verbose=0,
                           random_seed=42, auto_class_weights="Balanced", thread_count=4)
model.fit(X, y)

idx = np.random.RandomState(42).choice(len(X), min(4000, len(X)), replace=False)
sv = shap.TreeExplainer(model).shap_values(X[idx])
if isinstance(sv, list): sv = sv[1]
imp = np.abs(sv).mean(axis=0)
share = 100 * imp / imp.sum()
order = np.argsort(-share)
print("\nSHAP importance share (%), trace-based T2:")
out = []
for k in order:
    print(f"  {FEATURE_COLS[k]:26s} {share[k]:6.2f}")
    out.append({"feature": FEATURE_COLS[k], "shap_share_pct": round(float(share[k]), 2)})

cb = 100 * model.feature_importances_ / model.feature_importances_.sum()
print("\nCatBoost internal importance (%), top 8:")
for k in np.argsort(-cb)[:8]:
    print(f"  {FEATURE_COLS[k]:26s} {cb[k]:6.2f}")

json.dump({"shap": out,
           "catboost": [{"feature": FEATURE_COLS[k], "pct": round(float(cb[k]), 2)}
                        for k in np.argsort(-cb)]},
          open(os.path.join(OUT_DIR, "D-shap-new-t2.json"), "w"), indent=2)
print("\nwritten D-shap-new-t2.json")
