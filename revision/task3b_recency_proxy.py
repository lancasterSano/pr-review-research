#!/usr/bin/env python3
"""Task 3b — diagnostic: how much of T2 performance rides on the recency proxy?

Finding that motivates this script
----------------------------------
For the *open*-PR population, `compute_features.compute_project_context` counts,
for each PR, the prior PRs in the same list that are still OPEN. Since the open
list contains nothing but OPEN PRs, that count degenerates to the PR's 0-based
chronological rank inside its repository. Verified: 7156/7156 rows.

Consequence: `open_pr_count` — the top SHAP feature of the published T2 model —
is a deterministic within-repository recency index (Spearman rho = -1.000 with
PR age in every one of the 12 repositories). Because a PR younger than Delta
*cannot* be labelled stale, a recency index is a partial oracle for the label.

This script re-runs the Delta=30 pinned-T0 T2 model with `open_pr_count` removed
(23 features), on the full sample and on the uncensored sample, so the size of
that effect is a number rather than a suspicion. It changes no published artefact.
"""

import csv
import os

import numpy as np

import common as C
from task2_delta_sensitivity import cv5, loro, make_models, METRIC_KEYS

FEATURES_NO_OPC = [c for c in C.FEATURE_COLS if c != "open_pr_count"]


def matrix_for(rows, cols):
    X = np.array([[C.to_float(r.get(c, "")) for c in cols] for r in rows], dtype=np.float64)
    for j in range(X.shape[1]):
        col = X[:, j]
        m = np.isnan(col)
        if m.any():
            col[m] = np.nanmedian(col)
    return X


def run(name, X, y, repos_arr, results):
    print(f"\n--- {name}: n={len(y)}, stale={int(y.sum())} ({y.mean():.3f}), "
          f"features={X.shape[1]} ---")
    for model_name, fn in make_models().items():
        mean, std, _ = cv5(X, y, fn)
        lm, lper = loro(X, y, repos_arr, fn)
        row = {"setting": name, "model": model_name, "n_features": X.shape[1],
               "n": len(y), "n_stale": int(y.sum()), "stale_share": round(float(y.mean()), 3)}
        for k in METRIC_KEYS:
            row[f"cv5_{k}"] = round(mean.get(k, float("nan")), 3)
        row["loro_f1"] = round(lm["f1"], 3) if lm else ""
        row["loro_auc_roc"] = round(lm.get("auc_roc", float("nan")), 3) if lm else ""
        results.append(row)
        print(f"    {model_name:10s} CV5 f1={row['cv5_f1']:.3f} auc_roc={row['cv5_auc_roc']:.3f} "
              f"auc_pr={row['cv5_auc_pr']:.3f} | LORO f1={row['loro_f1']} auc={row['loro_auc_roc']}")


def main():
    rows = C.load_open_rows()
    records, t0, _ = C.load_raw_open()
    repos_arr = np.array([r["repo"] for r in rows])
    y, silent, age = C.relabel(rows, records, t0, 30)

    X24 = matrix_for(rows, C.FEATURE_COLS)
    X23 = matrix_for(rows, FEATURES_NO_OPC)

    # verify the rank claim
    exact = 0
    for repo in sorted(set(repos_arr.tolist())):
        idx = [i for i in range(len(rows)) if rows[i]["repo"] == repo]
        idx.sort(key=lambda i: rows[i]["created_at"] or "")
        exact += sum(1 for k, i in enumerate(idx) if int(rows[i]["open_pr_count"]) == k)
    print(f"open_pr_count == within-repo chronological rank for {exact}/{len(rows)} open PRs")

    results = []
    unc = age > 30
    run("Delta=30 pinned, all rows, 24 feat (baseline)", X24, y, repos_arr, results)
    run("Delta=30 pinned, all rows, 23 feat (no open_pr_count)", X23, y, repos_arr, results)
    run("Delta=30 pinned, age>30 only, 24 feat", X24[unc], y[unc], repos_arr[unc], results)
    run("Delta=30 pinned, age>30 only, 23 feat", X23[unc], y[unc], repos_arr[unc], results)

    out = os.path.join(C.OUT_DIR, "B-recency-proxy.csv")
    cols = ["setting", "model", "n_features", "n", "n_stale", "stale_share"] + \
           [f"cv5_{k}" for k in METRIC_KEYS] + ["loro_f1", "loro_auc_roc"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
