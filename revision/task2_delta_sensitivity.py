#!/usr/bin/env python3
"""Task 2 — sensitivity of the T2 (staleness) task to the threshold Delta.

For Delta in {15, 30, 45, 60, 90} the label is recomputed from the raw JSON at the
pinned per-repository observation date T0 (see task1_pin_t0.py) and the full T2
protocol from `predict/week4_modeling.py` is re-run:

  * 5-fold stratified CV (shuffle=True, random_state=42) -> accuracy, F1,
    precision, recall, AUC-ROC, AUC-PR
  * leave-one-repo-out CV -> mean F1, mean AUC-ROC

Everything else is held fixed: the same 24 features, the same median imputation,
the same CatBoost / LightGBM hyperparameters, the same random_state=42, the same
LORO admissibility rules (>=10 train and test rows, both classes present in test).
Only `thread_count` / `n_jobs` are pinned to 4 because other jobs share the box;
these affect wall-clock, not the learning algorithm's configuration.

A reproduction row (Delta=30 evaluated against the *published* datetime.now()
label) is also computed so the sweep can be anchored to the published Table 4.
"""

import csv
import json
import os
import time
import warnings

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)
from catboost import CatBoostClassifier
import lightgbm as lgb

import common as C

warnings.filterwarnings("ignore")

METRIC_KEYS = ["accuracy", "f1", "precision", "recall", "auc_roc", "auc_pr"]


def make_models():
    """Identical to week4_modeling.models_config, minus the baselines we do not sweep."""
    return {
        "CatBoost": lambda: CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            verbose=0, random_seed=42, auto_class_weights="Balanced",
            thread_count=C.THREADS, allow_writing_files=False,
        ),
        "LightGBM": lambda: lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            is_unbalance=True, verbose=-1, random_state=42, n_jobs=C.THREADS,
        ),
    }


def evaluate(y_true, y_pred, y_prob):
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        m["auc_roc"] = roc_auc_score(y_true, y_prob)
        m["auc_pr"] = average_precision_score(y_true, y_prob)
    return m


def cv5(X, y, model_fn):
    """5-fold stratified CV; returns (mean dict, std dict, per-fold list)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    for train_idx, test_idx in skf.split(X, y):
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        y_prob = model.predict_proba(X[test_idx])[:, 1]
        folds.append(evaluate(y[test_idx], y_pred, y_prob))
    mean = {k: float(np.mean([f[k] for f in folds])) for k in folds[0]}
    std = {k: float(np.std([f[k] for f in folds], ddof=1)) for k in folds[0]}
    return mean, std, folds


def loro(X, y, repos_arr, model_fn):
    """Leave-one-repo-out CV; same admissibility rules as week4_modeling."""
    per_repo = {}
    for repo in sorted(set(repos_arr.tolist())):
        test_mask = repos_arr == repo
        train_mask = ~test_mask
        if test_mask.sum() < 10 or train_mask.sum() < 10:
            continue
        if len(np.unique(y[test_mask])) < 2:
            continue
        model = model_fn()
        model.fit(X[train_mask], y[train_mask])
        y_pred = model.predict(X[test_mask])
        y_prob = model.predict_proba(X[test_mask])[:, 1]
        per_repo[repo] = evaluate(y[test_mask], y_pred, y_prob)
    if not per_repo:
        return None, {}
    keys = list(next(iter(per_repo.values())))
    mean = {k: float(np.mean([m[k] for m in per_repo.values()])) for k in keys}
    return mean, per_repo


def run_setting(name, X, y, repos_arr, results, extra=None):
    n = len(y)
    pos = int(y.sum())
    print(f"\n{'=' * 92}")
    print(f"  {name}   n={n}  stale={pos} ({pos / n:.3f})")
    print(f"{'=' * 92}")
    if len(np.unique(y)) < 2:
        print("  SKIPPED: only one class present — no model can be fitted.")
        return
    for model_name, model_fn in make_models().items():
        t = time.time()
        mean, std, _ = cv5(X, y, model_fn)
        lm, lper = loro(X, y, repos_arr, model_fn)
        row = {
            "setting": name,
            "model": model_name,
            "n": n,
            "n_stale": pos,
            "stale_share": round(pos / n, 3),
        }
        for k in METRIC_KEYS:
            row[f"cv5_{k}"] = round(mean.get(k, float("nan")), 3)
            row[f"cv5_{k}_std"] = round(std.get(k, float("nan")), 3)
        row["loro_f1"] = round(lm["f1"], 3) if lm else float("nan")
        row["loro_auc_roc"] = round(lm.get("auc_roc", float("nan")), 3) if lm else float("nan")
        row["loro_n_repos"] = len(lper)
        if extra:
            row.update(extra)
        results.append(row)
        print(f"  {model_name:10s} CV5  acc={row['cv5_accuracy']:.3f} f1={row['cv5_f1']:.3f} "
              f"prec={row['cv5_precision']:.3f} rec={row['cv5_recall']:.3f} "
              f"auc_roc={row['cv5_auc_roc']:.3f} auc_pr={row['cv5_auc_pr']:.3f}  ||  "
              f"LORO f1={row['loro_f1']:.3f} auc_roc={row['loro_auc_roc']:.3f} "
              f"({len(lper)} repos)   [{time.time() - t:.0f}s]")
        results[-1]["_loro_per_repo"] = {r: {k: round(v, 3) for k, v in m.items()}
                                        for r, m in lper.items()}


def main():
    os.makedirs(C.OUT_DIR, exist_ok=True)

    rows = C.load_open_rows()
    records, t0_by_repo, _ = C.load_raw_open()
    X = C.build_matrix(rows)
    repos_arr = np.array([r["repo"] for r in rows])
    print(f"Feature matrix: {X.shape}  (24 features, median-imputed, identical across Deltas)")

    results = []

    # ── anchor: reproduce published Table 4 (label = published datetime.now()) ──
    y_pub = C.published_label(rows)
    run_setting("PUBLISHED (now()-based, Delta=30)", X, y_pub, repos_arr, results,
                extra={"delta": 30, "label_source": "published_now"})

    # ── the sweep, all at pinned T0 ─────────────────────────────────────────
    per_repo_shares = {}
    for delta in C.DELTAS:
        y, silent, age = C.relabel(rows, records, t0_by_repo, delta)
        shares = {}
        for repo in sorted(set(repos_arr.tolist())):
            m = repos_arr == repo
            shares[repo] = {
                "n": int(m.sum()),
                "n_stale": int(y[m].sum()),
                "share": round(float(y[m].mean()), 3),
            }
        per_repo_shares[delta] = shares
        run_setting(f"T0-pinned, Delta={delta}", X, y, repos_arr, results,
                    extra={"delta": delta, "label_source": "pinned_T0"})

    # ── per-repository stale shares table ───────────────────────────────────
    print(f"\n{'=' * 92}")
    print("  Per-repository stale share by Delta (pinned T0)")
    print(f"{'=' * 92}")
    hdr = f"{'repository':26s} {'n':>5s}" + "".join(f"{'D=' + str(d):>16s}" for d in C.DELTAS)
    print(hdr)
    for repo in sorted(set(repos_arr.tolist())):
        line = f"{repo:26s} {per_repo_shares[C.DELTAS[0]][repo]['n']:5d}"
        for d in C.DELTAS:
            s = per_repo_shares[d][repo]
            line += f"{s['n_stale']:9d} ({s['share']:.2f})"
        print(line)
    line = f"{'POOLED':26s} {len(rows):5d}"
    for d in C.DELTAS:
        tot = sum(per_repo_shares[d][r]["n_stale"] for r in per_repo_shares[d])
        line += f"{tot:9d} ({tot / len(rows):.2f})"
    print(line)

    # ── persist ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(C.OUT_DIR, "B-delta-sensitivity.csv")
    cols = ["setting", "delta", "label_source", "model", "n", "n_stale", "stale_share"] + \
           [f"cv5_{k}" for k in METRIC_KEYS] + [f"cv5_{k}_std" for k in METRIC_KEYS] + \
           ["loro_f1", "loro_auc_roc", "loro_n_repos"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved {csv_path}")

    json_path = os.path.join(C.OUT_DIR, "B-delta-sensitivity.json")
    with open(json_path, "w") as f:
        json.dump({"results": results, "per_repo_shares": per_repo_shares}, f, indent=2)
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
