#!/usr/bin/env python3
"""Option C: rebuild the staleness target from the article's OWN Definition 3.

Why
---
The published T2 target is derived from a single collection snapshot: a PR counts
as stale if, at collection time, more than Delta days had passed since its last
activity. That operationalisation makes the label a near-function of PR age
(stale share by age: 0.000 below 30 days, 0.973 above one year), and PR age leaks
into the feature matrix through any feature that moves with calendar time. The
published `open_pr_count` column for open PRs is literally the within-repo
chronological rank, and PR age alone scores AUC-ROC 0.938 against the published
label -- higher than the published 24-feature model's 0.897.

Definition 3 as written in the article is not the snapshot rule. It says:

    stale(p) = 1 if there exists a time window of length Delta within the open
    lifetime of p containing no events.

That is computable exactly, from the event trace, for every MERGED pull request,
because a merged PR's lifetime is fully observed. No snapshot, no right-censoring,
no age proxy: a PR that lived three years with steady activity is not stale, and a
PR that lived 40 days with a 35-day silence is.

What this script does
---------------------
1. Rebuilds the event trace per merged PR from the raw GraphQL nodes.
2. Labels each PR by its longest intra-lifetime inactivity gap, for several Delta.
3. Joins the new label onto the PUBLISHED merged feature matrix, so the 24
   predictors and the row set stay bit-identical to Table 3's inputs.
4. Re-runs the Table 4 model comparison and the leave-one-repo-out protocol.
5. Runs leakage diagnostics: per-feature AUC against the new label, so that a
   repeat of the `open_pr_count` failure cannot pass unnoticed.

Nothing under enrich/ or predict/ is modified.

Known limitations, to be stated in the paper:
  - Restricted to merged PRs, so the label conditions on eventual success
    (survivorship). PRs abandoned forever are outside this population.
  - Commit timestamps are `authoredDate`, i.e. when the commit was authored, not
    when it was pushed. Rebases and cherry-picks can shift this backwards.
  - Continuous integration check events are not present in the collected data, so
    a PR whose only activity in a window was a CI run may be labelled stale.
  - GraphQL pagination truncates a small share of commit/thread node lists.
"""

import csv
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FEATURE_COLS, PROCESSED_DIR, RAW_DIR, OUT_DIR  # noqa: E402

csv.field_size_limit(10 ** 9)

MERGED_CSV = os.path.join(PROCESSED_DIR, "all_merged_features.csv")
DELTAS = [15, 30, 45, 60, 90]
PRIMARY_DELTA = 30
SIZE_ONLY = ["lines_added", "lines_deleted", "files_changed", "num_commits"]


def parse_dt(s):
    if not s or s in ("None", ""):
        return None
    try:
        return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def build_trace(pr):
    """Timestamped event sequence over the PR's open lifetime, per Definition 3."""
    opened = parse_dt(pr.get("createdAt"))
    merged = parse_dt(pr.get("mergedAt"))
    if not opened or not merged or merged < opened:
        return None
    events = [opened]
    truncated = False

    commits = pr.get("commits") or {}
    nodes = commits.get("nodes") or []
    if commits.get("totalCount", 0) > len(nodes):
        truncated = True
    for n in nodes:
        t = parse_dt(((n or {}).get("commit") or {}).get("authoredDate"))
        # commits predating the PR are ignored, as stated in the article
        if t and opened <= t <= merged:
            events.append(t)

    reviews = pr.get("reviews") or {}
    rnodes = reviews.get("nodes") or []
    if reviews.get("totalCount", 0) > len(rnodes):
        truncated = True
    for n in rnodes:
        t = parse_dt((n or {}).get("submittedAt"))
        if t and opened <= t <= merged:
            events.append(t)

    threads = pr.get("reviewThreads") or {}
    tnodes = threads.get("nodes") or []
    if threads.get("totalCount", 0) > len(tnodes):
        truncated = True
    for th in tnodes:
        cs = (th or {}).get("comments") or {}
        for c in cs.get("nodes") or []:
            t = parse_dt((c or {}).get("createdAt"))
            if t and opened <= t <= merged:
                events.append(t)

    events.append(merged)
    events.sort()
    return events, truncated


def max_gap_days(events):
    if len(events) < 2:
        return 0.0
    return max((events[i + 1] - events[i]).total_seconds() / 86400.0
               for i in range(len(events) - 1))


def repo_dir_to_slug(name):
    """django_django -> django/django ; vercel_next.js -> vercel/next.js"""
    owner, _, rest = name.partition("_")
    return f"{owner}/{rest}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- 1. label from the trace -------------------------------------------
    gaps = {}
    truncated_n = 0
    total_raw = 0
    for d in sorted(os.listdir(RAW_DIR)):
        path = os.path.join(RAW_DIR, d, "merged_prs.json")
        if not os.path.exists(path):
            continue
        slug = repo_dir_to_slug(d)
        with open(path, encoding="utf-8") as f:
            prs = json.load(f)
        for pr in prs:
            total_raw += 1
            built = build_trace(pr)
            if built is None:
                continue
            events, trunc = built
            truncated_n += bool(trunc)
            gaps[(slug, str(pr["number"]))] = (max_gap_days(events), len(events))
    print(f"traces built: {len(gaps)} of {total_raw} raw merged PRs "
          f"({truncated_n} with a truncated node list)")

    # ---- 2. join onto the published feature matrix --------------------------
    with open(MERGED_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"published merged feature matrix: {len(rows)} rows")

    matched = [r for r in rows if (r["repo"], str(r["pr_number"])) in gaps]
    print(f"matched to a trace: {len(matched)} "
          f"({100 * len(matched) / len(rows):.2f}%)")
    if len(matched) < 0.98 * len(rows):
        print("ERROR: join lost more than 2% of rows; aborting", file=sys.stderr)
        sys.exit(1)

    def to_float(v):
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

    X = np.array([[to_float(r.get(c, "")) for c in FEATURE_COLS] for r in matched],
                 dtype=np.float64)
    for j in range(X.shape[1]):
        col = X[:, j]
        m = np.isnan(col)
        if m.any():
            col[m] = np.nanmedian(col)
    repos = np.array([r["repo"] for r in matched])
    gap_days = np.array([gaps[(r["repo"], str(r["pr_number"]))][0] for r in matched])
    lifetime = np.array([to_float(r.get("time_to_merge_hours")) / 24.0 for r in matched])

    # ---- 3. base rates ------------------------------------------------------
    print("\n=== base rate of the trace-based label ===")
    print(f"{'Delta':>6s} {'stale n':>9s} {'share':>8s}")
    for delta in DELTAS:
        y = (gap_days > delta).astype(int)
        print(f"{delta:6d} {y.sum():9d} {y.mean():8.3f}")

    y = (gap_days > PRIMARY_DELTA).astype(int)
    print(f"\nper repository at Delta = {PRIMARY_DELTA}:")
    for rp in sorted(set(repos)):
        m = repos == rp
        print(f"  {rp:26s} n={m.sum():6d}  stale share={y[m].mean():.3f}")

    # ---- 4. leakage diagnostics --------------------------------------------
    from sklearn.metrics import roc_auc_score
    print("\n=== leakage check: AUC of each single submission-time feature ===")
    aucs = []
    for j, name in enumerate(FEATURE_COLS):
        col = X[:, j]
        if len(np.unique(col)) < 2:
            aucs.append((name, float("nan")))
            continue
        a = roc_auc_score(y, col)
        aucs.append((name, max(a, 1 - a)))
    for name, a in sorted(aucs, key=lambda t: -(t[1] if t[1] == t[1] else 0))[:8]:
        print(f"  {name:26s} {a:.3f}")
    print(f"  [reference] observed lifetime (NOT a feature): "
          f"{roc_auc_score(y, np.nan_to_num(lifetime)):.3f}")

    # ---- 5. models ----------------------------------------------------------
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, average_precision_score)
    from catboost import CatBoostClassifier
    import lightgbm as lgb

    def make(name):
        if name == "Dummy (majority)":
            return DummyClassifier(strategy="most_frequent")
        if name == "Logistic regression":
            return LogisticRegression(max_iter=1000, random_state=42)
        if name == "CatBoost (size-only)":
            return CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                                      verbose=0, random_seed=42,
                                      auto_class_weights="Balanced", thread_count=4)
        if name == "CatBoost (all features)":
            return CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                                      verbose=0, random_seed=42,
                                      auto_class_weights="Balanced", thread_count=4)
        if name == "LightGBM (all features)":
            return lgb.LGBMClassifier(n_estimators=500, max_depth=6,
                                      learning_rate=0.05, is_unbalance=True,
                                      verbose=-1, random_state=42, n_jobs=4)
        raise KeyError(name)

    MODELS = ["Dummy (majority)", "Logistic regression", "CatBoost (size-only)",
              "CatBoost (all features)", "LightGBM (all features)"]
    size_idx = [FEATURE_COLS.index(c) for c in SIZE_ONLY]

    def ev(yt, yp, pr):
        out = {"accuracy": accuracy_score(yt, yp),
               "f1": f1_score(yt, yp, zero_division=0),
               "precision": precision_score(yt, yp, zero_division=0),
               "recall": recall_score(yt, yp, zero_division=0)}
        if pr is not None and len(np.unique(yt)) > 1:
            out["auc_roc"] = roc_auc_score(yt, pr)
            out["auc_pr"] = average_precision_score(yt, pr)
        return out

    print(f"\n=== 5-fold stratified CV, Delta = {PRIMARY_DELTA} "
          f"(new Table 4) ===")
    print(f"{'Model':26s} {'Acc':>7s} {'F1':>7s} {'Prec':>7s} {'Rec':>7s} "
          f"{'AUC-ROC':>8s} {'AUC-PR':>7s}")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows, per_fold = {}, {}
    for name in MODELS:
        fm = []
        for tr, te in skf.split(X, y):
            Xtr, Xte = (X[tr][:, size_idx], X[te][:, size_idx]) \
                if "size-only" in name else (X[tr], X[te])
            mdl = make(name)
            if name == "Logistic regression":
                sc = StandardScaler()
                Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)
            mdl.fit(Xtr, y[tr])
            p = mdl.predict_proba(Xte)[:, 1] if hasattr(mdl, "predict_proba") else None
            fm.append(ev(y[te], mdl.predict(Xte), p))
        avg = {k: float(np.mean([m[k] for m in fm])) for k in fm[0]}
        sd = {k: float(np.std([m[k] for m in fm], ddof=1)) for k in fm[0]}
        cv_rows[name], per_fold[name] = (avg, sd), fm
        print(f"{name:26s} {avg['accuracy']:7.3f} {avg['f1']:7.3f} "
              f"{avg['precision']:7.3f} {avg['recall']:7.3f} "
              f"{avg.get('auc_roc', float('nan')):8.3f} "
              f"{avg.get('auc_pr', float('nan')):7.3f}")
    print("\n  mean +/- std over the 5 folds:")
    for name in MODELS:
        avg, sd = cv_rows[name]
        print(f"    {name:26s} F1 = {avg['f1']:.3f} +/- {sd['f1']:.3f}   "
              f"AUC-ROC = {avg.get('auc_roc', float('nan')):.3f} +/- "
              f"{sd.get('auc_roc', float('nan')):.3f}")

    print(f"\n=== leave-one-repo-out, Delta = {PRIMARY_DELTA} ===")
    loro = {}
    for name in ["CatBoost (all features)", "LightGBM (all features)"]:
        per_repo = {}
        for rp in sorted(set(repos)):
            te = repos == rp
            tr = ~te
            if te.sum() < 10 or len(np.unique(y[te])) < 2:
                continue
            mdl = make(name)
            mdl.fit(X[tr], y[tr])
            per_repo[rp] = ev(y[te], mdl.predict(X[te]), mdl.predict_proba(X[te])[:, 1])
        loro[name] = per_repo
        f1s = [m["f1"] for m in per_repo.values()]
        au = [m["auc_roc"] for m in per_repo.values()]
        print(f"{name:26s} F1 mean={np.mean(f1s):.3f} median={np.median(f1s):.3f} "
              f"range={min(f1s):.3f}-{max(f1s):.3f} | "
              f"AUC-ROC mean={np.mean(au):.3f} median={np.median(au):.3f} "
              f"range={min(au):.3f}-{max(au):.3f}")
        for rp, m in sorted(per_repo.items()):
            print(f"    {rp:26s} F1={m['f1']:.3f} AUC-ROC={m['auc_roc']:.3f} "
                  f"P={m['precision']:.3f} R={m['recall']:.3f}")

    # ---- 6. Delta sweep on the new label ------------------------------------
    print("\n=== Delta sweep on the trace-based label (CatBoost) ===")
    print(f"{'Delta':>6s} {'share':>7s} {'F1':>7s} {'AUC-ROC':>8s} {'LORO AUC':>9s}")
    sweep = []
    for delta in DELTAS:
        yy = (gap_days > delta).astype(int)
        if len(np.unique(yy)) < 2:
            continue
        fm = []
        for tr, te in skf.split(X, yy):
            mdl = make("CatBoost (all features)")
            mdl.fit(X[tr], yy[tr])
            fm.append(ev(yy[te], mdl.predict(X[te]), mdl.predict_proba(X[te])[:, 1]))
        cv = {k: float(np.mean([m[k] for m in fm])) for k in fm[0]}
        lo = []
        for rp in sorted(set(repos)):
            te = repos == rp
            if te.sum() < 10 or len(np.unique(yy[te])) < 2:
                continue
            mdl = make("CatBoost (all features)")
            mdl.fit(X[~te], yy[~te])
            lo.append(roc_auc_score(yy[te], mdl.predict_proba(X[te])[:, 1]))
        sweep.append((delta, yy.mean(), cv["f1"], cv["auc_roc"], float(np.mean(lo))))
        print(f"{delta:6d} {yy.mean():7.3f} {cv['f1']:7.3f} {cv['auc_roc']:8.3f} "
              f"{np.mean(lo):9.3f}")

    # ---- 7. persist ---------------------------------------------------------
    with open(os.path.join(OUT_DIR, "D-definition3-labels.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["repo", "pr_number", "max_gap_days", "n_events",
                    "lifetime_days"] + [f"stale_delta_{d}" for d in DELTAS])
        for r, g, lt in zip(matched, gap_days, lifetime):
            n_ev = gaps[(r["repo"], str(r["pr_number"]))][1]
            w.writerow([r["repo"], r["pr_number"], f"{g:.4f}", n_ev,
                        f"{lt:.4f}" if lt == lt else ""] +
                       [int(g > d) for d in DELTAS])

    with open(os.path.join(OUT_DIR, "D-definition3-metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "primary_delta": PRIMARY_DELTA,
            "n": int(len(matched)),
            "base_rate": {str(d): float((gap_days > d).mean()) for d in DELTAS},
            "cv": {k: {"mean": v[0], "std": v[1]} for k, v in cv_rows.items()},
            "loro": {k: {rp: m for rp, m in v.items()} for k, v in loro.items()},
            "delta_sweep": sweep,
            "feature_auc": {n: (a if a == a else None) for n, a in aucs},
        }, f, indent=2)
    print(f"\nwritten to {OUT_DIR}/D-definition3-labels.csv and "
          f"D-definition3-metrics.json")


if __name__ == "__main__":
    main()
