#!/usr/bin/env python3
"""Task C3 — repository AI-adoption rate as a 25th submission-time feature.

WHAT IS AND IS NOT BUILT HERE
-----------------------------
NOT built: "this PR received an AI review". That event happens AFTER submission,
so using it as a predictor would be temporal leakage — exactly what the article's
expanding-window protocol exists to prevent.

Built: `repo_ai_adoption_rate` — for each PR p in repository R, the share of R's
PRs created STRICTLY EARLIER than p that received an AI review:

    repo_ai_adoption_rate(p) = |{q in R : created(q) < created(p), ai_review(q)}|
                               ----------------------------------------------
                               |{q in R : created(q) < created(p)}|

This is known at submission time and follows the same expanding-window rule as
the author-history features in compute_features.py. Two deliberate choices:

  * strict `<` on the timestamp, so PRs sharing a timestamp never see each other
    (compute_features.py uses positional expansion, which is order-dependent on
    ties; strict comparison is the conservative variant);
  * the window runs over the SAME population as the target CSV (merged history
    for the merged dataset, open history for the open dataset), which is how
    compute_features.py computes author history — it processes merged and open
    separately. A pooled-history variant is reported as a sensitivity check on
    the feature itself (not re-modelled).
  * with no prior PRs the rate is undefined; it is set to 0.0, matching the
    pipeline's convention of 0 for `author_pr_count` / `author_commit_frequency`
    on an author's first PR. `repo_prior_pr_count` is emitted alongside so the
    convention is visible.

Then: 5-fold stratified CV and LORO for T1 (review_rounds>=2 on merged) and
T2 (is_stale on open), with 24 vs 25 features, identical hyperparameters and
random_state=42 to week4_modeling.py, plus SHAP rank of the new feature and the
feature's confounding with repository identity.

Usage:
    python taskC3_ai_adoption_feature.py            # everything
    python taskC3_ai_adoption_feature.py --feature-only   # skip model fitting
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taskC_common import OUT_DIR, MERGED_CSV, OPEN_CSV, THREADS, load_processed  # noqa: E402

warnings.filterwarnings("ignore")

FLAGS_CSV = os.path.join(OUT_DIR, "C-ai-pr-flags.csv")

# Exactly week4_modeling.FEATURE_COLS, same order.
FEATURE_COLS_24 = [
    "lines_added", "lines_deleted", "files_changed", "num_commits",
    "has_tests", "has_docs", "touches_ci",
    "desc_length", "title_length", "has_screenshots", "has_code_blocks",
    "references_issues", "is_draft", "label_count",
    "day_of_week", "hour_of_day",
    "author_merge_rate", "author_avg_review_rounds",
    "author_commit_frequency", "author_tenure_days", "author_pr_count",
    "open_pr_count", "merge_velocity_7d", "merge_velocity_30d",
]
NEW_FEATURE = "repo_ai_adoption_rate"
FEATURE_COLS_25 = FEATURE_COLS_24 + [NEW_FEATURE]


def to_float(val):
    """Identical coercion to week4_modeling.to_float."""
    if val in ("", "None", "none", None):
        return np.nan
    if val in ("True", "true"):
        return 1.0
    if val in ("False", "false"):
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


# ── the feature ─────────────────────────────────────────────────────────────

def load_ai_flags():
    """{(repo, pr_number, pr_type): has_ai_review}"""
    out = {}
    for r in load_processed(FLAGS_CSV):
        out[(r["repo"], int(r["pr_number"]), r["pr_type"])] = int(r["has_ai_review"])
    return out


def compute_adoption(rows, pr_type, ai_flags):
    """Attach repo_ai_adoption_rate / repo_prior_pr_count to `rows` (in place).

    `rows` are the dict rows of a processed feature CSV, in file order; the
    computation is done per repository on a chronological copy, then written
    back onto the original row objects, so file order is preserved.
    """
    by_repo = defaultdict(list)
    for r in rows:
        by_repo[r["repo"]].append(r)

    missing = 0
    for repo, rs in by_repo.items():
        rs_sorted = sorted(rs, key=lambda r: r["created_at"] or "")
        n = len(rs_sorted)
        ai = []
        for r in rs_sorted:
            key = (r["repo"], int(r["pr_number"]), pr_type)
            if key not in ai_flags:
                missing += 1
                ai.append(0)
            else:
                ai.append(ai_flags[key])
        # prefix sums over the chronological order
        pref = [0] * (n + 1)
        for i in range(n):
            pref[i + 1] = pref[i] + ai[i]
        # For each i, history = all j with created_at[j] < created_at[i].
        # With the list sorted, that is the block [0, lo) where lo is the first
        # index whose timestamp equals created_at[i].
        j = 0
        while j < n:
            k = j
            ts = rs_sorted[j]["created_at"] or ""
            while k < n and (rs_sorted[k]["created_at"] or "") == ts:
                k += 1
            prior_n = j
            prior_ai = pref[j]
            for idx in range(j, k):
                rs_sorted[idx][NEW_FEATURE] = (prior_ai / prior_n) if prior_n else 0.0
                rs_sorted[idx]["repo_prior_pr_count"] = prior_n
                rs_sorted[idx]["_ai_actual"] = ai[idx]
            j = k
    return missing


def pooled_variant(merged_rows, open_rows, ai_flags):
    """Sensitivity check: adoption rate over merged+open history combined.

    Returns {(repo, pr_number, pr_type): rate}. Not used for modelling; only
    to show how much the feature depends on the population choice.
    """
    recs = []
    for rs, t in ((merged_rows, "merged"), (open_rows, "open")):
        for r in rs:
            recs.append((r["repo"], int(r["pr_number"]), t, r["created_at"] or "",
                         ai_flags.get((r["repo"], int(r["pr_number"]), t), 0)))
    out = {}
    by_repo = defaultdict(list)
    for rec in recs:
        by_repo[rec[0]].append(rec)
    for repo, rs in by_repo.items():
        rs.sort(key=lambda x: x[3])
        n = len(rs)
        pref = [0] * (n + 1)
        for i, rec in enumerate(rs):
            pref[i + 1] = pref[i] + rec[4]
        j = 0
        while j < n:
            k = j
            ts = rs[j][3]
            while k < n and rs[k][3] == ts:
                k += 1
            for idx in range(j, k):
                out[(rs[idx][0], rs[idx][1], rs[idx][2])] = (pref[j] / j) if j else 0.0
            j = k
    return out


# ── confound diagnostics ────────────────────────────────────────────────────

def eta_squared(values, groups):
    """Correlation ratio eta^2 of a numeric variable against a categorical one.

    eta^2 = between-group sum of squares / total sum of squares, i.e. the share
    of the feature's variance explained by group membership. 1.0 means the
    feature is a relabelling of the group.
    """
    v = np.asarray(values, dtype=np.float64)
    grand = v.mean()
    ss_tot = ((v - grand) ** 2).sum()
    if ss_tot == 0:
        return float("nan")
    ss_between = 0.0
    g = np.asarray(groups)
    for gr in np.unique(g):
        vals = v[g == gr]
        ss_between += len(vals) * (vals.mean() - grand) ** 2
    return ss_between / ss_tot


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


# ── modelling (mirrors week4_modeling.py) ───────────────────────────────────

def build_xy(rows, feature_cols, target_fn):
    X_list, y_list, repos = [], [], []
    for row in rows:
        t = target_fn(row)
        if t is None:
            continue
        X_list.append([to_float(row.get(c, "")) if not isinstance(row.get(c), (int, float))
                       else float(row.get(c)) for c in feature_cols])
        y_list.append(t)
        repos.append(row.get("repo", ""))
    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int32)
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col)
        if mask.any():
            col[mask] = np.nanmedian(col)
    return X, y, repos


def dp2_target(row):
    v = to_float(row.get("review_rounds", ""))
    if np.isnan(v):
        return None
    return 1 if v >= 2 else 0


def dp4_target(row):
    v = row.get("is_stale", "")
    if v in ("True", "true", "1"):
        return 1
    if v in ("False", "false", "0"):
        return 0
    return None


def make_models():
    from catboost import CatBoostClassifier
    import lightgbm as lgb
    return {
        "CatBoost": lambda: CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            verbose=0, random_seed=42, auto_class_weights="Balanced",
            thread_count=THREADS,
        ),
        "LightGBM": lambda: lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            is_unbalance=True, verbose=-1, random_state=42, n_jobs=THREADS,
        ),
    }


def evaluate(y_true, y_pred, y_prob):
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score, average_precision_score)
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


def run_protocol(X, y, repos, tag):
    """5-fold stratified CV (per-fold metrics kept) + LORO (per-repo metrics kept)."""
    from sklearn.model_selection import StratifiedKFold
    res = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = make_models()
    repos_arr = np.array(repos)

    for name, fn in models.items():
        folds = []
        for tr, te in skf.split(X, y):
            m = fn()
            m.fit(X[tr], y[tr])
            yp = m.predict(X[te])
            pp = m.predict_proba(X[te])[:, 1]
            folds.append(evaluate(y[te], yp, pp))
        res[f"cv5_{name}"] = {
            "mean": {k: float(np.mean([f[k] for f in folds])) for k in folds[0]},
            "std": {k: float(np.std([f[k] for f in folds], ddof=1)) for k in folds[0]},
            "folds": folds,
        }
        print(f"    [{tag}] CV5 {name:9s} " +
              " ".join(f"{k}={np.mean([f[k] for f in folds]):.3f}" for k in folds[0]))

    for name, fn in models.items():
        per_repo = {}
        for repo in sorted(set(repos)):
            te = repos_arr == repo
            tr = ~te
            if te.sum() < 10 or tr.sum() < 10 or len(np.unique(y[te])) < 2:
                continue
            m = fn()
            m.fit(X[tr], y[tr])
            yp = m.predict(X[te])
            pp = m.predict_proba(X[te])[:, 1]
            per_repo[repo] = evaluate(y[te], yp, pp)
        keys = list(next(iter(per_repo.values())).keys())
        res[f"loro_{name}"] = {
            "mean": {k: float(np.mean([v[k] for v in per_repo.values()])) for k in keys},
            "median": {k: float(np.median([v[k] for v in per_repo.values()])) for k in keys},
            "per_repo": per_repo,
        }
        print(f"    [{tag}] LORO {name:9s} " +
              " ".join(f"{k}={np.mean([v[k] for v in per_repo.values()]):.3f}" for k in keys))
    return res


def shap_rank(X, y, feature_cols, tag):
    """Global SHAP importance of the final CatBoost, same recipe as week4_modeling."""
    import shap
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                           verbose=0, random_seed=42, auto_class_weights="Balanced",
                           thread_count=THREADS)
    m.fit(X, y)
    n = min(2000, len(X))
    idx = np.random.RandomState(42).choice(len(X), n, replace=False)
    sv = shap.TreeExplainer(m).shap_values(X[idx])
    if isinstance(sv, list):
        sv = sv[1]
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(-imp)
    ranked = [(feature_cols[i], float(imp[i])) for i in order]
    cb = dict(zip(feature_cols, [float(v) for v in m.feature_importances_]))
    print(f"    [{tag}] SHAP top 8: " + ", ".join(f"{n_}={v:.4f}" for n_, v in ranked[:8]))
    return ranked, cb


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    ai_flags = load_ai_flags()

    merged_rows = load_processed(MERGED_CSV)
    open_rows = load_processed(OPEN_CSV)

    miss_m = compute_adoption(merged_rows, "merged", ai_flags)
    miss_o = compute_adoption(open_rows, "open", ai_flags)
    print(f"rows without an AI flag (treated as no-AI): merged={miss_m} open={miss_o}")

    pooled = pooled_variant(merged_rows, open_rows, ai_flags)

    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ── feature description + confound diagnostics ──────────────────────────
    for tag, rows, pr_type in (("T1_merged", merged_rows, "merged"),
                               ("T2_open", open_rows, "open")):
        vals = np.array([r[NEW_FEATURE] for r in rows], dtype=np.float64)
        reps = [r["repo"] for r in rows]
        actual = np.array([r["_ai_actual"] for r in rows], dtype=np.float64)
        pooled_vals = np.array([pooled[(r["repo"], int(r["pr_number"]), pr_type)] for r in rows])

        d = {
            "n": len(vals),
            "mean": float(vals.mean()), "median": float(np.median(vals)),
            "min": float(vals.min()), "max": float(vals.max()),
            "share_zero": float((vals == 0).mean()),
            "eta2_vs_repo": float(eta_squared(vals, reps)),
            "spearman_vs_actual_ai_review": spearman(vals, actual),
            "spearman_vs_pooled_history_variant": spearman(vals, pooled_vals),
            "max_abs_diff_vs_pooled_variant": float(np.abs(vals - pooled_vals).max()),
            "per_repo_mean": {},
            "per_repo_max": {},
        }
        for repo in sorted(set(reps)):
            m = np.array([v for v, rp in zip(vals, reps) if rp == repo])
            d["per_repo_mean"][repo] = float(m.mean())
            d["per_repo_max"][repo] = float(m.max())
        out[f"feature_{tag}"] = d

        print(f"\n=== {NEW_FEATURE} — {tag} ===")
        print(f"  n={d['n']}  mean={d['mean']:.3f}  median={d['median']:.3f}  "
              f"max={d['max']:.3f}  share_exactly_zero={d['share_zero']:.3f}")
        print(f"  eta^2 against repository identity: {d['eta2_vs_repo']:.3f}   "
              f"(1.000 = feature is a relabelling of the repo)")
        print(f"  Spearman vs the (leaky) actual AI-review flag: "
              f"{d['spearman_vs_actual_ai_review']:.3f}")
        print(f"  Spearman vs pooled-history variant: "
              f"{d['spearman_vs_pooled_history_variant']:.3f} "
              f"(max abs diff {d['max_abs_diff_vs_pooled_variant']:.3f})")
        print("  per-repo mean:")
        for repo, v in sorted(d["per_repo_mean"].items(), key=lambda x: -x[1]):
            print(f"    {repo:26s} mean={v:.3f}  max={d['per_repo_max'][repo]:.3f}")

    # persist the feature so it can be audited / reused
    with open(os.path.join(OUT_DIR, "C-ai-adoption-feature.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["repo", "pr_number", "pr_type", "created_at",
                    "repo_prior_pr_count", NEW_FEATURE,
                    "repo_ai_adoption_rate_pooled_history", "actual_ai_review"])
        for rows, t in ((merged_rows, "merged"), (open_rows, "open")):
            for r in rows:
                w.writerow([r["repo"], r["pr_number"], t, r["created_at"],
                            r["repo_prior_pr_count"], f"{r[NEW_FEATURE]:.6f}",
                            f"{pooled[(r['repo'], int(r['pr_number']), t)]:.6f}",
                            r["_ai_actual"]])

    if args.feature_only:
        with open(os.path.join(OUT_DIR, "C-model-deltas.json"), "w") as f:
            json.dump(out, f, indent=1)
        return

    # ── modelling ───────────────────────────────────────────────────────────
    for tag, rows, target in (("T1", merged_rows, dp2_target),
                              ("T2", open_rows, dp4_target)):
        print(f"\n{'=' * 78}\n  {tag}\n{'=' * 78}")
        X24, y, repos = build_xy(rows, FEATURE_COLS_24, target)
        X25, y25, repos25 = build_xy(rows, FEATURE_COLS_25, target)
        assert (y == y25).all() and repos == repos25
        print(f"  n={len(y)}  positives={int(y.sum())} ({y.mean():.3f})")

        t0 = time.time()
        base = run_protocol(X24, y, repos, f"{tag}/24")
        print(f"  -- 24-feature protocol done in {time.time() - t0:.0f}s")
        t0 = time.time()
        plus = run_protocol(X25, y, repos, f"{tag}/25")
        print(f"  -- 25-feature protocol done in {time.time() - t0:.0f}s")

        ranked24, cb24 = shap_rank(X24, y, FEATURE_COLS_24, f"{tag}/24")
        ranked25, cb25 = shap_rank(X25, y, FEATURE_COLS_25, f"{tag}/25")
        names25 = [n for n, _ in ranked25]
        rank_new = names25.index(NEW_FEATURE) + 1
        total = sum(v for _, v in ranked25)
        share_new = dict(ranked25)[NEW_FEATURE] / total if total else float("nan")

        out[tag] = {
            "n": int(len(y)), "positives": int(y.sum()),
            "baseline_24": base, "with_25": plus,
            "shap_24": ranked24, "shap_25": ranked25,
            "catboost_importance_25": cb25,
            "new_feature_shap_rank": rank_new,
            "new_feature_shap_share": float(share_new),
            "n_features": 25,
        }

        print(f"\n  --- {tag}: deltas (25 features minus 24 features) ---")
        for proto in ("cv5", "loro"):
            for model in ("CatBoost", "LightGBM"):
                k = f"{proto}_{model}"
                b, p = base[k]["mean"], plus[k]["mean"]
                line = "  ".join(f"{m}: {p[m] - b[m]:+.4f} ({b[m]:.3f}->{p[m]:.3f})"
                                 for m in ("f1", "auc_roc", "auc_pr"))
                print(f"    {k:16s} {line}")
        print(f"    SHAP rank of {NEW_FEATURE}: {rank_new}/25 "
              f"({share_new * 100:.2f}% of total |SHAP|)")

    with open(os.path.join(OUT_DIR, "C-model-deltas.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nWrote C-ai-adoption-feature.csv and C-model-deltas.json into {OUT_DIR}")


if __name__ == "__main__":
    main()
