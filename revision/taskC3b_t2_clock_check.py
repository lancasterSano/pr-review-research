#!/usr/bin/env python3
"""Task C3b — is the T2 gain from `repo_ai_adoption_rate` an AI signal or a clock?

Motivation
----------
Adding `repo_ai_adoption_rate` to T2 (staleness) lifts 5-fold CV F1 from 0.851
to 0.904 and makes it the top SHAP feature (33% of importance). That looks far
too good, and there is an obvious alternative explanation.

AI adoption inside a repository rises over calendar time. The T2 label is
derived from a single collection snapshot: a PR is stale if more than 30 days
of silence had elapsed AT COLLECTION TIME, so a PR opened a week before
collection is mechanically non-stale (right-censoring, already acknowledged in
the article's construct-validity paragraph). Any submission-time feature that
increases monotonically with calendar time therefore predicts the T2 label
without carrying any information about AI.

This script separates the two explanations:
  1. rank correlation of the feature with PR age at the observation date;
  2. the same protocol with `pr_age_days` substituted for the AI feature --
     if a bare clock reproduces the gain, the gain is a clock;
  3. both features together -- does the AI feature add anything on top of the
     clock?

T0 (observation date) is computed here as the maximum `updatedAt` over the open
PRs of each repository, i.e. the last moment the snapshot can attest to. It is
used ONLY for this diagnostic; no published artefact is modified.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taskC_common import (  # noqa: E402
    OUT_DIR, OPEN_CSV, RAW_DIR, REPO_DIRS, dir_to_repo, parse_dt, load_processed,
)
from taskC3_ai_adoption_feature import (  # noqa: E402
    FEATURE_COLS_24, NEW_FEATURE, load_ai_flags, compute_adoption,
    build_xy, dp4_target, run_protocol, shap_rank, spearman,
)

import json as _json


def t0_by_repo():
    out = {}
    for d in REPO_DIRS:
        path = os.path.join(RAW_DIR, d, "open_prs.json")
        with open(path) as f:
            prs = _json.load(f)
        mx = None
        for pr in prs:
            u = parse_dt(pr.get("updatedAt"))
            if u and (mx is None or u > mx):
                mx = u
        out[dir_to_repo(d)] = mx
    return out


def main():
    ai_flags = load_ai_flags()
    rows = load_processed(OPEN_CSV)
    compute_adoption(rows, "open", ai_flags)

    t0 = t0_by_repo()
    for r in rows:
        c = parse_dt(r["created_at"])
        r["pr_age_days"] = (t0[r["repo"]] - c).total_seconds() / 86400.0

    feat = np.array([r[NEW_FEATURE] for r in rows])
    age = np.array([r["pr_age_days"] for r in rows])
    stale = np.array([1.0 if r["is_stale"] in ("True", "true", "1") else 0.0 for r in rows])
    reps = np.array([r["repo"] for r in rows])

    print("=" * 84)
    print("1) Is the AI-adoption feature a clock?")
    print("=" * 84)
    print(f"  pooled  Spearman(adoption, pr_age_days)   = {spearman(feat, age):+.3f}")
    print(f"  pooled  Spearman(adoption, is_stale)      = {spearman(feat, stale):+.3f}")
    print(f"  pooled  Spearman(pr_age_days, is_stale)   = {spearman(age, stale):+.3f}")
    print("\n  within repository:")
    print(f"  {'repository':26s} {'n':>6} {'rho(adopt,age)':>16} {'rho(adopt,stale)':>18} "
          f"{'rho(age,stale)':>16}")
    for repo in sorted(set(reps)):
        m = reps == repo
        print(f"  {repo:26s} {m.sum():>6} {spearman(feat[m], age[m]):>16.3f} "
              f"{spearman(feat[m], stale[m]):>18.3f} {spearman(age[m], stale[m]):>16.3f}")

    print("\n" + "=" * 84)
    print("2) Same protocol with a bare clock instead of the AI feature")
    print("=" * 84)
    X24, y, repos = build_xy(rows, FEATURE_COLS_24, dp4_target)
    X_ai, _, _ = build_xy(rows, FEATURE_COLS_24 + [NEW_FEATURE], dp4_target)
    X_age, _, _ = build_xy(rows, FEATURE_COLS_24 + ["pr_age_days"], dp4_target)
    X_both, _, _ = build_xy(rows, FEATURE_COLS_24 + ["pr_age_days", NEW_FEATURE], dp4_target)

    res = {}
    for tag, X in (("24 baseline", X24), ("24+ai_adoption", X_ai),
                   ("24+pr_age", X_age), ("24+pr_age+ai_adoption", X_both)):
        print(f"\n  --- {tag} ---")
        res[tag] = run_protocol(X, y, repos, tag)

    print("\n" + "=" * 84)
    print("3) Summary (5-fold CV, CatBoost)")
    print("=" * 84)
    print(f"  {'configuration':26s} {'F1':>8} {'AUC-ROC':>9} {'AUC-PR':>9} || "
          f"{'LORO F1':>9} {'LORO AUC':>9}")
    for tag in res:
        c = res[tag]["cv5_CatBoost"]["mean"]
        l = res[tag]["loro_CatBoost"]["mean"]
        print(f"  {tag:26s} {c['f1']:>8.3f} {c['auc_roc']:>9.3f} {c['auc_pr']:>9.3f} || "
              f"{l['f1']:>9.3f} {l['auc_roc']:>9.3f}")

    ranked, _ = shap_rank(X_both, y, FEATURE_COLS_24 + ["pr_age_days", NEW_FEATURE],
                          "T2/26")
    names = [n for n, _ in ranked]
    tot = sum(v for _, v in ranked)
    print(f"\n  SHAP with both present: pr_age_days rank {names.index('pr_age_days') + 1}/26 "
          f"({dict(ranked)['pr_age_days'] / tot * 100:.2f}%), "
          f"{NEW_FEATURE} rank {names.index(NEW_FEATURE) + 1}/26 "
          f"({dict(ranked)[NEW_FEATURE] / tot * 100:.2f}%)")

    with open(os.path.join(OUT_DIR, "C-t2-clock-check.json"), "w") as f:
        json.dump({
            "spearman_pooled": {
                "adoption_vs_age": spearman(feat, age),
                "adoption_vs_stale": spearman(feat, stale),
                "age_vs_stale": spearman(age, stale),
            },
            "spearman_within_repo": {
                repo: {
                    "n": int((reps == repo).sum()),
                    "adoption_vs_age": spearman(feat[reps == repo], age[reps == repo]),
                    "adoption_vs_stale": spearman(feat[reps == repo], stale[reps == repo]),
                    "age_vs_stale": spearman(age[reps == repo], stale[reps == repo]),
                } for repo in sorted(set(reps))
            },
            "protocols": {k: {kk: vv for kk, vv in v.items()} for k, v in res.items()},
            "shap_26": ranked,
        }, f, indent=1, default=float)
    print(f"\nWrote C-t2-clock-check.json into {OUT_DIR}")


if __name__ == "__main__":
    main()
