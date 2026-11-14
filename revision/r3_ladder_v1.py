#!/usr/bin/env python3
"""Contribution table, corpus v1 steps (dissertation, chapter 3).

Runs after `taskF1_duration_language.py --stage prep` of this version (new
account lists) has written F-pr-level.csv into F1_OUT_DIR. Every step changes
one thing relative to the previous one; CatBoost, LORO, the four targets:

  S1_lists        ITIA F1 configuration with the corpus v2 account lists
  S2_no_bot_auth  + PRs whose author is ordinary automation removed
                    (taskC_common.in_main_population on login and account type)
  S3_t0_features  + features restricted to the t0 allow-list as far as corpus v1
                    allows: final size, labels, has_tests/docs/ci, the final
                    is_draft and author_merge_rate (no closed PRs in v1) dropped;
                    merge velocity recomputed strictly before t0
S0 (exact ITIA, tag v1.2.0) runs separately from an archive of the tag; the
corpus v2 steps (S4*, S5) are in r3_v2_models.py --block ladder.

Imputation and protocol are the ITIA ones (taskF1.Runner) so that S1 equals the
ITIA main LORO numbers except for the account lists.

    F1_OUT_DIR=... F1_D_LABELS=... PRR_RAW_DIR=... PRR_PROCESSED_DIR=... \
        python revision/r3_ladder_v1.py
"""
import bisect
import os
import sys
from datetime import timedelta

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import taskF1_duration_language as f1  # noqa: E402
from review_iterations import parse_dt  # noqa: E402
from taskC_common import in_main_population  # noqa: E402

DROP_S3 = ["lines_added", "lines_deleted", "files_changed", "has_tests", "has_docs",
           "touches_ci", "label_count", "is_draft", "author_merge_rate"]
TARGETS = ["IT", "TM", "TFR", "ST"]


def strict_velocity(created, merged, repo):
    """Merges of the same repository in [t0 - 7 d, t0) and [t0 - 30 d, t0)."""
    v7 = np.zeros(len(created))
    v30 = np.zeros(len(created))
    by = {}
    for i, r in enumerate(repo):
        by.setdefault(r, []).append(i)
    for r, idx in by.items():
        ms = sorted(merged[i] for i in idx)
        for i in idx:
            t0 = created[i]
            hi = bisect.bisect_left(ms, t0)
            v7[i] = hi - bisect.bisect_left(ms, t0 - timedelta(days=7))
            v30[i] = hi - bisect.bisect_left(ms, t0 - timedelta(days=30))
    return v7, v30


def main():
    f1.configure("t0min1")
    f1.METRICS_LONG = os.path.join(f1.OUT_DIR, "ladder-v1-metrics.jsonl")
    f1.OOF_FILE = os.path.join(f1.OUT_DIR, "ladder-v1-oof.npz")
    d = f1.Data()
    pub = f1._read_published_rows()
    in_main = np.array([in_main_population(r["author_login"], r.get("author_typename") or None)
                        for r in pub])
    cols = d.f1_cols
    X = d.X_f1_raw.copy()
    created = [parse_dt(r["created_at"]) for r in pub]
    merged = [parse_dt(r["merged_at"]) for r in pub]
    v7, v30 = strict_velocity(created, merged, d.repo)
    X[:, cols.index("merge_velocity_7d")] = v7
    X[:, cols.index("merge_velocity_30d")] = v30
    keep = [j for j, c in enumerate(cols) if c not in DROP_S3]
    # Runner selects matrices by name from X_by_commits; register S3 there and
    # keep its column subset by blanking the dropped columns to a constant.
    Xs3 = X.copy()
    for j, c in enumerate(cols):
        if c in DROP_S3:
            Xs3[:, j] = 0.0
    d.X_by_commits["s3_t0"] = Xs3
    print(f"S3 features ({len(keep)}): {[cols[j] for j in keep]}", flush=True)
    print(f"rows {len(d.t)}, main population {int(in_main.sum())}", flush=True)
    if "--dry" in sys.argv:
        return
    run = f1.Runner(d, resume=True)
    for target in TARGETS:
        print(f"== ladder v1 {target}", flush=True)
        run.run("S1_lists", target, target, "CatBoost", ["loro"], keep_oof=True)
        run.run("S2_no_bot_auth", target, target, "CatBoost", ["loro"],
                mask=d.masks[target] & in_main, keep_oof=True)
        run.run("S3_t0_features", target, target, "CatBoost", ["loro"],
                features="commits:s3_t0", mask=d.masks[target] & in_main, keep_oof=True)
    run.save_oof()
    print(f"ladder v1 done: {run.n_fit} new fits", flush=True)


if __name__ == "__main__":
    main()
