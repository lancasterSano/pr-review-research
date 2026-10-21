#!/usr/bin/env python3
"""Task 3 — age stratification and right-censoring of the staleness label (R2.3).

Two questions:

(a) Stratification. Every open PR gets an age at the pinned observation date,
    age(p) = T0(repo) - created_at(p), bucketed into <=30 / 31-90 / 91-365 / >365
    days. Per bucket we report n, the stale share at Delta=30, and 5-fold CV
    performance of CatBoost / LightGBM trained *and* evaluated inside the bucket.

(b) Right-censoring. Because updated_at >= created_at, days_silent <= age always.
    A PR whose age at T0 is <= Delta therefore *cannot* be labelled stale — the
    label is structurally forced to 0, independently of any behaviour of the PR.
    We quantify that group for every Delta and re-run the model on the complement
    (age > Delta), which is the only population where the label carries
    information in both directions.

Full survival analysis (Kaplan-Meier / Cox) is deliberately out of scope; it is
declared future work in the revision plan.
"""

import csv
import json
import os
import time
import warnings

import numpy as np

import common as C
from task2_delta_sensitivity import cv5, loro, make_models, METRIC_KEYS

warnings.filterwarnings("ignore")

BUCKETS = [
    ("<=30", 0, 30),
    ("31-90", 31, 90),
    ("91-365", 91, 365),
    (">365", 366, 10 ** 9),
]

# A stratum is modelled only if it is large enough for 5-fold stratified CV to be
# meaningful: both classes present and at least MIN_MINORITY positives/negatives.
MIN_N = 100
MIN_MINORITY = 25


def model_subset(name, X, y, repos_arr, mask, results, extra):
    n = int(mask.sum())
    ys = y[mask]
    pos = int(ys.sum())
    neg = n - pos
    base = {"stratum": name, "n": n, "n_stale": pos,
            "stale_share": round(pos / n, 3) if n else float("nan")}
    base.update(extra)

    print(f"\n--- {name}: n={n}, stale={pos} ({pos / n if n else float('nan'):.3f}) ---")

    reason = None
    if n < MIN_N:
        reason = f"n={n} < {MIN_N}"
    elif pos == 0 or neg == 0:
        reason = ("all rows are non-stale (label structurally forced)" if pos == 0
                  else "all rows are stale")
    elif min(pos, neg) < MIN_MINORITY:
        reason = f"minority class has only {min(pos, neg)} rows (< {MIN_MINORITY})"

    if reason:
        print(f"    NOT MODELLED: {reason}")
        for model_name in make_models():
            row = dict(base)
            row["model"] = model_name
            row["modelled"] = False
            row["skip_reason"] = reason
            for k in METRIC_KEYS:
                row[f"cv5_{k}"] = ""
                row[f"cv5_{k}_std"] = ""
            row["loro_f1"] = ""
            row["loro_auc_roc"] = ""
            row["loro_n_repos"] = ""
            results.append(row)
        return

    Xs, ys_, rs = X[mask], y[mask], repos_arr[mask]
    for model_name, model_fn in make_models().items():
        t = time.time()
        mean, std, _ = cv5(Xs, ys_, model_fn)
        lm, lper = loro(Xs, ys_, rs, model_fn)
        row = dict(base)
        row["model"] = model_name
        row["modelled"] = True
        row["skip_reason"] = ""
        for k in METRIC_KEYS:
            row[f"cv5_{k}"] = round(mean.get(k, float("nan")), 3)
            row[f"cv5_{k}_std"] = round(std.get(k, float("nan")), 3)
        row["loro_f1"] = round(lm["f1"], 3) if lm else ""
        row["loro_auc_roc"] = round(lm.get("auc_roc", float("nan")), 3) if lm else ""
        row["loro_n_repos"] = len(lper)
        results.append(row)
        print(f"    {model_name:10s} CV5 f1={row['cv5_f1']:.3f} auc_roc={row['cv5_auc_roc']:.3f} "
              f"acc={row['cv5_accuracy']:.3f} prec={row['cv5_precision']:.3f} "
              f"rec={row['cv5_recall']:.3f} auc_pr={row['cv5_auc_pr']:.3f} | "
              f"LORO f1={row['loro_f1']} auc={row['loro_auc_roc']} ({len(lper)} repos) "
              f"[{time.time() - t:.0f}s]")


def main():
    os.makedirs(C.OUT_DIR, exist_ok=True)

    rows = C.load_open_rows()
    records, t0_by_repo, _ = C.load_raw_open()
    X = C.build_matrix(rows)
    repos_arr = np.array([r["repo"] for r in rows])
    n_total = len(rows)

    y30, silent, age = C.relabel(rows, records, t0_by_repo, 30)

    # sanity: silence can never exceed age
    assert (silent <= age + 1e-9).all(), "found days_silent > age — impossible"

    print("=" * 92)
    print("TASK 3 — age stratification and right-censoring")
    print("=" * 92)
    print(f"n = {n_total} open PRs, T0 = per-repository max(updatedAt)")
    print(f"age at T0 (days): min={age.min():.0f} median={np.median(age):.0f} "
          f"mean={age.mean():.1f} max={age.max():.0f}")

    results = []

    # ── (a) age strata, label at Delta = 30 ─────────────────────────────
    print("\n" + "=" * 92)
    print("  (a) Age strata (label: pinned T0, Delta = 30)")
    print("=" * 92)
    strata_summary = []
    for name, lo, hi in BUCKETS:
        mask = (age >= lo) & (age <= hi)
        n = int(mask.sum())
        pos = int(y30[mask].sum())
        strata_summary.append({
            "bucket": name, "n": n, "share_of_total": round(n / n_total, 3),
            "n_stale": pos, "stale_share": round(pos / n, 3) if n else None,
        })
        model_subset(f"age {name} d", X, y30, repos_arr, mask, results,
                     extra={"kind": "age_bucket", "delta": 30, "bucket": name})

    print("\n  Age bucket summary (Delta = 30):")
    print(f"  {'bucket':>10s} {'n':>6s} {'%of total':>10s} {'stale':>7s} {'stale share':>12s}")
    for s in strata_summary:
        print(f"  {s['bucket']:>10s} {s['n']:6d} {s['share_of_total']:10.3f} "
              f"{s['n_stale']:7d} {s['stale_share']:12.3f}")

    # per-repo bucket counts
    per_repo_buckets = {}
    for repo in sorted(set(repos_arr.tolist())):
        rm = repos_arr == repo
        per_repo_buckets[repo] = {}
        for name, lo, hi in BUCKETS:
            m = rm & (age >= lo) & (age <= hi)
            per_repo_buckets[repo][name] = {
                "n": int(m.sum()),
                "n_stale": int(y30[m].sum()),
                "stale_share": round(float(y30[m].mean()), 3) if m.sum() else None,
            }

    # ── (b) right-censoring ─────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("  (b) Structural right-censoring: PRs younger than Delta cannot be stale")
    print("=" * 92)
    print(f"  {'Delta':>6s} {'censored n':>11s} {'censored %':>11s} "
          f"{'eligible n':>11s} {'stale|all':>10s} {'stale|eligible':>15s}")
    censoring = []
    for delta in C.DELTAS:
        y, _, _ = C.relabel(rows, records, t0_by_repo, delta)
        cens = age <= delta
        elig = ~cens
        assert y[cens].sum() == 0, "a censored PR was labelled stale — logic error"
        rec = {
            "delta": delta,
            "censored_n": int(cens.sum()),
            "censored_share": round(float(cens.mean()), 3),
            "eligible_n": int(elig.sum()),
            "stale_share_all": round(float(y.mean()), 3),
            "stale_share_eligible": round(float(y[elig].mean()), 3) if elig.sum() else None,
            "n_stale_all": int(y.sum()),
        }
        censoring.append(rec)
        print(f"  {delta:6d} {rec['censored_n']:11d} {rec['censored_share']:11.3f} "
              f"{rec['eligible_n']:11d} {rec['stale_share_all']:10.3f} "
              f"{rec['stale_share_eligible']:15.3f}")

    # ── (c) honest evaluation population: age > Delta ───────────────────
    print("\n" + "=" * 92)
    print("  (c) Model performance on the uncensored population (age > Delta)")
    print("=" * 92)
    for delta in C.DELTAS:
        y, _, _ = C.relabel(rows, records, t0_by_repo, delta)
        mask = age > delta
        model_subset(f"uncensored age>{delta}, Delta={delta}", X, y, repos_arr, mask, results,
                     extra={"kind": "uncensored", "delta": delta, "bucket": f"age>{delta}"})

    # ── persist ─────────────────────────────────────────────────────────
    cols = ["kind", "delta", "bucket", "stratum", "model", "modelled", "skip_reason",
            "n", "n_stale", "stale_share"] + \
           [f"cv5_{k}" for k in METRIC_KEYS] + [f"cv5_{k}_std" for k in METRIC_KEYS] + \
           ["loro_f1", "loro_auc_roc", "loro_n_repos"]
    csv_path = os.path.join(C.OUT_DIR, "B-age-strata.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved {csv_path}")

    json_path = os.path.join(C.OUT_DIR, "B-age-strata.json")
    with open(json_path, "w") as f:
        json.dump({
            "n_total": n_total,
            "age_stats": {"min": float(age.min()), "median": float(np.median(age)),
                          "mean": round(float(age.mean()), 1), "max": float(age.max())},
            "strata_summary": strata_summary,
            "per_repo_buckets": per_repo_buckets,
            "censoring": censoring,
            "results": results,
        }, f, indent=2)
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
