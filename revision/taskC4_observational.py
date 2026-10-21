#!/usr/bin/env python3
"""Task C4 — observational comparison of PRs with and without an AI review.

Compares review iterations (`review_rounds`) and time-to-merge
(`time_to_merge_hours`) between PRs that received an AI review and PRs that did
not, holding TWO things fixed:

  * repository — the AI/no-AI split is wildly unbalanced across repositories
    (vscode alone supplies most AI cases), so any pooled comparison would
    measure repository differences, not AI;
  * time period — no AI review event exists in the corpus before 2025-02-27, so
    the comparison is restricted to PRs created in 2025-2026, where both groups
    coexist.

THIS IS PURELY DESCRIPTIVE. Adoption is not random: teams enable an AI reviewer
on the repositories, and often the paths and PR types, where they already expect
review effort. Nothing here supports a causal reading, and no prediction claim
is made from it. Reported so the AI Assistant lane of the BPMN model becomes a
measured element of the process rather than a declaration.

Test: two-sided Mann-Whitney U (rank-sum) with a normal approximation, tie
correction and continuity correction, plus the rank-biserial correlation as an
effect size. Implemented locally to avoid a SciPy dependency; the implementation
was checked against scipy.stats.mannwhitneyu(alternative="two-sided",
use_continuity=True) on seven heavy-tie and continuous samples and reproduces
both U and p exactly.

Output: C-ai-observational.csv + console tables.
"""

import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taskC_common import OUT_DIR, MERGED_CSV, TABLE2_ORDER, load_processed  # noqa: E402

FLAGS_CSV = os.path.join(OUT_DIR, "C-ai-pr-flags.csv")
MIN_N = 20  # minimum group size for a test to be reported


def mannwhitneyu(x, y):
    """Two-sided Mann-Whitney U with tie-corrected normal approximation.

    Returns (U_x, p, rank_biserial). rank_biserial = 2*U_x/(n1*n2) - 1, i.e.
    +1 when every x exceeds every y, -1 in the opposite case, 0 at stochastic
    equality.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan")
    allv = np.concatenate([x, y])
    order = np.argsort(allv, kind="mergesort")
    sorted_v = allv[order]
    ranks = np.empty(len(allv), dtype=np.float64)
    i = 0
    tie_term = 0.0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        t = j - i + 1
        if t > 1:
            tie_term += t ** 3 - t
        i = j + 1
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    n = n1 + n2
    mu = n1 * n2 / 2.0
    sigma2 = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))
    if sigma2 <= 0:
        return u1, float("nan"), 2.0 * u1 / (n1 * n2) - 1.0
    z = (u1 - mu)
    z = (z - math.copysign(0.5, z)) / math.sqrt(sigma2)   # continuity correction
    p = math.erfc(abs(z) / math.sqrt(2.0))                 # two-sided
    return u1, p, 2.0 * u1 / (n1 * n2) - 1.0


def med(v):
    return float(np.median(v)) if len(v) else float("nan")


def main():
    ai = {}
    for r in load_processed(FLAGS_CSV):
        if r["pr_type"] == "merged":
            ai[(r["repo"], int(r["pr_number"]))] = int(r["has_ai_review"])

    rows = load_processed(MERGED_CSV)

    # "rounds" and "ttm" are the outcomes asked for; "lines_added"/"files_changed"
    # are carried along as a selection-effect check — if AI review is switched on
    # preferentially for bigger changes, that alone would explain more iterations.
    METRICS = {"rounds": "review_rounds", "ttm": "time_to_merge_hours",
               "lines_added": "lines_added", "files_changed": "files_changed"}

    groups = defaultdict(lambda: {"ai": defaultdict(list), "no": defaultdict(list)})
    n_skipped = 0
    for r in rows:
        year = (r["created_at"] or "")[:4]
        if year not in ("2025", "2026"):
            continue
        key = (r["repo"], int(r["pr_number"]))
        if key not in ai:
            n_skipped += 1
            continue
        g = "ai" if ai[key] else "no"
        for name, col in METRICS.items():
            try:
                groups[r["repo"]][g][name].append(float(r[col]))
            except (TypeError, ValueError, KeyError):
                pass

    out_rows = []
    for repo in TABLE2_ORDER + ["__POOLED__"]:
        if repo == "__POOLED__":
            get = lambda g, m: [v for rp in groups for v in groups[rp][g][m]]  # noqa: E731
            label = "Pooled 2025-2026 (CONFOUNDED by repo mix - do not read as an effect)"
        else:
            if repo not in groups:
                continue
            get = lambda g, m, _r=repo: groups[_r][g][m]  # noqa: E731
            label = repo

        row = {"repo": label,
               "n_ai": len(get("ai", "rounds")), "n_no_ai": len(get("no", "rounds"))}

        for metric in ("rounds", "ttm", "lines_added", "files_changed"):
            a, b = get("ai", metric), get("no", metric)
            metric = "ttm_hours" if metric == "ttm" else metric
            row[f"median_{metric}_ai"] = round(med(a), 3)
            row[f"median_{metric}_no_ai"] = round(med(b), 3)
            row[f"mean_{metric}_ai"] = round(float(np.mean(a)), 3) if a else None
            row[f"mean_{metric}_no_ai"] = round(float(np.mean(b)), 3) if b else None
            if len(a) >= MIN_N and len(b) >= MIN_N:
                u, p, rb = mannwhitneyu(a, b)
                row[f"U_{metric}"] = round(u, 1)
                row[f"p_{metric}"] = p
                row[f"rank_biserial_{metric}"] = round(rb, 3)
            else:
                row[f"U_{metric}"] = None
                row[f"p_{metric}"] = None
                row[f"rank_biserial_{metric}"] = None
        out_rows.append(row)

    cols = list(out_rows[0].keys())
    with open(os.path.join(OUT_DIR, "C-ai-observational.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            rr = dict(r)
            for k in list(rr):
                if k.startswith("p_") and rr[k] is not None:
                    rr[k] = f"{rr[k]:.3e}"
            w.writerow(rr)

    def fmtp(p):
        if p is None:
            return "     n/a"
        if p < 1e-3:
            return f"{p:8.1e}"
        return f"{p:8.3f}"

    def f3(v):
        return f"{v:8.3f}" if v is not None and v == v else "     n/a"

    print(f"PRs created 2025-2026 in the merged analysis set: "
          f"{sum(r['n_ai'] + r['n_no_ai'] for r in out_rows if r['repo'] != out_rows[-1]['repo'])}"
          f"  (rows without an AI flag: {n_skipped})")
    print("\nReview iterations (review_rounds), 2025-2026, within repository")
    print(f"{'repository':60s} {'n_AI':>6} {'n_noAI':>7} {'med_AI':>8} {'med_noAI':>9} "
          f"{'r_rb':>7} {'p':>9}")
    for r in out_rows:
        print(f"{r['repo'][:60]:60s} {r['n_ai']:>6} {r['n_no_ai']:>7} "
              f"{f3(r['median_rounds_ai'])} {f3(r['median_rounds_no_ai'])} "
              f"{f3(r['rank_biserial_rounds']):>8} {fmtp(r['p_rounds'])}")

    print("\nTime to merge (hours), 2025-2026, within repository")
    print(f"{'repository':60s} {'n_AI':>6} {'n_noAI':>7} {'med_AI':>8} {'med_noAI':>9} "
          f"{'r_rb':>7} {'p':>9}")
    for r in out_rows:
        print(f"{r['repo'][:60]:60s} {r['n_ai']:>6} {r['n_no_ai']:>7} "
              f"{f3(r['median_ttm_hours_ai'])} {f3(r['median_ttm_hours_no_ai'])} "
              f"{f3(r['rank_biserial_ttm_hours']):>8} {fmtp(r['p_ttm_hours'])}")

    print("\nSELECTION CHECK — change size, 2025-2026, within repository")
    print(f"{'repository':40s} {'metric':14s} {'med_AI':>9} {'med_noAI':>9} {'r_rb':>8} {'p':>9}")
    for r in out_rows:
        for m in ("lines_added", "files_changed"):
            print(f"{r['repo'][:40]:40s} {m:14s} {f3(r[f'median_{m}_ai']):>9} "
                  f"{f3(r[f'median_{m}_no_ai']):>9} {f3(r[f'rank_biserial_{m}']):>8} "
                  f"{fmtp(r[f'p_{m}'])}")

    print(f"\nrank-biserial r_rb > 0  =>  the AI group tends to the LARGER value.")
    print(f"Tests reported only where both groups have n >= {MIN_N}.")
    print(f"\nWrote C-ai-observational.csv into {OUT_DIR}")


if __name__ == "__main__":
    main()
