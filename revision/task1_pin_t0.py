#!/usr/bin/env python3
"""Task 1 — pin the observation date T0 and re-derive the staleness label.

Background
----------
`enrich/compute_features.py:121-127` computes the DP4/T2 staleness label as

    now = datetime.now(timezone.utc)
    is_stale = (now - updated_at).days > 30

`now` is the moment the *feature script* was executed, not the moment the data
was *collected*. The raw JSON was fetched 2026-04-08/09; `all_open_features.csv`
has an mtime of 2026-04-15. Every open PR therefore carries ~6 extra days of
apparent silence.

This script:
  1. gathers every candidate for the true observation instant,
  2. reconstructs (by interval arithmetic on the published labels) the `now`
     that the published run actually used,
  3. pins T0 per repository and recomputes the label at Delta = 30,
  4. reports how many labels move, pooled and per repository.

Read-only w.r.t. all existing data files. Writes only into
<DISSERTATION_DIR>/journal_articles/HAIT_2026/revision_results/.
"""

import json
import os
from datetime import timedelta

import numpy as np

import common as C


def reconstruct_published_now(rows, records):
    """Bound the `datetime.now()` that produced the published is_stale column.

    stale      <=> (now - upd).days > 30  <=>  now >= upd + 31d
    not stale  <=> (now - upd).days <= 30 <=>  now <  upd + 31d

    => lower bound = max over stale rows of (upd + 31d)
       upper bound = min over non-stale rows of (upd + 31d)
    """
    lo, hi = None, None
    lo_key, hi_key = None, None
    for r in rows:
        rec = records[(r["repo"], int(r["pr_number"]))]
        upd = rec["updated_at"]
        if upd is None:
            continue
        boundary = upd + timedelta(days=31)
        v = r.get("is_stale", "")
        if v in ("True", "true", "1"):
            if lo is None or boundary > lo:
                lo, lo_key = boundary, (r["repo"], r["pr_number"])
        else:
            if hi is None or boundary < hi:
                hi, hi_key = boundary, (r["repo"], r["pr_number"])
    return lo, hi, lo_key, hi_key


def main():
    os.makedirs(C.OUT_DIR, exist_ok=True)

    rows = C.load_open_rows()
    records, t0_max_upd, t0_mtime = C.load_raw_open()
    repos = sorted(t0_max_upd)

    print("=" * 100)
    print("TASK 1 — pinning the observation date T0")
    print("=" * 100)
    print(f"Open PRs in raw JSON : {len(records)}")
    print(f"Rows in published CSV: {len(rows)}")

    missing = [(r['repo'], r['pr_number']) for r in rows
               if (r["repo"], int(r["pr_number"])) not in records]
    print(f"CSV rows not found in raw JSON: {len(missing)}")
    assert not missing, missing[:5]

    # ── candidate observation instants ──────────────────────────────────
    print("\n--- Candidate T0 per repository ---")
    print(f"{'repository':26s} {'n':>5s}  {'max(updatedAt)':22s} {'file mtime (UTC)':22s} {'gap':>10s}")
    per_repo_n = {}
    for repo in repos:
        n = sum(1 for r in rows if r["repo"] == repo)
        per_repo_n[repo] = n
        gap = (t0_mtime[repo] - t0_max_upd[repo]).total_seconds() / 3600
        print(f"{repo:26s} {n:5d}  {t0_max_upd[repo].strftime('%Y-%m-%d %H:%M:%SZ'):22s} "
              f"{t0_mtime[repo].strftime('%Y-%m-%d %H:%M:%SZ'):22s} {gap:9.1f}h")

    global_max = max(t0_max_upd.values())
    print(f"\nGlobal max(updatedAt) across all repos: {global_max.isoformat()}")

    # ── reconstruct the `now` the published run used ────────────────────
    lo, hi, lo_key, hi_key = reconstruct_published_now(rows, records)
    print("\n--- Reconstructed datetime.now() of the published feature run ---")
    print(f"  lower bound : {lo.isoformat()}   (forced by stale PR {lo_key})")
    print(f"  upper bound : {hi.isoformat()}   (forced by non-stale PR {hi_key})")
    print(f"  => the published run executed inside [{lo.isoformat()}, {hi.isoformat()})")
    print(f"  drift vs global max(updatedAt): "
          f"{(lo - global_max).total_seconds()/86400:.2f} .. "
          f"{(hi - global_max).total_seconds()/86400:.2f} days")

    # ── relabel at Delta = 30 with pinned T0 ────────────────────────────
    y_pub = C.published_label(rows)
    y_new, silent, age = C.relabel(rows, records, t0_max_upd, 30)

    n = len(rows)
    changed = y_pub != y_new
    to_false = (y_pub == 1) & (y_new == 0)
    to_true = (y_pub == 0) & (y_new == 1)

    print("\n--- Label change at Delta = 30, T0 = per-repo max(updatedAt) ---")
    print(f"  published stale : {int(y_pub.sum())} / {n} = {y_pub.mean():.3f}")
    print(f"  pinned-T0 stale : {int(y_new.sum())} / {n} = {y_new.mean():.3f}")
    print(f"  labels changed  : {int(changed.sum())} ({changed.mean():.3f})")
    print(f"    stale -> active: {int(to_false.sum())}")
    print(f"    active -> stale: {int(to_true.sum())}")

    # ── per repository ──────────────────────────────────────────────────
    print("\n--- Per repository ---")
    print(f"{'repository':26s} {'n':>5s} {'stale_pub':>10s} {'sh_pub':>7s} "
          f"{'stale_T0':>9s} {'sh_T0':>7s} {'changed':>8s}")
    per_repo = []
    for repo in repos:
        m = np.array([r["repo"] == repo for r in rows])
        sp, sn = int(y_pub[m].sum()), int(y_new[m].sum())
        ch = int((y_pub[m] != y_new[m]).sum())
        per_repo.append({
            "repo": repo, "n": int(m.sum()),
            "stale_published": sp, "share_published": round(sp / m.sum(), 3),
            "stale_pinned_T0": sn, "share_pinned_T0": round(sn / m.sum(), 3),
            "changed": ch, "changed_share": round(ch / m.sum(), 3),
            "T0": t0_max_upd[repo].isoformat(),
        })
        print(f"{repo:26s} {int(m.sum()):5d} {sp:10d} {sp/m.sum():7.3f} "
              f"{sn:9d} {sn/m.sum():7.3f} {ch:8d}")

    # ── robustness of the T0 choice itself ──────────────────────────────
    print("\n--- Robustness of the T0 choice (Delta = 30) ---")
    variants = {
        "per-repo max(updatedAt)  [CHOSEN]": t0_max_upd,
        "per-repo file mtime": t0_mtime,
        "global max(updatedAt)": {r: global_max for r in repos},
    }
    t0_variants = {}
    for name, mapping in variants.items():
        yy, _, _ = C.relabel(rows, records, mapping, 30)
        t0_variants[name] = int(yy.sum())
        print(f"  {name:36s} stale = {int(yy.sum()):5d}  ({yy.mean():.3f})   "
              f"vs chosen: {int(yy.sum()) - int(y_new.sum()):+d}")

    # ── silence distribution near the boundary ──────────────────────────
    print("\n--- Days of silence at pinned T0: distribution near the Delta=30 boundary ---")
    for lo_b, hi_b in [(0, 15), (16, 24), (25, 30), (31, 36), (37, 45), (46, 90), (91, 10 ** 6)]:
        c = int(((silent >= lo_b) & (silent <= hi_b)).sum())
        label = f"{lo_b}-{hi_b}" if hi_b < 10 ** 6 else f">={lo_b}"
        print(f"  silence {label:>10s} days : {c:5d} ({c/n:.3f})")
    print(f"  median silence: {np.median(silent):.1f} d, mean: {silent.mean():.1f} d, "
          f"max: {silent.max():.0f} d")

    # ── persist ─────────────────────────────────────────────────────────
    payload = {
        "n_open_prs": n,
        "t0_choice": "per-repository max(updatedAt) over that repository's open PRs",
        "t0_by_repo": {r: t0_max_upd[r].isoformat() for r in repos},
        "file_mtime_by_repo": {r: t0_mtime[r].isoformat() for r in repos},
        "global_max_updated_at": global_max.isoformat(),
        "reconstructed_published_now": {"lower": lo.isoformat(), "upper": hi.isoformat()},
        "delta": 30,
        "stale_published": int(y_pub.sum()),
        "share_published": round(float(y_pub.mean()), 3),
        "stale_pinned": int(y_new.sum()),
        "share_pinned": round(float(y_new.mean()), 3),
        "changed": int(changed.sum()),
        "changed_share": round(float(changed.mean()), 3),
        "stale_to_active": int(to_false.sum()),
        "active_to_stale": int(to_true.sum()),
        "per_repo": per_repo,
        "t0_variant_stale_counts": t0_variants,
    }
    out = os.path.join(C.OUT_DIR, "B-task1-t0.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {out}")

    # also cache silence/age vectors for tasks 2 and 3
    np.savez(os.path.join(C.OUT_DIR, "B-silence-age.npz"),
             silent=silent, age=age,
             repo=np.array([r["repo"] for r in rows]),
             pr_number=np.array([int(r["pr_number"]) for r in rows]),
             y_published=y_pub)
    print(f"Saved {os.path.join(C.OUT_DIR, 'B-silence-age.npz')}")


if __name__ == "__main__":
    main()
