#!/usr/bin/env python3
"""Shared helpers for the HAIT 2026 revision experiments (staleness relabelling).

NOTHING in this module mutates the published artefacts. It only *reads*:
  - data/raw/*/open_prs.json          (raw GitHub GraphQL nodes)
  - data/processed/all_open_features.csv  (published feature matrix)

and writes into <DISSERTATION_DIR>/journal_articles/HAIT_2026/revision_results/
(see paths.py).

Key idea
--------
The 24 predictor features published in Table 4 are *label-independent*: they are
derived from PR content, author history and project context, never from
`is_stale`. So instead of re-running the whole feature pipeline (which would risk
diverging from the published numbers) we re-use the published feature matrix
verbatim and only swap the *label* column, recomputed from the raw JSON against a
pinned observation date T0.
"""

import csv
import json
import os
from datetime import datetime, timezone

import numpy as np

csv.field_size_limit(10 ** 9)

from paths import DISSERTATION

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
OPEN_CSV = os.path.join(PROCESSED_DIR, "all_open_features.csv")
OUT_DIR = os.path.join(DISSERTATION, "journal_articles", "HAIT_2026", "revision_results")

# Exactly the 24 columns used by week4_modeling.py (FEATURE_COLS), in the same order.
FEATURE_COLS = [
    # Group 1: PR-level
    "lines_added", "lines_deleted", "files_changed", "num_commits",
    "has_tests", "has_docs", "touches_ci",
    "desc_length", "title_length", "has_screenshots", "has_code_blocks",
    "references_issues", "is_draft", "label_count",
    "day_of_week", "hour_of_day",
    # Group 2: Developer history
    "author_merge_rate", "author_avg_review_rounds",
    "author_commit_frequency", "author_tenure_days", "author_pr_count",
    # Group 3: Project context
    "open_pr_count", "merge_velocity_7d", "merge_velocity_30d",
]

DELTAS = [15, 30, 45, 60, 90]

# Concurrency budget — other agents are running on this machine.
THREADS = 4


# ── time parsing ────────────────────────────────────────────────────────────

def parse_dt(s):
    """Parse a GitHub ISO-8601 'Z' timestamp into an aware UTC datetime."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


# ── raw JSON ────────────────────────────────────────────────────────────────

def repo_dirs():
    """Directory names under data/raw that actually hold an open_prs.json."""
    return sorted(
        d for d in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, d))
        and os.path.exists(os.path.join(RAW_DIR, d, "open_prs.json"))
    )


def dir_to_repo(d):
    """'woocommerce_woocommerce' -> 'woocommerce/woocommerce' (same rule as compute_features.py)."""
    return d.replace("_", "/", 1)


def load_raw_open():
    """Return {(repo, pr_number): {'created_at': dt, 'updated_at': dt}} plus per-repo T0.

    T0 (per repository) = max(updatedAt) over that repository's open PRs.
    Rationale is documented in task1_pin_t0.py.
    """
    records = {}
    t0_by_repo = {}
    mtime_by_repo = {}
    for d in repo_dirs():
        repo = dir_to_repo(d)
        path = os.path.join(RAW_DIR, d, "open_prs.json")
        with open(path) as f:
            prs = json.load(f)
        max_upd = None
        for pr in prs:
            upd = parse_dt(pr.get("updatedAt"))
            cre = parse_dt(pr.get("createdAt"))
            records[(repo, int(pr["number"]))] = {
                "created_at": cre,
                "updated_at": upd,
                "state": pr.get("state"),
            }
            if upd and (max_upd is None or upd > max_upd):
                max_upd = upd
        t0_by_repo[repo] = max_upd
        mtime_by_repo[repo] = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
    return records, t0_by_repo, mtime_by_repo


# ── published feature matrix ────────────────────────────────────────────────

def to_float(val):
    """Identical coercion rule to week4_modeling.to_float."""
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


def load_open_rows():
    """Read the published open-PR feature CSV as a list of dicts (order preserved)."""
    with open(OPEN_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_matrix(rows):
    """Build X (n x 24) with median imputation, exactly as week4_modeling.load_data does.

    IMPORTANT: week4_modeling imputes column medians over the *whole* loaded set
    (all rows whose label parsed), before any CV split. Since the label is never
    used for imputation and every open-PR row has a parseable label under every
    Delta we test, the imputed X is identical across Deltas. We therefore build it
    once and reuse it, which also guarantees byte-identical inputs to the
    published Table 4 run.
    """
    X = np.array(
        [[to_float(r.get(c, "")) for c in FEATURE_COLS] for r in rows],
        dtype=np.float64,
    )
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col)
        if mask.any():
            col[mask] = np.nanmedian(col)
    return X


def published_label(rows):
    """The `is_stale` column as published (datetime.now() based)."""
    out = []
    for r in rows:
        v = r.get("is_stale", "")
        if v in ("True", "true", "1"):
            out.append(1)
        elif v in ("False", "false", "0"):
            out.append(0)
        else:
            out.append(-1)
    return np.array(out, dtype=np.int32)


def relabel(rows, records, t0_by_repo, delta):
    """Recompute is_stale at the pinned T0 for a given Delta.

    Uses the same integer-day convention as the original code:
        stale  <=>  (T0 - updatedAt).days > delta
    i.e. strictly more than `delta` whole days of silence.

    Returns (y, days_silent, age_days) as numpy arrays aligned with `rows`.
    """
    y = np.zeros(len(rows), dtype=np.int32)
    silent = np.full(len(rows), np.nan)
    age = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        key = (r["repo"], int(r["pr_number"]))
        rec = records.get(key)
        if rec is None:
            raise KeyError(f"row {i} {key} missing from raw JSON")
        t0 = t0_by_repo[r["repo"]]
        d_sil = (t0 - rec["updated_at"]).days
        d_age = (t0 - rec["created_at"]).days
        silent[i] = d_sil
        age[i] = d_age
        y[i] = 1 if d_sil > delta else 0
    return y, silent, age


# ── metrics ─────────────────────────────────────────────────────────────────

def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"
