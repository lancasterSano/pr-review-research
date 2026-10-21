#!/usr/bin/env python3
"""
ITIA-2026 theses (task F1): submission-time prediction of review DURATION and
transfer of all review-characteristic predictions across projects and
programming languages.

Author decisions this script implements (2026-10-14)
----------------------------------------------------
1. Corpus and filters exactly as HAIT: the published merged feature matrix
   (all_merged_features.csv, 64,820 rows) joined to the raw GraphQL nodes.
2. Duration thresholds are project-relative and leak-free. For a PR p of repo r,
       thr(p) = median{ t(q) : q in r, merged_at(q) < t0(p) }
   where t0(p) = created_at(p) and t is the same characteristic (t_m or t_fr).
   PRs whose history holds fewer than N_MIN_HISTORY values are not evaluated
   for that target. A fixed 24 h threshold is reported as sensitivity.
3. Characteristics are computed from HUMAN events only. A review event counts
   iff its author is not the PR author and participant_class(login) == "human".
   AI assistants, other bots, any `*[bot]` login and a short list of service
   accounts found by audit (EXTRA_AUTOMATION) are excluded. Assistant events are
   summarised separately as descriptive statistics.
   enrich/review_rounds.py is NOT modified (published numbers depend on it);
   review_iterations.count_iterations() is a parameterised copy of
   compute_review_rounds() and reproduces it exactly with every participant
   included and rule="code" (checked at run time on all 64,820 PRs, n_it_all).
   Iteration rule (author decision 14.10, evening): human iterations n_it_human
   and n_it_human_pre follow the LITERAL HAIT Definition 2: commits after the
   last counted review followed by the merge without a new review do NOT add an
   iteration. n_it_human_code keeps the earlier code rule for comparison.
4. Targets: t_m (time to merge) and human t_fr (time to first human review),
   both binary vs. the project median; human iterations n_it >= 2; staleness =
   Definition-3 trace label (Delta = 30 d) rebuilt from HUMAN events only:
   assistant and bot reviews / thread comments neither count as activity nor
   reset the inactivity timer (author decision 14.10). Commits carry no login
   and stay in the trace. The published T2 label (D-definition3-labels.csv) is
   reproduced exactly and reported alongside for comparison.

Feature set ("F1 features"): the 24 HAIT features, except that the two author
history features that depend on the OUTCOME of earlier PRs are recomputed with a
completion cut-off (only the author's PRs merged strictly before t0(p)):
    author_avg_review_rounds -> mean HUMAN n_it of those PRs
    author_commit_frequency  -> commits of those PRs created in the last 90 days
plus one new feature, author_avg_tfr_human (median human t_fr of those PRs).
The n_it and t_fr of earlier PRs used for author history and for the t_fr
threshold are computed from events up to THEIR merge (n_it_human_pre,
t_fr_human_pre): a review posted after q was merged may come after t0(p).
Imputation: column median over the evaluated rows, as week4_modeling.load_data.

Commit count (author decisions 14.10, --commits):
    t0min1 (default, ITIA primary)  num_commits = max(1, commits with authoredDate <= t0(p))
    t0     (earlier ITIA run)        num_commits = commits with authoredDate <= t0(p)
    final  (HAIT / first ITIA run)   num_commits = final commits.totalCount
A PR has at least one commit when it is opened, so a zero count by t0 means the
commits were rewritten after submission (force-push, squash with a new date):
that zero is information from the future and is floored to 1 in `t0min1`.
Outputs of `t0` / `final` carry the suffixes "_commits-t0" / "_commits-final";
the HAIT reproduction variants (hait_*) use the published matrix in all modes.

Model hyperparameters are the ones in predict/week4_modeling.py; only the
thread count is pinned (shared machine), which does not change results.

Stages (all by default):  --stage prep | models | report | all  [--commits t0|final]
Outputs: <DISSERTATION_DIR>/conf_papers/october_2026/ITIA2026/data/F-*.csv|json, numbers.json
The outputs stay outside this repository by design (see paths.py).
"""

import argparse
import bisect
import csv
import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np

warnings.filterwarnings("ignore")
csv.field_size_limit(10 ** 9)

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
from paths import DISSERTATION  # noqa: E402

from taskC_common import (  # noqa: E402
    REPO_DIRS, RAW_DIR, MERGED_CSV, classify_login, dir_to_repo,
)
from review_iterations import (  # noqa: E402,F401
    RULE_CODE, RULE_LITERAL, count_iterations, include_all, iteration_trace, login_of,
    parse_dt, reviewer_events,
)

OUT_DIR = os.path.join(DISSERTATION, "conf_papers", "october_2026", "ITIA2026", "data")
D_LABELS = os.path.join(DISSERTATION, "journal_articles", "HAIT_2026", "revision_results",
                        "D-definition3-labels.csv")
PR_TABLE = os.path.join(OUT_DIR, "F-pr-level.csv")

COMMITS_MODE = "t0min1"      # set by configure(); see module docstring
SUFFIX = ""
MODE_SUFFIX = {"t0min1": "", "t0": "_commits-t0", "final": "_commits-final"}
# Feature matrix (num_commits definition) of each mode, see Data.X_by_commits.
MODE_MATRIX = {"t0min1": "t0_floor1", "t0": "t0", "final": "final"}
# Sensitivity variants: the other commit-count definitions (names kept stable
# for the model cache) and the matrix each one uses.
VARIANT_MATRIX = {"commits_final": "final", "commits_at_t0": "t0",
                  "commits_t0_floor1": "t0_floor1"}
SENS_VARIANTS = {"t0min1": ["commits_final", "commits_at_t0"],
                 "t0": ["commits_final", "commits_t0_floor1"],
                 "final": ["commits_at_t0"]}


def configure(mode):
    """Select the commit-count definition and the matching output suffix."""
    global COMMITS_MODE, SUFFIX, METRICS_LONG, OOF_FILE
    if mode not in MODE_SUFFIX:
        raise ValueError(mode)
    COMMITS_MODE = mode
    SUFFIX = MODE_SUFFIX[mode]
    METRICS_LONG = out_path("F-metrics-long.jsonl")
    OOF_FILE = out_path("F-oof.npz")


def out_path(name):
    """OUT_DIR/name with the mode suffix inserted before the extension."""
    base, ext = os.path.splitext(name)
    return os.path.join(OUT_DIR, f"{base}{SUFFIX}{ext}")

# Primary language of each repository (GitHub "primary language"; nodejs/node
# is JavaScript there although it carries a large C++ share).
LANG = {
    "woocommerce/woocommerce": "PHP", "facebook/react": "JavaScript",
    "microsoft/vscode": "TypeScript", "vercel/next.js": "TypeScript",
    "angular/angular": "TypeScript", "supabase/supabase": "TypeScript",
    "strapi/strapi": "TypeScript", "grafana/grafana": "Go",
    "mattermost/mattermost": "Go", "nodejs/node": "JavaScript",
    "discourse/discourse": "Ruby", "django/django": "Python",
}
TS = "TypeScript"

N_MIN_HISTORY = 30
STALE_DELTA_DAYS = 30
AI_ERA_START = datetime(2025, 2, 1, tzinfo=timezone.utc)
N_THREADS = 6
CTRL_REPEATS = 15
PROJ_CTRL_REPEATS = 10
N_BOOT = 10000
SEED = 42

# Service accounts that the HAIT login audit left in the "human" class but that
# are evidently automation (found while building this script, 2026-10-14):
EXTRA_AUTOMATION = {
    # approves grafana-pr-automation translation PRs 0.1 min after creation, no body
    "grafana-pr-approver": "automated approver (grafana)",
    # "-bot" service account approving backport/changelog PRs, empty bodies
    "woocommercebot": "service account (woocommerce)",
    # CI credentials account
    "angular-builds": "CI service account (angular)",
}


# ── Participant classification ──────────────────────────────────────────────

def participant_class(login):
    """'human' | 'ai_assistant' | 'other_bot' | 'uncertain'.

    A missing login (deleted account, shown as "ghost" on GitHub) is treated as
    human: GitHub App accounts are not deleted this way.
    """
    if not login:
        return "human"
    k = login.lower()
    base = classify_login(login)
    if base != "human":
        return base
    if k.endswith("[bot]"):
        return "other_bot"
    if k in EXTRA_AUTOMATION:
        return "other_bot"
    return "human"


def is_human(login):
    return participant_class(login) == "human"


# ── Event-level computations (pure functions; unit-tested) ──────────────────
# parse_dt, reviewer_events and count_iterations live in review_iterations.py
# (one implementation for the dissertation); re-exported here for callers.

_login = login_of


def first_reaction_hours(pr, include):
    """Hours from creation to the first accepted non-author review, or None."""
    created = parse_dt(pr.get("createdAt"))
    ev = reviewer_events(pr, include)
    if not created or not ev:
        return None
    return (ev[0][0] - created).total_seconds() / 3600.0


def _dt_trace(s):
    """Same parsing as taskD_definition3_label.parse_dt (seconds precision)."""
    if not s or s in ("None", ""):
        return None
    try:
        return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def activity_trace(pr, include):
    """Definition-3 event trace (as taskD_definition3_label.build_trace), keeping
    review / review-thread events only from participants accepted by `include`.
    The PR author's own events are kept if `include` accepts the author.
    Commits carry no login in the collected data and are always kept."""
    opened = _dt_trace(pr.get("createdAt"))
    merged = _dt_trace(pr.get("mergedAt"))
    if not opened or not merged or merged < opened:
        return None
    events = [opened]
    for n in (pr.get("commits") or {}).get("nodes") or []:
        t = _dt_trace(((n or {}).get("commit") or {}).get("authoredDate"))
        if t and opened <= t <= merged:
            events.append(t)
    for n in (pr.get("reviews") or {}).get("nodes") or []:
        if not include(_login(n)):
            continue
        t = _dt_trace((n or {}).get("submittedAt"))
        if t and opened <= t <= merged:
            events.append(t)
    for th in (pr.get("reviewThreads") or {}).get("nodes") or []:
        for c in (((th or {}).get("comments") or {}).get("nodes") or []):
            if not include(_login(c)):
                continue
            t = _dt_trace((c or {}).get("createdAt"))
            if t and opened <= t <= merged:
                events.append(t)
    events.append(merged)
    events.sort()
    return events


def max_gap_days(events):
    if not events or len(events) < 2:
        return 0.0
    return max((events[i + 1] - events[i]).total_seconds() / 86400.0
               for i in range(len(events) - 1))


def truncate_until(pr, until):
    """Copy of `pr` keeping only reviews, review-thread comments and commits at
    or before `until` (used to compute n_it / t_fr "as known at merge")."""
    def keep(t):
        return t is not None and t <= until
    out = dict(pr)
    out["reviews"] = {"nodes": [r for r in (pr.get("reviews") or {}).get("nodes") or []
                                if keep(parse_dt((r or {}).get("submittedAt")))]}
    out["commits"] = {"nodes": [c for c in (pr.get("commits") or {}).get("nodes") or []
                                if keep(parse_dt(((c or {}).get("commit") or {}).get("authoredDate")))]}
    threads = []
    for th in (pr.get("reviewThreads") or {}).get("nodes") or []:
        cs = [c for c in (((th or {}).get("comments") or {}).get("nodes") or [])
              if keep(parse_dt((c or {}).get("createdAt")))]
        threads.append({"comments": {"nodes": cs}})
    out["reviewThreads"] = {"nodes": threads}
    return out


def assistant_summary(pr):
    """Descriptive assistant/bot participation for one PR (non-author events)."""
    created = parse_dt(pr.get("createdAt"))
    author = _login(pr)
    ev = reviewer_events(pr, include_all)
    classes = [participant_class(w) for _, w, _ in ev]
    first_class = classes[0] if classes else ""
    t_ai = next((t for (t, _, _), c in zip(ev, classes) if c == "ai_assistant"), None)
    t_auto = next((t for (t, _, _), c in zip(ev, classes) if c != "human"), None)
    ai_in_threads = False
    for th in (pr.get("reviewThreads") or {}).get("nodes") or []:
        for c in (((th or {}).get("comments") or {}).get("nodes") or []):
            w = _login(c)
            if w != author and participant_class(w) == "ai_assistant":
                ai_in_threads = True
    hrs = (lambda t: (t - created).total_seconds() / 3600.0 if (t and created) else None)
    return {
        "has_ai_review": int(any(c == "ai_assistant" for c in classes) or ai_in_threads),
        "n_ai_review_events": sum(c == "ai_assistant" for c in classes),
        "n_bot_review_events": sum(c in ("other_bot", "uncertain") for c in classes),
        "first_reactor_class": first_class,
        "t_first_ai_h": hrs(t_ai),
        "t_first_auto_h": hrs(t_auto),
    }


# ── Stage 1: per-PR table ───────────────────────────────────────────────────

PR_FIELDS = [
    "repo", "pr_number", "lang", "created_at", "merged_at", "is_bot_author",
    "t_m_h", "t_fr_any_h", "t_fr_human_h", "n_it_all", "n_it_human",
    "n_it_human_pre", "t_fr_human_pre", "n_it_human_code",
    "gap_all_days", "gap_human_days", "stale30_pub", "stale30_human",
    "has_ai_review", "n_ai_review_events", "n_bot_review_events",
    "first_reactor_class", "t_first_ai_h", "t_first_auto_h",
    "author_avg_review_rounds_h", "author_avg_tfr_human", "author_commit_frequency_s",
    "n_hist_tm", "thr_tm_h", "n_hist_tfr", "thr_tfr_h", "num_commits_t0",
]


def commits_at_submission(pr):
    """Commits authored no later than PR creation (sensitivity proxy for the
    commit count visible at t0; the published `num_commits` is the FINAL count)."""
    created = parse_dt(pr.get("createdAt"))
    n = 0
    for c in (pr.get("commits") or {}).get("nodes") or []:
        a = parse_dt(((c or {}).get("commit") or {}).get("authoredDate"))
        if a and created and a <= created:
            n += 1
    return n


def floor_commits(n_t0):
    """Commit-count feature of the primary mode: max(1, commits authored by t0).

    A PR cannot be opened without a commit, so 0 means the commits seen now were
    rewritten after submission; the floor removes that post-submission signal."""
    return np.maximum(1.0, np.asarray(n_t0, dtype=float))


def _read_published_rows():
    with open(MERGED_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _prefix_medians(values):
    """medians[k] = median(values[:k]) for k = 0..n (nan for k = 0)."""
    s = []
    out = [np.nan]
    for v in values:
        bisect.insort(s, v)
        n = len(s)
        out.append(s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]))
    return out


def stage_prep():
    t_start = time.time()
    rows = _read_published_rows()
    print(f"published merged matrix: {len(rows)} rows")
    want = {(r["repo"], int(r["pr_number"])) for r in rows}

    d_lab = {}
    with open(D_LABELS, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d_lab[(r["repo"], int(r["pr_number"]))] = (float(r["max_gap_days"]),
                                                       int(r[f"stale_delta_{STALE_DELTA_DAYS}"]))

    per_pr = {}
    for d in REPO_DIRS:
        repo = dir_to_repo(d)
        with open(os.path.join(RAW_DIR, d, "merged_prs.json")) as f:
            prs = json.load(f)
        for pr in prs:
            key = (repo, int(pr["number"]))
            if key not in want or key in per_pr:
                continue
            merged_at = parse_dt(pr.get("mergedAt"))
            pr_pre = truncate_until(pr, merged_at) if merged_at else pr
            tr_all = activity_trace(pr, include_all)
            tr_h = activity_trace(pr, lambda w: participant_class(w) == "human")
            rec = {
                "t_fr_any_h": first_reaction_hours(pr, include_all),
                "t_fr_human_h": first_reaction_hours(pr, is_human),
                # published HAIT algorithm (all events, code rule): reproduction check
                "n_it_all": count_iterations(pr, include_all, rule=RULE_CODE),
                "n_it_human": count_iterations(pr, is_human, rule=RULE_LITERAL),
                "n_it_human_code": count_iterations(pr, is_human, rule=RULE_CODE),
                # as known at merge: input to later PRs' author history and thresholds
                "n_it_human_pre": count_iterations(pr_pre, is_human, rule=RULE_LITERAL),
                "t_fr_human_pre": first_reaction_hours(pr_pre, is_human),
                "gap_all_days": max_gap_days(tr_all) if tr_all else np.nan,
                "gap_human_days": max_gap_days(tr_h) if tr_h else np.nan,
                "num_commits_t0": commits_at_submission(pr),
            }
            rec.update(assistant_summary(pr))
            per_pr[key] = rec
        print(f"  {repo}: raw {len(prs)}  ({time.time() - t_start:.0f}s)")
    missing = want - set(per_pr)
    if missing:
        raise SystemExit(f"{len(missing)} published rows not found in raw JSON")

    # Assemble with published columns.
    tab = []
    for r in rows:
        key = (r["repo"], int(r["pr_number"]))
        rec = dict(per_pr[key])
        gap_pub, st_pub = d_lab[key]
        rec.update({
            "repo": r["repo"], "pr_number": key[1], "lang": LANG[r["repo"]],
            "created_at": r["created_at"], "merged_at": r["merged_at"],
            "is_bot_author": int(r["is_bot"] in ("True", "true")),
            "t_m_h": float(r["time_to_merge_hours"]),
            "stale30_pub": st_pub,
            "stale30_human": int(rec["gap_human_days"] > STALE_DELTA_DAYS),
            "_gap_pub": gap_pub,
            "_rr_pub": int(float(r["review_rounds"])),
            "_tfr_pub": r["time_to_first_review_hours"],
            "_num_commits": float(r["num_commits"] or 0),
            "_author": r["author_login"],
        })
        tab.append(rec)

    # ── reproduction checks against the published columns ──
    rr_mismatch = sum(t["n_it_all"] != t["_rr_pub"] for t in tab)
    tfr_mismatch = 0
    for t in tab:
        pub = t["_tfr_pub"]
        mine = t["t_fr_any_h"]
        if pub in ("", "None"):
            tfr_mismatch += mine is not None
        elif mine is None or abs(round(mine, 2) - float(pub)) > 0.011:
            tfr_mismatch += 1
    gap_mismatch = sum(abs(t["gap_all_days"] - t["_gap_pub"]) > 1e-3 for t in tab)
    checks = {"n_rows": len(tab), "review_rounds_mismatch": rr_mismatch,
              "t_fr_any_mismatch": tfr_mismatch, "staleness_gap_mismatch": gap_mismatch,
              "human_review_after_merge_changes_n_it": sum(t["n_it_human"] != t["n_it_human_pre"] for t in tab),
              "iteration_rule": RULE_LITERAL,
              "n_it_human_literal_ne_code": sum(t["n_it_human"] != t["n_it_human_code"] for t in tab),
              "human_review_only_after_merge": sum(t["t_fr_human_h"] is not None and t["t_fr_human_pre"] is None
                                                   for t in tab),
              "commits_t0_zero": sum(t["num_commits_t0"] == 0 for t in tab)}
    print("reproduction checks:", checks)

    # ── strict author history (completion cut-off) and thresholds ──
    by_repo = {}
    for i, t in enumerate(tab):
        by_repo.setdefault(t["repo"], []).append(i)
    for repo, idx in by_repo.items():
        c = {i: parse_dt(tab[i]["created_at"]) for i in idx}
        m = {i: parse_dt(tab[i]["merged_at"]) for i in idx}
        order = sorted(idx, key=lambda i: m[i])
        m_sorted = [m[i] for i in order]
        med_tm = _prefix_medians([tab[i]["t_m_h"] for i in order])
        tfr_vals, tfr_pos = [], []   # human t_fr known at merge, in merge order (non-null only)
        for k, i in enumerate(order):
            v = tab[i]["t_fr_human_pre"]
            if v is not None:
                tfr_vals.append(v)
                tfr_pos.append(k)
        med_tfr = _prefix_medians(tfr_vals)
        for i in idx:
            k = bisect.bisect_left(m_sorted, c[i])          # merged strictly before t0
            j = bisect.bisect_left(tfr_pos, k)              # of them, with a human t_fr
            tab[i]["n_hist_tm"] = k
            tab[i]["thr_tm_h"] = med_tm[k]
            tab[i]["n_hist_tfr"] = j
            tab[i]["thr_tfr_h"] = med_tfr[j]

        by_author = {}
        for i in order:
            by_author.setdefault(tab[i]["_author"], []).append(i)
        for author, a_idx in by_author.items():
            a_m = [m[i] for i in a_idx]                    # already merge-ordered
            for i in a_idx:
                if not author:
                    prior = []
                else:
                    prior = a_idx[:bisect.bisect_left(a_m, c[i])]
                if prior:
                    # q merged before t0(p): its commit count is closed; n_it / t_fr as known at its merge
                    tab[i]["author_avg_review_rounds_h"] = float(np.mean([tab[q]["n_it_human_pre"] for q in prior]))
                    v = [tab[q]["t_fr_human_pre"] for q in prior if tab[q]["t_fr_human_pre"] is not None]
                    tab[i]["author_avg_tfr_human"] = float(np.median(v)) if v else np.nan
                    lo = c[i] - timedelta(days=90)
                    tab[i]["author_commit_frequency_s"] = float(sum(
                        tab[q]["_num_commits"] for q in prior if c[q] >= lo))
                else:
                    tab[i]["author_avg_review_rounds_h"] = np.nan
                    tab[i]["author_avg_tfr_human"] = np.nan
                    tab[i]["author_commit_frequency_s"] = 0.0 if author else np.nan

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(PR_TABLE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PR_FIELDS, extrasaction="ignore")
        w.writeheader()
        for t in tab:
            w.writerow({k: ("" if (t.get(k) is None or (isinstance(t.get(k), float) and np.isnan(t.get(k))))
                            else t.get(k)) for k in PR_FIELDS})
    with open(os.path.join(OUT_DIR, "F-prep-checks.json"), "w") as f:
        json.dump({**checks, "extra_automation": EXTRA_AUTOMATION,
                   "n_min_history": N_MIN_HISTORY}, f, indent=2)
    print(f"wrote {PR_TABLE} ({time.time() - t_start:.0f}s)")
    return checks


# ── Stage 2: models ─────────────────────────────────────────────────────────

METRICS_LONG = OOF_FILE = None
configure("t0min1")


def _w4():
    sys.path.insert(0, os.path.join(_SCRIPTS, "predict"))
    import week4_modeling as w4  # noqa: E402
    return w4


def f1_feature_cols(w4):
    cols = list(w4.FEATURE_COLS)
    cols[cols.index("author_avg_review_rounds")] = "author_avg_review_rounds_h"
    cols[cols.index("author_commit_frequency")] = "author_commit_frequency_s"
    return cols + ["author_avg_tfr_human"]


def make_model(name):
    from catboost import CatBoostClassifier, CatBoostRegressor
    import lightgbm as lgb
    if name == "CatBoost":
        return CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, verbose=0,
                                  random_seed=42, auto_class_weights="Balanced",
                                  thread_count=N_THREADS)
    if name == "CatBoost (size-only)":
        return CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=0,
                                  random_seed=42, auto_class_weights="Balanced",
                                  thread_count=N_THREADS)
    if name == "LightGBM":
        return lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                                  is_unbalance=True, verbose=-1, random_state=42,
                                  n_jobs=N_THREADS)
    if name == "CatBoostRegressor":
        return CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, verbose=0,
                                 random_seed=42, thread_count=N_THREADS)
    raise KeyError(name)


def impute_median(X):
    """Column-median imputation over the given rows (week4_modeling.load_data rule)."""
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        m = np.isnan(col)
        if m.any():
            col[m] = np.nanmedian(col) if (~m).any() else 0.0
    return X


class Data:
    """Everything the experiments need, aligned with the published row order."""

    def __init__(self):
        import pandas as pd
        w4 = _w4()
        self.w4 = w4
        t = pd.read_csv(PR_TABLE)
        self.t = t
        self.repo = t["repo"].to_numpy()
        self.lang = t["lang"].to_numpy()
        # Published 24-feature matrix, exactly as week4_modeling builds it.
        X_pub, y_pub, repos_pub = w4.load_data(MERGED_CSV, w4.FEATURE_COLS, w4.dp2_target)
        assert len(repos_pub) == len(t) and list(repos_pub) == list(self.repo)
        assert (y_pub == (t["n_it_all"].to_numpy() >= 2)).all()
        self.X_pub = X_pub
        # Raw (un-imputed) F1 matrix.
        pub_rows = _read_published_rows()
        self.f1_cols = f1_feature_cols(w4)
        X = np.empty((len(t), len(self.f1_cols)))
        for j, c in enumerate(self.f1_cols):
            if c in t.columns:
                X[:, j] = pd.to_numeric(t[c], errors="coerce").to_numpy()
            else:
                X[:, j] = [w4.to_float(r.get(c, "")) for r in pub_rows]
        # num_commits variants: final count (published) / authored by t0 / t0 floored at 1.
        j_nc = self.f1_cols.index("num_commits")
        self.n_commits_final = X[:, j_nc].copy()
        Xc = X.copy()
        Xc[:, j_nc] = t["num_commits_t0"].to_numpy(dtype=float)
        Xf = X.copy()
        Xf[:, j_nc] = floor_commits(t["num_commits_t0"].to_numpy(dtype=float))
        self.X_by_commits = {"final": X, "t0": Xc, "t0_floor1": Xf}
        # "f1" is the primary matrix of the selected mode; sensitivity variants
        # pick theirs as features="commits:<matrix>".
        self.X_f1_raw = self.X_by_commits[MODE_MATRIX[COMMITS_MODE]]
        self.size_idx = [self.f1_cols.index(c) for c in w4.SIZE_ONLY_COLS]
        self.size_idx_pub = [w4.FEATURE_COLS.index(c) for c in w4.SIZE_ONLY_COLS]

        tm_ok = (t["n_hist_tm"] >= N_MIN_HISTORY).to_numpy()
        tfr_ok = ((t["n_hist_tfr"] >= N_MIN_HISTORY) & t["t_fr_human_h"].notna()).to_numpy()
        self.masks = {"IT": np.ones(len(t), bool), "ST": np.ones(len(t), bool),
                      "TM": tm_ok, "TFR": tfr_ok}
        tm = t["t_m_h"].to_numpy()
        tfr = t["t_fr_human_h"].to_numpy()
        self.labels = {
            "IT": (t["n_it_human"].to_numpy() >= 2).astype(int),
            "IT_ALL": (t["n_it_all"].to_numpy() >= 2).astype(int),
            # Author decision (14.10): staleness from human events only; the
            # published T2 label is kept as ST_PUB for the comparison.
            "ST": t["stale30_human"].to_numpy().astype(int),
            "ST_PUB": t["stale30_pub"].to_numpy().astype(int),
            "TM": np.where(tm_ok, tm > t["thr_tm_h"].to_numpy(), 0).astype(int),
            "TFR": np.where(tfr_ok, np.nan_to_num(tfr) > t["thr_tfr_h"].fillna(0).to_numpy(), 0).astype(int),
            "TM24": (tm > 24).astype(int),
            "TFR24": (np.nan_to_num(tfr) > 24).astype(int),
        }
        created = pd.to_datetime(t["created_at"], utc=True)
        self.ai_era = (created >= pd.Timestamp(AI_ERA_START)).to_numpy()
        self.is_bot_author = t["is_bot_author"].to_numpy().astype(bool)


class Runner:
    def __init__(self, data, resume=True):
        self.d = data
        self.done = set()
        if resume and os.path.exists(METRICS_LONG):
            with open(METRICS_LONG) as f:
                for line in f:
                    r = json.loads(line)
                    self.done.add(self._key(r))
        self.oof = {}
        if resume and os.path.exists(OOF_FILE):
            z = np.load(OOF_FILE)
            self.oof = {k: z[k] for k in z.files}
        self.n_fit = 0
        self.t0 = time.time()

    @staticmethod
    def _key(r):
        return (r["variant"], r["target"], r["protocol"], r["model"], r["test"], r.get("rep", 0))

    def _log(self, rec):
        with open(METRICS_LONG, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.done.add(self._key(rec))

    def _fit_eval(self, model, X, y, tr, te, size_idx):
        Xtr, Xte = X[tr], X[te]
        if model == "CatBoost (size-only)":
            Xtr, Xte = Xtr[:, size_idx], Xte[:, size_idx]
        m = make_model(model)
        m.fit(Xtr, y[tr])
        prob = m.predict_proba(Xte)[:, 1]
        pred = m.predict(Xte).astype(int).ravel()
        self.n_fit += 1
        if self.n_fit % 25 == 0:
            print(f"    fits={self.n_fit}  elapsed={time.time() - self.t0:.0f}s", flush=True)
        return pred, prob

    def run(self, variant, target, label, model, protocols, features="f1", mask=None,
            keep_oof=False):
        d = self.d
        mask = d.masks.get(target, np.ones(len(d.t), bool)) if mask is None else mask
        rows = np.where(mask)[0]
        mats = {"f1": d.X_f1_raw, "pub": d.X_pub}
        mats.update({f"commits:{k}": v for k, v in d.X_by_commits.items()})
        X = impute_median(mats[features][rows])
        y = d.labels[label][rows]
        repo, lang = d.repo[rows], d.lang[rows]
        size_idx = d.size_idx_pub if features == "pub" else d.size_idx
        w4 = d.w4
        base = {"variant": variant, "target": target, "label": label, "model": model,
                "features": features}

        def record(protocol, test, tr, te, rep=0, extra=None):
            rec = dict(base, protocol=protocol, test=test, rep=rep)
            if self._key(rec) in self.done:
                return None
            pred, prob = self._fit_eval(model, X, y, tr, te, size_idx)
            mt = w4.evaluate(y[te], pred, prob)
            rec.update({k: float(v) for k, v in mt.items()})
            rec.update({"n_train": int(len(tr)), "n_test": int(len(te)),
                        "pos_test": float(y[te].mean()), "pos_train": float(y[tr].mean())})
            if extra:
                rec.update(extra)
            self._log(rec)
            return prob

        oof_key = f"{variant}|{target}|{model}|cv"
        if "cv" in protocols:
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            oof = np.full(len(rows), np.nan)
            for k, (tr, te) in enumerate(skf.split(X, y)):
                prob = record("cv", f"fold{k}", tr, te)
                if prob is not None:
                    oof[te] = prob
            if keep_oof and not np.isnan(oof).all():
                self.oof[oof_key] = oof
        repos = sorted(set(repo))
        if "loro" in protocols:
            loro_oof = np.full(len(rows), np.nan)
            for r in repos:
                te = np.where(repo == r)[0]
                tr = np.where(repo != r)[0]
                if len(np.unique(y[te])) < 2:
                    continue
                prob = record("loro", r, tr, te)
                if prob is not None:
                    loro_oof[te] = prob
            if keep_oof and not np.isnan(loro_oof).all():
                self.oof[f"{variant}|{target}|{model}|loro"] = loro_oof
        if "lolo" in protocols:
            for r in repos:
                te = np.where(repo == r)[0]
                tr = np.where(lang != LANG[r])[0]
                record("lolo", r, tr, te, extra={"lang": LANG[r]})
        non_ts = [r for r in repos if LANG[r] != TS]
        ts_rows = np.where(lang == TS)[0]
        if "ts" in protocols:
            for r in non_ts:
                record("ts", r, ts_rows, np.where(repo == r)[0])
        if "ctrl_rows" in protocols:
            n_ts = len(ts_rows)
            for r in non_ts:
                te = np.where(repo == r)[0]
                pool = np.where(repo != r)[0]
                for rep in range(CTRL_REPEATS):
                    rng = np.random.default_rng(SEED + 1000 * rep + repos.index(r))
                    tr = np.sort(rng.choice(pool, size=n_ts, replace=False))
                    record("ctrl_rows", r, tr, te, rep=rep)
        if "ctrl_proj" in protocols:
            n_ts = len(ts_rows)
            for r in non_ts:
                te = np.where(repo == r)[0]
                others = [q for q in repos if q != r]
                for rep in range(PROJ_CTRL_REPEATS):
                    rng = np.random.default_rng(SEED + 7000 + 1000 * rep + repos.index(r))
                    while True:
                        pick = list(rng.choice(others, size=5, replace=False))
                        if not all(LANG[q] == TS for q in pick):
                            break
                    pool = np.where(np.isin(repo, pick))[0]
                    tr = np.sort(rng.choice(pool, size=n_ts, replace=False)) if len(pool) > n_ts else pool
                    record("ctrl_proj", r, tr, te, rep=rep, extra={"pool_repos": "|".join(sorted(pick))})

    def save_oof(self):
        np.savez_compressed(OOF_FILE, **self.oof)


def run_regression(data):
    """LORO regression of log1p(t_m): Spearman and MAE vs. a train-median constant."""
    from scipy.stats import spearmanr
    out_file = out_path("F-regression-loro.csv")
    if os.path.exists(out_file):
        return
    d = data
    X = impute_median(d.X_f1_raw)
    y = np.log1p(d.t["t_m_h"].to_numpy())
    res = []
    for r in sorted(set(d.repo)):
        te, tr = d.repo == r, d.repo != r
        m = make_model("CatBoostRegressor")
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        const = np.median(y[tr])
        res.append({"repo": r, "spearman": float(spearmanr(p, y[te]).correlation),
                    "mae_model": float(np.mean(np.abs(p - y[te]))),
                    "mae_const": float(np.mean(np.abs(const - y[te])))})
        print(f"  regression {r}: {res[-1]}")
    with open(out_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)


def run_shap(data):
    out_file = out_path("F-shap.json")
    if os.path.exists(out_file):
        return
    import shap
    d = data
    out = {}
    for target in ["TM", "TFR", "IT", "ST"]:
        rows = np.where(d.masks[target])[0]
        X = impute_median(d.X_f1_raw[rows])
        y = d.labels[target][rows]
        m = make_model("CatBoost")
        m.fit(X, y)
        idx = np.random.RandomState(SEED).choice(len(X), min(4000, len(X)), replace=False)
        sv = shap.TreeExplainer(m).shap_values(X[idx])
        if isinstance(sv, list):
            sv = sv[1]
        imp = np.abs(sv).mean(axis=0)
        share = 100 * imp / imp.sum()
        order = np.argsort(-share)
        out[target] = [{"feature": d.f1_cols[k], "shap_share_pct": round(float(share[k]), 2)}
                       for k in order]
        print(f"  SHAP {target}: " + ", ".join(f"{e['feature']} {e['shap_share_pct']}" for e in out[target][:5]))
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)


def stage_models(resume=True):
    d = Data()
    run = Runner(d, resume=resume)
    print(f"commits mode: {COMMITS_MODE}; metrics -> {METRICS_LONG}")
    print(f"rows: {len(d.t)}; TM eligible {d.masks['TM'].sum()}, TFR eligible {d.masks['TFR'].sum()}")
    main_protocols = ["cv", "loro", "lolo", "ts", "ctrl_rows", "ctrl_proj"]
    # 1. Reproduction of HAIT (all events, published features) and decision-3 impact.
    run.run("hait_all", "IT", "IT_ALL", "CatBoost", ["cv", "loro"], features="pub", keep_oof=True)
    run.run("hait_human", "IT", "IT", "CatBoost", ["cv", "loro"], features="pub", keep_oof=True)
    run.run("hait_st", "ST", "ST_PUB", "CatBoost", ["cv", "loro"], features="pub")
    run.run("hait_st_human", "ST", "ST", "CatBoost", ["cv", "loro"], features="pub")
    run.save_oof()
    # 2. Main table: F1 features, human events, project-median thresholds.
    for target in ["TM", "TFR", "IT", "ST"]:
        print(f"== main {target}")
        run.run("main", target, target, "CatBoost", main_protocols)
        run.run("main", target, target, "CatBoost (size-only)", ["cv", "loro"])
        run.run("main", target, target, "LightGBM", ["cv", "loro"])
    # 3. Sensitivity and robustness.
    run.run("fixed24", "TM", "TM24", "CatBoost", ["cv", "loro"])
    run.run("fixed24", "TFR", "TFR24", "CatBoost", ["cv", "loro"])
    for target in ["TM", "TFR", "IT", "ST"]:
        run.run("no_bot_author", target, target, "CatBoost", ["loro"],
                mask=d.masks[target] & ~d.is_bot_author)
    for target in ["IT", "ST"]:
        run.run("tm_eligible_rows", target, target, "CatBoost", ["loro"], mask=d.masks["TM"])
    # The other commit-count definitions (variant names kept stable for the cache).
    for v in SENS_VARIANTS[COMMITS_MODE]:
        for target in ["IT", "TM", "TFR", "ST"]:
            run.run(v, target, target, "CatBoost", ["cv", "loro"],
                    features=f"commits:{VARIANT_MATRIX[v]}")
    run.save_oof()
    print("== regression")
    run_regression(d)
    print("== SHAP")
    run_shap(d)
    print(f"models done: {run.n_fit} new fits, {time.time() - run.t0:.0f}s")


# ── Stage 3: report ─────────────────────────────────────────────────────────

FEATURE_UA = {
    "lines_added": "кількість доданих рядків", "lines_deleted": "кількість вилучених рядків",
    "files_changed": "кількість змінених файлів", "num_commits": "кількість комітів",
    "has_tests": "наявність змін у тестах", "has_docs": "наявність змін у документації",
    "touches_ci": "зміни конфігурації CI", "desc_length": "довжина опису запиту",
    "title_length": "довжина назви запиту", "has_screenshots": "наявність знімків екрана",
    "has_code_blocks": "наявність блоків коду в описі", "references_issues": "посилання на задачі",
    "is_draft": "статус чернетки", "label_count": "кількість міток",
    "day_of_week": "день тижня подання", "hour_of_day": "година подання",
    "author_merge_rate": "частка злитих запитів автора",
    "author_avg_review_rounds_h": "середня кількість ітерацій попередніх запитів автора",
    "author_commit_frequency_s": "частота комітів автора",
    "author_tenure_days": "стаж автора в проєкті", "author_pr_count": "кількість попередніх запитів автора",
    "open_pr_count": "кількість відкритих запитів у проєкті",
    "merge_velocity_7d": "кількість злиттів у проєкті за 7 днів",
    "merge_velocity_30d": "кількість злиттів у проєкті за 30 днів",
    "author_avg_tfr_human": "медіана часу до першої рецензії людини на попередні запити автора",
}


def boot_ci(values, n=N_BOOT, seed=SEED):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def ua(x, nd=3):
    """Number -> string with a decimal comma (for the theses generator)."""
    if isinstance(x, str):
        return x
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


def stage_report():
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    t = pd.read_csv(PR_TABLE)
    t["created"] = pd.to_datetime(t["created_at"], utc=True)
    t["ai_era"] = t["created"] >= pd.Timestamp(AI_ERA_START)
    t["period"] = np.where(t["ai_era"], "from_2025-02", "before_2025-02")
    m = pd.DataFrame([json.loads(l) for l in open(METRICS_LONG)])
    repos = sorted(t["repo"].unique())
    non_ts = [r for r in repos if LANG[r] != TS]
    S = {}       # raw floats for F-summary.json
    rows_main = []

    def sel(variant, target, protocol, model="CatBoost"):
        q = m[(m.variant == variant) & (m.target == target) & (m.protocol == protocol) & (m.model == model)]
        return q

    def per_repo(q, metric="auc_roc"):
        return q.groupby("test")[metric].mean()

    masks = {"IT": np.ones(len(t), bool), "ST": np.ones(len(t), bool),
             "TM": (t.n_hist_tm >= N_MIN_HISTORY).to_numpy(),
             "TFR": ((t.n_hist_tfr >= N_MIN_HISTORY) & t.t_fr_human_h.notna()).to_numpy()}
    ylab = {"IT": (t.n_it_human >= 2).to_numpy(), "ST": t.stale30_human.to_numpy() == 1,
            "TM": (t.t_m_h > t.thr_tm_h).to_numpy(),
            "TFR": (t.t_fr_human_h > t.thr_tfr_h).to_numpy()}

    # ── main table ──
    for target in ["IT", "TM", "TFR", "ST"]:
        r = {"target": target, "n": int(masks[target].sum()),
             "share_pos_pct": float(100 * ylab[target][masks[target]].mean())}
        for model, tag in [("CatBoost", "cb"), ("LightGBM", "lgb"), ("CatBoost (size-only)", "size")]:
            cv = sel("main", target, "cv", model)
            lo = per_repo(sel("main", target, "loro", model))
            for metric in ["auc_roc", "auc_pr", "f1"]:
                r[f"{tag}_cv_{metric}"] = float(cv[metric].mean())
                r[f"{tag}_loro_{metric}"] = float(per_repo(sel("main", target, "loro", model), metric).mean())
            r[f"{tag}_cv_auc_sd"] = float(cv["auc_roc"].std(ddof=1))
            r[f"{tag}_loro_auc_min"] = float(lo.min())
            r[f"{tag}_loro_auc_max"] = float(lo.max())
            r[f"{tag}_loro_auc_median"] = float(lo.median())
            r[f"{tag}_loro_auc_ci_lo"], r[f"{tag}_loro_auc_ci_hi"] = boot_ci(lo.values)
            r[f"{tag}_loro_f1_median"] = float(per_repo(sel("main", target, "loro", model), "f1").median())
        lo = per_repo(sel("main", target, "loro"))
        lolo = per_repo(sel("main", target, "lolo"))
        ts = per_repo(sel("main", target, "ts"))
        cr = per_repo(sel("main", target, "ctrl_rows"))
        cp = per_repo(sel("main", target, "ctrl_proj"))
        r["lolo_auc"] = float(lolo.mean())
        r["lolo_auc_ci_lo"], r["lolo_auc_ci_hi"] = boot_ci(lolo.values)
        r["lolo_minus_loro"] = float((lolo - lo).mean())
        r["lolo_minus_loro_max_abs_repo"] = float((lolo - lo).abs().max())
        r["loro_auc_nonts7"] = float(lo[non_ts].mean())
        r["ts_auc"] = float(ts[non_ts].mean())
        r["ctrl_rows_auc"] = float(cr[non_ts].mean())
        r["ctrl_proj_auc"] = float(cp[non_ts].mean())
        d_rows = (ts - cr)[non_ts]
        d_proj = (ts - cp)[non_ts]
        r["ts_minus_ctrl_rows"] = float(d_rows.mean())
        r["ts_minus_ctrl_rows_ci_lo"], r["ts_minus_ctrl_rows_ci_hi"] = boot_ci(d_rows.values)
        r["ts_minus_ctrl_proj"] = float(d_proj.mean())
        r["ts_minus_ctrl_proj_ci_lo"], r["ts_minus_ctrl_proj_ci_hi"] = boot_ci(d_proj.values)
        r["ts_minus_loro_nonts"] = float((ts - lo)[non_ts].mean())
        q = sel("main", target, "ctrl_rows")
        r["ctrl_rows_rep_sd_mean"] = float(q.groupby("test")["auc_roc"].std(ddof=1).mean())
        r["n_ts_pool"] = int(sel("main", target, "ts")["n_train"].iloc[0])
        r["ctrl_rows_n_reps"] = int(q.groupby("test").size().min())
        r["ctrl_proj_n_reps"] = int(sel("main", target, "ctrl_proj").groupby("test").size().min())
        rows_main.append(r)
        # per-repo table for this target
        pr = pd.DataFrame({"loro": lo, "lolo": lolo, "ts": ts, "ctrl_rows": cr, "ctrl_proj": cp,
                           "loro_size_only": per_repo(sel("main", target, "loro", "CatBoost (size-only)")),
                           "loro_lgb": per_repo(sel("main", target, "loro", "LightGBM")),
                           "loro_f1": per_repo(sel("main", target, "loro"), "f1"),
                           "loro_auc_pr": per_repo(sel("main", target, "loro"), "auc_pr"),
                           "pos_test": per_repo(sel("main", target, "loro"), "pos_test")})
        pr.insert(0, "lang", [LANG[x] for x in pr.index])
        pr.index.name = "repo"
        pr.to_csv(out_path(f"F-per-repo-{target}.csv"))
    main = pd.DataFrame(rows_main)
    main.to_csv(out_path("F-main-table.csv"), index=False)
    S["main"] = rows_main

    # ── decision 3: impact on published numbers ──
    oof = np.load(OOF_FILE)
    imp = {}
    for v, lab in [("hait_all", "IT_ALL"), ("hait_human", "IT")]:
        cv = sel(v, "IT", "cv")
        lo = per_repo(sel(v, "IT", "loro"))
        y = (t.n_it_all >= 2).to_numpy() if lab == "IT_ALL" else (t.n_it_human >= 2).to_numpy()
        e = {"cv_auc": float(cv.auc_roc.mean()), "loro_auc": float(lo.mean()),
             "loro_auc_min": float(lo.min()), "loro_auc_max": float(lo.max()),
             "loro_f1_median": float(per_repo(sel(v, "IT", "loro"), "f1").median()),
             "share_pct": float(100 * y.mean())}
        o_cv, o_lo = oof[f"{v}|IT|CatBoost|cv"], oof[f"{v}|IT|CatBoost|loro"]
        for per in ["before_2025-02", "from_2025-02"]:
            pm = (t.period == per).to_numpy()
            e[f"share_pct_{per}"] = float(100 * y[pm].mean())
            e[f"cv_pooled_oof_auc_{per}"] = float(roc_auc_score(y[pm], o_cv[pm]))
            aucs = []
            for rp in repos:
                mm = pm & (t.repo == rp).to_numpy()
                if mm.sum() >= 30 and len(np.unique(y[mm])) == 2:
                    aucs.append(roc_auc_score(y[mm], o_lo[mm]))
            e[f"loro_auc_{per}"] = float(np.mean(aucs))
            e[f"loro_n_repos_{per}"] = len(aucs)
        imp[v] = e
    for v in ["hait_st", "hait_st_human"]:
        lo = per_repo(sel(v, "ST", "loro"))
        imp[v] = {"cv_auc": float(sel(v, "ST", "cv").auc_roc.mean()), "loro_auc": float(lo.mean()),
                  "loro_auc_min": float(lo.min()), "loro_auc_max": float(lo.max()),
                  "loro_auc_per_repo": {k: float(x) for k, x in lo.items()}}
    by = []
    for scope, g in [("pool", t)] + [(rp, t[t.repo == rp]) for rp in repos]:
        for per, gg in [("all", g), ("before_2025-02", g[g.period == "before_2025-02"]),
                        ("from_2025-02", g[g.period == "from_2025-02"])]:
            if len(gg) == 0:
                continue
            by.append({
                "scope": scope, "period": per, "n": len(gg),
                "it_share_all_pct": 100 * (gg.n_it_all >= 2).mean(),
                "it_share_human_pct": 100 * (gg.n_it_human >= 2).mean(),
                "it_share_human_code_rule_pct": 100 * (gg.n_it_human_code >= 2).mean(),
                "it_label_changed": int(((gg.n_it_all >= 2) != (gg.n_it_human >= 2)).sum()),
                "tfr_median_any_h": gg.t_fr_any_h.median(),
                "tfr_median_human_h": gg.t_fr_human_h.median(),
                "no_human_review": int(gg.t_fr_human_h.isna().sum()),
                "st_share_pub_pct": 100 * gg.stale30_pub.mean(),
                "st_share_human_pct": 100 * gg.stale30_human.mean(),
                "st_label_changed": int((gg.stale30_pub != gg.stale30_human).sum()),
                "share_with_ai_review_pct": 100 * gg.has_ai_review.mean(),
                "share_ai_first_pct": 100 * (gg.first_reactor_class == "ai_assistant").mean(),
                "share_nonhuman_first_pct": 100 * gg.first_reactor_class.isin(["ai_assistant", "other_bot", "uncertain"]).mean(),
                "median_t_first_ai_h": gg.t_first_ai_h.median(),
                "median_t_first_auto_h": gg.t_first_auto_h.median(),
                "share_tm_gt24_pct": 100 * (gg.t_m_h > 24).mean(),
                "median_tm_h": gg.t_m_h.median(),
            })
    byd = pd.DataFrame(by)
    byd.to_csv(out_path("F-decision3-and-assistants.csv"), index=False)
    gap_shift = (t.gap_human_days - t.gap_all_days)
    imp["staleness_timer"] = {
        "prs_where_gap_grows": int((gap_shift > 1e-9).sum()),
        "prs_where_gap_grows_pct": float(100 * (gap_shift > 1e-9).mean()),
        "median_gap_growth_days_when_grows": float(gap_shift[gap_shift > 1e-9].median()),
        "labels_changed_delta30": int((t.stale30_pub != t.stale30_human).sum()),
        "share_pub_pct": float(100 * t.stale30_pub.mean()),
        "share_human_pct": float(100 * t.stale30_human.mean()),
    }
    with open(out_path("F-decision3-impact.json"), "w") as f:
        json.dump(imp, f, indent=2)
    S["decision3"] = imp

    # ── exclusions ──
    excl = {"n_rows": int(len(t)), "n_min_history": N_MIN_HISTORY,
            "no_human_review": int(t.t_fr_human_h.isna().sum()),
            "tm_excluded_min_history": int((t.n_hist_tm < N_MIN_HISTORY).sum()),
            "tfr_excluded_min_history": int(((t.n_hist_tfr < N_MIN_HISTORY) & t.t_fr_human_h.notna()).sum()),
            "tm_eligible": int(masks["TM"].sum()), "tfr_eligible": int(masks["TFR"].sum()),
            "tm_excluded_per_repo": {rp: int(((t.repo == rp) & (t.n_hist_tm < N_MIN_HISTORY)).sum()) for rp in repos},
            "thr_tm_median_per_repo_h": {rp: float(t[(t.repo == rp) & masks["TM"]].thr_tm_h.median()) for rp in repos},
            "thr_tfr_median_per_repo_h": {rp: float(t[(t.repo == rp) & masks["TFR"]].thr_tfr_h.median()) for rp in repos}}
    S["exclusions"] = excl

    # ── sensitivity / robustness ──
    sens = {}
    for target, lab in [("TM", "TM24"), ("TFR", "TFR24")]:
        lo = per_repo(sel("fixed24", target, "loro"))
        sens[f"fixed24_{target}"] = {
            "share_pct": float(100 * sel("fixed24", target, "loro").pipe(
                lambda q: (q.pos_test * q.n_test).sum() / q.n_test.sum())),
            "cv_auc": float(sel("fixed24", target, "cv").auc_roc.mean()),
            "loro_auc": float(lo.mean()), "loro_auc_min": float(lo.min()), "loro_auc_max": float(lo.max()),
            "loro_f1_mean": float(per_repo(sel("fixed24", target, "loro"), "f1").mean()),
            "loro_f1_min": float(per_repo(sel("fixed24", target, "loro"), "f1").min()),
            "loro_f1_max": float(per_repo(sel("fixed24", target, "loro"), "f1").max())}
    for target in ["IT", "TM", "TFR", "ST"]:
        sens[f"no_bot_author_{target}_loro_auc"] = float(per_repo(sel("no_bot_author", target, "loro")).mean())
    sens_variants = SENS_VARIANTS[COMMITS_MODE]
    for v in sens_variants:
        for target in ["IT", "TM", "TFR", "ST"]:
            sens[f"{v}_{target}_cv_auc"] = float(sel(v, target, "cv").auc_roc.mean())
            sens[f"{v}_{target}_loro_auc"] = float(per_repo(sel(v, target, "loro")).mean())
    pub = pd.read_csv(MERGED_CSV, usecols=["num_commits"])
    nc_final = pub["num_commits"].fillna(0).to_numpy()
    nc_t0 = t["num_commits_t0"].to_numpy()
    sens["commits_profile"] = {
        "share_final_gt_t0_pct": float(100 * (nc_final > nc_t0).mean()),
        "share_t0_zero_pct": float(100 * (nc_t0 == 0).mean()),
        "median_final": float(np.median(nc_final)), "median_t0": float(np.median(nc_t0))}
    for target in ["IT", "ST"]:
        sens[f"tm_eligible_rows_{target}_loro_auc"] = float(per_repo(sel("tm_eligible_rows", target, "loro")).mean())
    reg = pd.read_csv(out_path("F-regression-loro.csv"))
    sens["regression_log_tm"] = {
        "spearman_mean": float(reg.spearman.mean()), "spearman_min": float(reg.spearman.min()),
        "spearman_max": float(reg.spearman.max()),
        "mae_model_mean": float(reg.mae_model.mean()), "mae_const_mean": float(reg.mae_const.mean()),
        "factor_model": float(np.exp(reg.mae_model.mean())), "factor_const": float(np.exp(reg.mae_const.mean()))}
    S["commits_mode"] = COMMITS_MODE
    S["sensitivity"] = sens
    shap_res = json.load(open(out_path("F-shap.json")))
    S["shap_top5"] = {k: v[:5] for k, v in shap_res.items()}
    with open(out_path("F-summary.json"), "w") as f:
        json.dump(S, f, indent=2, ensure_ascii=False)

    # ── numbers.json for the theses generator (placeholders of draft_ua.md) ──
    mt = {r["target"]: r for r in rows_main}
    pool = byd[(byd.scope == "pool") & (byd.period == "all")].iloc[0]
    rep_all = lambda rp: byd[(byd.scope == rp) & (byd.period == "all")].iloc[0]  # noqa: E731
    N = {}
    for tg in ["IT", "TM", "TFR", "ST"]:
        r = mt[tg]
        N[f"SHARE_{tg}"] = ua(r["share_pos_pct"], 1)
        N[f"AUC_{tg}_CV"] = ua(r["cb_cv_auc_roc"])
        N[f"AUC_{tg}_LORO"] = ua(r["cb_loro_auc_roc"])
        N[f"AUC_{tg}_LORO_MIN"] = ua(r["cb_loro_auc_min"])
        N[f"AUC_{tg}_LORO_MAX"] = ua(r["cb_loro_auc_max"])
        N[f"AUC_{tg}_LOLO"] = ua(r["lolo_auc"])
        N[f"AUC_{tg}_TS"] = ua(r["ts_auc"])
        N[f"AUC_{tg}_CTRL"] = ua(r["ctrl_rows_auc"])
        # extra (not in the draft)
        N[f"AUC_{tg}_LORO_CI"] = f"[{ua(r['cb_loro_auc_ci_lo'])}; {ua(r['cb_loro_auc_ci_hi'])}]"
        N[f"AUC_{tg}_LORO_NONTS7"] = ua(r["loro_auc_nonts7"])
        N[f"AUC_{tg}_CTRL_PROJ"] = ua(r["ctrl_proj_auc"])
        N[f"DELTA_{tg}_TS_CTRL"] = ua(r["ts_minus_ctrl_rows"])
        N[f"CI_{tg}_TS_CTRL"] = f"[{ua(r['ts_minus_ctrl_rows_ci_lo'])}; {ua(r['ts_minus_ctrl_rows_ci_hi'])}]"
        N[f"DELTA_{tg}_TS_CTRL_PROJ"] = ua(r["ts_minus_ctrl_proj"])
        N[f"CI_{tg}_TS_CTRL_PROJ"] = f"[{ua(r['ts_minus_ctrl_proj_ci_lo'])}; {ua(r['ts_minus_ctrl_proj_ci_hi'])}]"
        N[f"DELTA_{tg}_LOLO_LORO"] = ua(r["lolo_minus_loro"])
        N[f"AUCPR_{tg}_LORO"] = ua(r["cb_loro_auc_pr"])
        N[f"F1_{tg}_LORO"] = ua(r["cb_loro_f1"])
        N[f"F1_{tg}_LORO_MEDIAN"] = ua(r["cb_loro_f1_median"])
        N[f"AUC_{tg}_LGB_CV"] = ua(r["lgb_cv_auc_roc"])
        N[f"AUC_{tg}_LGB_LORO"] = ua(r["lgb_loro_auc_roc"])
        N[f"AUC_{tg}_SIZEONLY_CV"] = ua(r["size_cv_auc_roc"])
        N[f"AUC_{tg}_SIZEONLY"] = ua(r["size_loro_auc_roc"])
        N[f"N_TS_POOL_{tg}"] = str(r["n_ts_pool"])
    N["MAX_DELTA_LOLO"] = ua(max(abs(mt[tg]["lolo_minus_loro"]) for tg in mt))
    N["MAX_DELTA_LOLO_REPO"] = ua(max(mt[tg]["lolo_minus_loro_max_abs_repo"] for tg in mt))
    N["MAX_DELTA_TS_CTRL_IT_TM_TFR"] = ua(max(abs(mt[tg]["ts_minus_ctrl_rows"]) for tg in ["IT", "TM", "TFR"]))
    # Paired LORO differences between characteristics (same 12 test repos).
    paired = {}
    for a_, b_ in [("IT", "TM"), ("IT", "TFR"), ("TM", "TFR"), ("TM", "ST"), ("TFR", "ST")]:
        da = per_repo(sel("main", a_, "loro")) - per_repo(sel("main", b_, "loro"))
        lo_, hi_ = boot_ci(da.values)
        paired[f"{a_}-{b_}"] = {"mean": float(da.mean()), "ci_lo": lo_, "ci_hi": hi_,
                                "n_repos_pos": int((da > 0).sum()), "n_repos": int(da.notna().sum())}
        N[f"PAIR_{a_}_{b_}"] = ua(float(da.mean()))
        N[f"CI_PAIR_{a_}_{b_}"] = f"[{ua(lo_)}; {ua(hi_)}]"
        N[f"NPOS_PAIR_{a_}_{b_}"] = str(int((da > 0).sum()))
    S["paired_loro"] = paired
    with open(out_path("F-summary.json"), "w") as f:
        json.dump(S, f, indent=2, ensure_ascii=False)
    N["SHARE_AI_FIRST"] = ua(pool.share_ai_first_pct, 1)
    N["SHARE_AI_FIRST_VSCODE"] = ua(rep_all("microsoft/vscode").share_ai_first_pct, 1)
    N["SHARE_WITH_AI_REVIEW"] = ua(pool.share_with_ai_review_pct, 1)
    N["MEDIAN_T_FIRST_AI_H"] = ua(pool.median_t_first_ai_h, 2)
    N["SHARE_TM24_NODE"] = ua(rep_all("nodejs/node").share_tm_gt24_pct, 1)
    N["SHARE_TM24_VSCODE"] = ua(rep_all("microsoft/vscode").share_tm_gt24_pct, 1)
    N["N_NO_HUMAN"] = str(excl["no_human_review"])
    N["N_MIN_HIST"] = str(N_MIN_HISTORY)
    N["N_TS_POOL"] = f'{mt["IT"]["n_ts_pool"]:,}'.replace(",", " ")  # 25 971
    N["N_CTRL_REPEATS"] = str(CTRL_REPEATS)
    N["N_CTRL_PROJ_REPEATS"] = str(PROJ_CTRL_REPEATS)
    N["N_EXCL_TM"] = str(excl["tm_excluded_min_history"])
    N["N_EXCL_TFR"] = str(excl["tfr_excluded_min_history"])
    N["N_TM_ELIGIBLE"] = str(excl["tm_eligible"])
    N["N_TFR_ELIGIBLE"] = str(excl["tfr_eligible"])
    N["MEDIAN_TFR_ANY_H"] = ua(pool.tfr_median_any_h, 2)
    N["MEDIAN_TFR_HUMAN_H"] = ua(pool.tfr_median_human_h, 2)
    N["MEDIAN_TM_H"] = ua(pool.median_tm_h, 2)
    for tg in ["TM", "TFR"]:
        top = shap_res[tg][0]
        name = FEATURE_UA.get(top["feature"], top["feature"])
        if top["feature"] == "num_commits" and COMMITS_MODE != "final":
            name += " на момент подання"
        N[f"SHAP_{tg}_TOP1"] = name
        N[f"SHAP_{tg}_TOP1_PCT"] = ua(top["shap_share_pct"], 1)
        N[f"SHAP_{tg}_TOP1_CODE"] = top["feature"]
        for i, e in enumerate(shap_res[tg][:5], 1):
            N[f"SHAP_{tg}_TOP{i}_CODE"] = e["feature"]
            N[f"SHAP_{tg}_TOP{i}_PCT"] = ua(e["shap_share_pct"], 1)
    d3 = imp
    N["HAIT_IT_ALL_SHARE"] = ua(d3["hait_all"]["share_pct"], 1)
    N["HAIT_IT_ALL_CV"] = ua(d3["hait_all"]["cv_auc"])
    N["HAIT_IT_ALL_LORO"] = ua(d3["hait_all"]["loro_auc"])
    N["HAIT_IT_HUMAN_CV"] = ua(d3["hait_human"]["cv_auc"])
    N["HAIT_IT_HUMAN_LORO"] = ua(d3["hait_human"]["loro_auc"])
    N["HAIT_ST_CV"] = ua(d3["hait_st"]["cv_auc"])
    N["HAIT_ST_LORO"] = ua(d3["hait_st"]["loro_auc"])
    N["HAIT_ST_HUMAN_CV"] = ua(d3["hait_st_human"]["cv_auc"])
    N["HAIT_ST_HUMAN_LORO"] = ua(d3["hait_st_human"]["loro_auc"])
    N["SHARE_ST_PUB"] = ua(d3["staleness_timer"]["share_pub_pct"], 2)
    N["SHARE_ST_HUMAN"] = ua(d3["staleness_timer"]["share_human_pct"], 2)
    N["SHARE_IT_VSCODE_ALL"] = ua(rep_all("microsoft/vscode").it_share_all_pct, 1)
    N["SHARE_IT_VSCODE_HUMAN"] = ua(rep_all("microsoft/vscode").it_share_human_pct, 1)
    N["N_IT_LABEL_CHANGED"] = str(int(pool.it_label_changed))
    N["N_ST_LABEL_CHANGED"] = str(d3["staleness_timer"]["labels_changed_delta30"])
    for k in ["TM", "TFR"]:
        s = sens[f"fixed24_{k}"]
        N[f"FIXED24_{k}_SHARE"] = ua(s["share_pct"], 1)
        N[f"FIXED24_{k}_CV"] = ua(s["cv_auc"])
        N[f"FIXED24_{k}_LORO"] = ua(s["loro_auc"])
    rg = sens["regression_log_tm"]
    N["REG_TM_SPEARMAN"] = ua(rg["spearman_mean"], 2)
    N["REG_TM_FACTOR_MODEL"] = ua(rg["factor_model"], 1)
    N["REG_TM_FACTOR_CONST"] = ua(rg["factor_const"], 1)
    for tg in ["IT", "TM", "TFR", "ST"]:
        N[f"NOBOT_{tg}_LORO"] = ua(sens[f"no_bot_author_{tg}_loro_auc"])
    key_of = {"commits_at_t0": "COMMITS_T0", "commits_final": "COMMITS_FINAL",
              "commits_t0_floor1": "COMMITS_T0_FLOOR1"}
    for v in sens_variants:
        for tg in ["IT", "TM", "TFR", "ST"]:
            N[f"{key_of[v]}_{tg}_CV"] = ua(sens[f"{v}_{tg}_cv_auc"])
            N[f"{key_of[v]}_{tg}_LORO"] = ua(sens[f"{v}_{tg}_loro_auc"])
    cp = sens["commits_profile"]
    N["SHARE_COMMITS_FINAL_GT_T0"] = ua(cp["share_final_gt_t0_pct"], 1)
    N["SHARE_COMMITS_T0_ZERO"] = ua(cp["share_t0_zero_pct"], 1)
    N["COMMITS_MODE"] = COMMITS_MODE
    with open(out_path("numbers.json"), "w", encoding="utf-8") as f:
        json.dump(N, f, indent=2, ensure_ascii=False)
    print(main[["target", "n", "share_pos_pct", "cb_cv_auc_roc", "cb_loro_auc_roc", "cb_loro_auc_min",
                "cb_loro_auc_max", "lolo_auc", "loro_auc_nonts7", "ts_auc", "ctrl_rows_auc",
                "ctrl_proj_auc", "size_loro_auc_roc"]].round(3).to_string(index=False))
    print(json.dumps(imp, indent=1))
    print(f"numbers.json: {len(N)} keys")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "models", "report", "all"], default="all")
    ap.add_argument("--fresh", action="store_true", help="ignore cached model results")
    ap.add_argument("--commits", choices=list(MODE_SUFFIX), default="t0min1",
                    help="num_commits definition: max(1, authored by t0) (default), "
                         "authored by t0, or final count (HAIT)")
    args = ap.parse_args()
    configure(args.commits)
    if args.stage in ("prep", "all"):
        stage_prep()
    if args.stage in ("models", "all"):
        stage_models(resume=not args.fresh)
    if args.stage in ("report", "all"):
        stage_report()
