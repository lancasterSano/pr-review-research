#!/usr/bin/env python3
"""Chapter 3 on corpus v2: per-PR table (labels, features at t0, reverts).

Decisions implemented (dissertation, chapters 2 and 3):
  * prediction moment t0 = PR creation; every feature uses only data known at t0;
  * clocks of t_r, t_m and of the staleness timer start at t_rd, the first exit
    from draft for PRs created as drafts, otherwise t_rd = t0 (clock "D");
    clock "A" (all from creation) is kept as a sensitivity column set;
  * characteristics from HUMAN events only (taskF1.participant_class with the
    corpus v2 account type); conversation comments (IssueComment) are not
    activity;
  * iterations: literal HAIT rule (review_iterations.RULE_LITERAL);
  * human reviews of a draft (before t_rd) are not events of t_r (clock D) or of
    the iterations (decision of 02.11.2026); clock A keeps them;
  * main population: authors that are people or LLM coding agents
    (taskC_common.in_main_population); bot-authored PRs are kept in the table
    with in_main = 0 for the sensitivity check.

Data are read only. Paths come from the environment (see r3_paths()).

    python revision/r3_v2_prep.py --corpus v2 [--repos owner/name,...]
"""
import argparse
import bisect
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from review_iterations import RULE_LITERAL, count_iterations, login_of, parse_dt  # noqa: E402
from taskC_common import author_group, in_main_population  # noqa: E402
from taskF1_duration_language import participant_class  # noqa: E402

csv.field_size_limit(10 ** 9)

WINDOW_END = datetime(2026, 8, 31, 23, 59, 59, tzinfo=timezone.utc)
REVERT_DAYS = 30
STALE_DELTA_DAYS = 30
N_MIN_HISTORY = 30
ALLOWED_BASES = ("main", "master", "trunk", "develop", "release", "canary", "dev")
# Data fix of 01.11.2026: when the name
# filter keeps under this share of a repository's merged PRs, its main line is the
# most frequent base branch of the merged PRs (version branches: mautic 7.x, matomo 5.x-dev).
MIN_NAMED_BASE_SHARE = 0.01


def r3_paths():
    """Input and output locations; every one can be overridden from the environment."""
    return {
        "raw_v2": os.environ.get("R3_RAW_V2") or os.path.join(ROOT, "data", "raw_v2"),
        "raw_v1": os.environ.get("R3_RAW_V1") or os.path.join(ROOT, "data", "raw"),
        "processed": os.environ.get("R3_PROCESSED") or os.path.join(ROOT, "data", "processed"),
        "out": os.environ.get("R3_OUT") or os.path.join(ROOT, "revision", "output", "r3_v2"),
    }


# ── Features: explicit allow-list at t0 ─────────────────────────────────────
# Everything a model may see. Leak test: tests/test_r3_v2.py.
FEATURES_T0 = [
    # size of the change as pushed by t0 (commits authored no later than t0)
    "lines_added_t0", "lines_deleted_t0", "commit_files_t0", "num_commits_t0",
    # description (final text: GitHub keeps no edit history; limitation)
    "desc_length", "title_length", "has_screenshots", "has_code_blocks", "references_issues",
    # state and time of submission
    "is_draft_t0", "day_of_week", "hour_of_day",
    # author history: PRs of the author in this repository closed before t0
    "author_pr_count", "author_tenure_days", "author_merge_rate",
    "author_avg_review_rounds_h", "author_avg_tfr_human", "author_commit_frequency_s",
    # project context at t0
    "open_pr_count_t0", "merge_velocity_7d_t0", "merge_velocity_30d_t0",
]
TEXT_FEATURES = ["desc_length", "title_length", "has_screenshots", "has_code_blocks",
                 "references_issues"]
SIZE_T0_FEATURES = ["lines_added_t0", "lines_deleted_t0", "commit_files_t0"]
# Never allowed (final state, known only after t0): decision 29.10.2026.
FORBIDDEN = {"lines_added", "lines_deleted", "files_changed", "num_commits", "has_tests",
             "has_docs", "touches_ci", "label_count", "num_file_types", "is_draft",
             "author_avg_review_rounds", "author_commit_frequency", "merge_velocity_7d",
             "merge_velocity_30d", "open_pr_count", "review_count", "review_rounds"}

IMAGE_PATTERN = re.compile(r"!\[.*?\]\(.*?\)|<img\s", re.IGNORECASE)
CODE_BLOCK_PATTERN = re.compile(r"```")
ISSUE_REF_PATTERN = re.compile(r"(?:closes?|fixes?|resolves?)\s+#\d+", re.IGNORECASE)


def _dt_trace(s):
    """Seconds precision, as taskD_definition3_label / taskF1.activity_trace."""
    if not s:
        return None
    try:
        return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _hours(a, b):
    return (b - a).total_seconds() / 3600.0 if (a and b) else None


def typename_map(pr):
    """login -> GitHub account type over every node of the PR (corpus v2)."""
    out = {}

    def add(node):
        a = (node or {}).get("author") or (node or {}).get("actor") or {}
        if a.get("login") and a.get("__typename"):
            out.setdefault(a["login"], a["__typename"])
    add(pr)
    for r in (pr.get("reviews") or {}).get("nodes") or []:
        add(r)
    for th in (pr.get("reviewThreads") or {}).get("nodes") or []:
        for c in (((th or {}).get("comments") or {}).get("nodes") or []):
            add(c)
    for n in (pr.get("timelineItems") or {}).get("nodes") or []:
        add(n)
    return out


def human_include(pr):
    tm = typename_map(pr)
    return lambda w: participant_class(w, tm.get(w)) == "human"


# ── Draft and readiness (t_rd) ──────────────────────────────────────────────

def draft_info(pr):
    """(created_as_draft, t_rd, n_ready, n_to_draft, t_rd_fallback).

    A PR was created as a draft iff its first draft-related timeline event is
    ReadyForReviewEvent; without such events the stored isDraft is the state at
    creation (it never changed). t_rd = first ReadyForReviewEvent for drafts,
    creation otherwise. A merged draft without a ReadyForReviewEvent (should not
    happen; truncated timeline) falls back to creation and is flagged.
    """
    created = parse_dt(pr.get("createdAt"))
    ev = []
    for n in (pr.get("timelineItems") or {}).get("nodes") or []:
        tn = (n or {}).get("__typename")
        if tn in ("ReadyForReviewEvent", "ConvertToDraftEvent"):
            t = parse_dt(n.get("createdAt"))
            if t:
                ev.append((t, tn))
    ev.sort(key=lambda e: e[0])
    n_ready = sum(1 for _, k in ev if k == "ReadyForReviewEvent")
    n_to_draft = sum(1 for _, k in ev if k == "ConvertToDraftEvent")
    if ev:
        as_draft = ev[0][1] == "ReadyForReviewEvent"
    else:
        as_draft = bool(pr.get("isDraft"))
    t_rd, fallback = created, False
    if as_draft:
        first_ready = next((t for t, k in ev if k == "ReadyForReviewEvent"), None)
        if first_ready is not None:
            t_rd = first_ready
        else:
            fallback = True
    return as_draft, t_rd, n_ready, n_to_draft, fallback


# ── Human first review, staleness trace ─────────────────────────────────────

def first_human_review(pr, include):
    """Time of the first review submission by a non-author accepted by include."""
    author = login_of(pr)
    ts = []
    for r in (pr.get("reviews") or {}).get("nodes") or []:
        w = login_of(r)
        if w == author or not include(w):
            continue
        t = parse_dt((r or {}).get("submittedAt"))
        if t:
            ts.append(t)
    return min(ts) if ts else None


def reviews_from(pr, start):
    """PR with the reviews submitted before `start` (t_rd) removed: human reviews
    of a draft are not events of the characteristics (decision of 02.11.2026)."""
    out = dict(pr)
    out["reviews"] = {"nodes": [r for r in (pr.get("reviews") or {}).get("nodes") or []
                                if (parse_dt((r or {}).get("submittedAt")) or start) >= start]}
    return out


def activity_trace_from(pr, include, start):
    """Staleness trace from `start` to merge: taskF1.activity_trace with its first
    event moved from creation to `start` (t_rd). Commits carry no login and are
    always kept; reviews and review-thread comments only from `include`."""
    opened = _dt_trace(start.strftime("%Y-%m-%dT%H:%M:%S")) if isinstance(start, datetime) else _dt_trace(start)
    merged = _dt_trace(pr.get("mergedAt"))
    if not opened or not merged or merged < opened:
        return None
    events = [opened]
    for n in (pr.get("commits") or {}).get("nodes") or []:
        t = _dt_trace(((n or {}).get("commit") or {}).get("authoredDate"))
        if t and opened <= t <= merged:
            events.append(t)
    for n in (pr.get("reviews") or {}).get("nodes") or []:
        if not include(login_of(n)):
            continue
        t = _dt_trace((n or {}).get("submittedAt"))
        if t and opened <= t <= merged:
            events.append(t)
    for th in (pr.get("reviewThreads") or {}).get("nodes") or []:
        for c in (((th or {}).get("comments") or {}).get("nodes") or []):
            if not include(login_of(c)):
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
    return max((events[i + 1] - events[i]).total_seconds() / 86400.0 for i in range(len(events) - 1))


def truncate_until(pr, until):
    """Reviews, review-thread comments and commits at or before `until`."""
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


# ── Size at t0 ──────────────────────────────────────────────────────────────

def size_at_t0(pr):
    """(lines_added, lines_deleted, commit_files, n_commits) over commits authored
    no later than creation. Commit-level counts exist only in corpus v2; v1 gives
    NaN sizes. commit_files sums per-commit file counts (a file touched by two
    commits counts twice). force-push rewrites the commit list: limitation."""
    created = parse_dt(pr.get("createdAt"))
    nodes = [((c or {}).get("commit") or {}) for c in (pr.get("commits") or {}).get("nodes") or []]
    have_stats = any("additions" in cm for cm in nodes)
    a = d = f = n = 0
    for cm in nodes:
        t = parse_dt(cm.get("authoredDate"))
        if not (t and created and t <= created):
            continue
        n += 1
        a += cm.get("additions") or 0
        d += cm.get("deletions") or 0
        f += cm.get("changedFilesIfAvailable") or 0
    if not have_stats:
        return math.nan, math.nan, math.nan, n
    return a, d, f, n


# ── HAIT inclusion filter (compute_features.should_include_merged) ──────────

def hait_filter(pr, check_base=True):
    created, merged = parse_dt(pr.get("createdAt")), parse_dt(pr.get("mergedAt"))
    if not merged:
        return False
    tm = round((merged - created).total_seconds() / 3600, 2)
    if tm < 1 / 60 or tm > 365 * 24:
        return False
    if (pr.get("additions") or 0) == 0 and (pr.get("deletions") or 0) == 0:
        return False
    if ((pr.get("reviews") or {}).get("totalCount") or 0) == 0:
        return False
    if not check_base:
        return True
    base = pr.get("baseRefName", "") or ""
    if base not in ALLOWED_BASES and not any(p in base for p in ("main", "master", "trunk", "release", "canary")):
        return False
    return True


def main_base_fallback(recs, nobase):
    """Apply the data fix: if the name filter keeps < MIN_NAMED_BASE_SHARE of the
    merged PRs, re-filter on the most frequent base branch (other HAIT rules kept).
    Sets rec["base_rule"] on merged records; returns the fallback branch or None."""
    merged = [r for r in recs if r["merged_at"]]
    if not merged:
        return None
    named = sum(r["hait_filter"] for r in merged) / len(merged)
    main = None
    if named < MIN_NAMED_BASE_SHARE:
        main = Counter(r["base_ref"] for r in merged).most_common(1)[0][0]
        for r in merged:
            r["hait_filter"] = int(nobase[r["pr_number"]] and r["base_ref"] == main)
    for r in merged:
        r["base_rule"] = "most_frequent" if main is not None else "name"
    return main


# ── Reverts ─────────────────────────────────────────────────────────────────

RE_BODY_REVERTS = re.compile(r"(?<![A-Za-z])Reverts\s+(?:([\w.-]+/[\w.-]+))?#(\d+)")
RE_TITLE_REVERT = re.compile(r'^Revert\s+"(.*)"\s*(?:\(#\d+\))?\s*$', re.DOTALL)
RE_COMMIT_REVERTS = re.compile(r"This reverts commit ([0-9a-f]{7,40})")


def revert_signals(pr, repo):
    """What a merged PR says it reverts: (numbers, quoted_title, shas)."""
    nums = set()
    for m in RE_BODY_REVERTS.finditer(pr.get("body") or ""):
        if m.group(1) and m.group(1).lower() != repo.lower():
            continue
        nums.add(int(m.group(2)))
    mt = RE_TITLE_REVERT.match((pr.get("title") or "").strip())
    quoted = mt.group(1).strip() if mt else None
    shas = set()
    for c in (pr.get("commits") or {}).get("nodes") or []:
        for m in RE_COMMIT_REVERTS.finditer(((c or {}).get("commit") or {}).get("message") or ""):
            shas.add(m.group(1).lower())
    return nums, quoted, shas


def merge_commit_oid(pr):
    for n in (pr.get("timelineItems") or {}).get("nodes") or []:
        if (n or {}).get("__typename") == "MergedEvent":
            return (((n.get("commit") or {}).get("oid")) or "").lower() or None
    return None


def match_reverts(targets, reverters, window_end=WINDOW_END, days=REVERT_DAYS):
    """Link merged PRs to the merged PRs that revert them.

    targets: number -> {merged_at, title, oids (set), base}; reverters: list of
    {number, merged_at, nums, quoted, shas, base}. A link needs the same base
    branch (reverts on release / backport branches are not counted). Evidence:
    body_reverts ("Reverts owner/repo#N" / "Reverts #N"), title_revert
    (Revert "<title>"), commit_hash ("This reverts commit <sha>" matching a
    commit of the PR or its merge commit from MergedEvent).

    Returns (columns per target number, list of links). Columns:
      is_revert_pr      the PR reverts another PR (excluded from the denominator);
      reverted_30d      reverted within `days` after merge by a merged PR, and that
                        revert was not itself reverted by the end of the window;
      revert_reverted   the first such revert was reverted in turn;
      revert_censored   merged later than window_end - days.
    """
    by_title = defaultdict(list)
    for n, t in targets.items():
        by_title[(t["title"] or "").strip()].append(n)
    by_prefix = defaultdict(list)
    for n, t in targets.items():
        for o in t["oids"]:
            by_prefix[o[:7]].append((o, n))
    hits = defaultdict(lambda: defaultdict(set))   # target -> reverter -> evidence
    for r in reverters:
        for n in r["nums"]:
            if n in targets:
                hits[n][r["number"]].add("body_reverts")
        if r["quoted"]:
            q = r["quoted"]
            cands = list(by_title.get(q, []))
            mq = re.match(r"^(.*)\s+\(#(\d+)\)$", q)
            if mq and int(mq.group(2)) in targets and \
                    (targets[int(mq.group(2))]["title"] or "").strip() == mq.group(1).strip():
                cands.append(int(mq.group(2)))
            for n in set(cands):
                hits[n][r["number"]].add("title_revert")
        for sh in r["shas"]:
            for o, n in by_prefix.get(sh[:7], []):
                if o.startswith(sh):
                    hits[n][r["number"]].add("commit_hash")
    rev_by_num = {r["number"]: r for r in reverters}
    links = []
    for n, rs in hits.items():
        for rn, evid in rs.items():
            if rn == n or rn not in rev_by_num:
                continue
            if (rev_by_num[rn].get("base") or "") != (targets[n].get("base") or ""):
                continue
            links.append({"target": n, "reverter": rn, "evidence": "|".join(sorted(evid)),
                          "target_merged_at": targets[n]["merged_at"],
                          "reverter_merged_at": rev_by_num[rn]["merged_at"]})
    is_revert = {l["reverter"] for l in links}
    reverted_by = defaultdict(list)
    for l in links:
        if l["reverter_merged_at"] and l["reverter_merged_at"] >= l["target_merged_at"]:
            reverted_by[l["target"]].append(l)
    out = {}
    for n, t in targets.items():
        m = t["merged_at"]
        cand = [l for l in reverted_by.get(n, [])
                if l["reverter_merged_at"] - m <= timedelta(days=days)]
        cand.sort(key=lambda l: l["reverter_merged_at"])
        best = cand[0] if cand else None
        rr = bool(best and reverted_by.get(best["reverter"]))
        out[n] = {"is_revert_pr": int(n in is_revert),
                  "reverted_30d": int(best is not None and not rr),
                  "revert_evidence": best["evidence"] if best else "",
                  "revert_pr": best["reverter"] if best else "",
                  "revert_merged_at": best["reverter_merged_at"].strftime("%Y-%m-%dT%H:%M:%SZ") if best else "",
                  "revert_reverted": int(rr),
                  "revert_censored": int(m > window_end - timedelta(days=days))}
    return out, links


# ── Per-PR record ───────────────────────────────────────────────────────────

def pr_record(pr, repo):
    """Everything about one PR that later stages need (no history yet)."""
    created = parse_dt(pr.get("createdAt"))
    merged = parse_dt(pr.get("mergedAt"))
    closed = parse_dt(pr.get("closedAt"))
    a = pr.get("author") or {}
    login, tname = a.get("login") or "", a.get("__typename")
    commits_total = (pr.get("commits") or {}).get("totalCount")
    if commits_total is None:
        commits_total = len((pr.get("commits") or {}).get("nodes") or [])
    rec = {
        "repo": repo, "pr_number": int(pr["number"]), "state": pr.get("state"),
        "author_login": login, "author_typename": tname or "",
        "author_group": author_group(login, tname),
        "in_main": int(in_main_population(login, tname)),
        "created_at": created, "merged_at": merged, "closed_at": closed,
        "commits_total": commits_total,
    }
    if not merged:
        return rec
    inc = human_include(pr)
    as_draft, t_rd, n_ready, n_to_draft, fb = draft_info(pr)
    pre = truncate_until(pr, merged)
    # clock A keeps every human review (comparison with ITIA and HAIT)
    fhr_a = first_human_review(pr, inc)
    fhr_pre_a = first_human_review(pre, inc)
    # clock D and the iterations: only reviews submitted at or after t_rd
    pr_d = reviews_from(pr, t_rd)
    pre_d = truncate_until(pr_d, merged)
    fhr = first_human_review(pr_d, inc)
    fhr_pre = first_human_review(pre_d, inc)
    tr_a = activity_trace_from(pr, inc, pr.get("createdAt"))
    tr_d = activity_trace_from(pr, inc, t_rd)
    la, ld, cf, nc = size_at_t0(pr)
    body = pr.get("body") or ""
    rec.update({
        "hait_filter": int(hait_filter(pr)),
        "created_as_draft": int(as_draft), "t_rd": t_rd, "t_rd_fallback": int(fb),
        "n_ready_events": n_ready, "n_to_draft_events": n_to_draft,
        "draft_h": _hours(created, t_rd),
        # characteristics, clock D (from t_rd) and A (from creation)
        "t_m_h_D": _hours(t_rd, merged), "t_m_h_A": _hours(created, merged),
        "first_review_in_draft": int(fhr_a is not None and fhr_a < t_rd),
        "t_r_h_D_raw": _hours(t_rd, fhr_a),
        "t_r_h_D": (_hours(t_rd, fhr) if fhr else None),
        "t_r_h_A": _hours(created, fhr_a),
        "t_r_pre_h_D": (_hours(t_rd, fhr_pre) if fhr_pre else None),
        "t_r_pre_h_A": _hours(created, fhr_pre_a),
        "n_it_human": count_iterations(pr_d, inc, rule=RULE_LITERAL),
        "n_it_human_pre": count_iterations(pre_d, inc, rule=RULE_LITERAL),
        "gap_human_days_D": max_gap_days(tr_d) if tr_d else math.nan,
        "gap_human_days_A": max_gap_days(tr_a) if tr_a else math.nan,
        "n_issue_comments": sum(1 for n in (pr.get("timelineItems") or {}).get("nodes") or []
                                if (n or {}).get("__typename") == "IssueComment"),
        # features known at t0 (PR-level)
        "lines_added_t0": la, "lines_deleted_t0": ld, "commit_files_t0": cf,
        "num_commits_t0_raw": nc, "num_commits_t0": max(1, nc),
        "desc_length": len(body), "title_length": len(pr.get("title") or ""),
        "has_screenshots": int(bool(IMAGE_PATTERN.search(body))),
        "has_code_blocks": int(bool(CODE_BLOCK_PATTERN.search(body))),
        "references_issues": int(bool(ISSUE_REF_PATTERN.search(body))),
        "is_draft_t0": int(as_draft),
        "day_of_week": created.weekday(), "hour_of_day": created.hour,
        # final values kept only for descriptive tables (never features)
        "final_additions": pr.get("additions"), "final_deletions": pr.get("deletions"),
        "final_changed_files": pr.get("changedFiles"),
    })
    for k in ("D", "A"):
        rec[f"stale30_{k}"] = int(rec[f"gap_human_days_{k}"] > STALE_DELTA_DAYS)
    return rec


# ── History features and thresholds (per repository) ────────────────────────

def _prefix_medians(values):
    s, out = [], [math.nan]
    for v in values:
        bisect.insort(s, v)
        n = len(s)
        out.append(s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]))
    return out


def _end(r):
    return r["merged_at"] or r["closed_at"]


def history_features(recs, sample, clock="D"):
    """Author history and project context at t0 for each record index in `sample`.

    recs: every collected PR of ONE repository (any state, any author).
    Only events strictly before t0 = created_at of the target PR are used:
      author_pr_count      author's PRs created before t0;
      author_tenure_days   days since the author's first PR created before t0;
      author_merge_rate    merged / (merged + closed unmerged), both before t0;
      author_avg_*         over author's PRs merged before t0, values as known
                           at their merge (n_it_human_pre, t_r_pre);
      author_commit_frequency_s  final commits of those PRs created in t0-90d;
      open_pr_count_t0     PRs created before t0 and not ended before t0 (only
                           collected PRs: PRs created before the window or still
                           open at collection are missing; limitation);
      merge_velocity_*_t0  merges in [t0 - 7 d, t0) and [t0 - 30 d, t0).
    Returns {index: {feature: value}}.
    """
    tr_key = f"t_r_pre_h_{clock}"
    created = sorted(r["created_at"] for r in recs)
    ends = sorted(_end(r) for r in recs if _end(r))
    merges = sorted(r["merged_at"] for r in recs if r["merged_at"])
    by_author = defaultdict(list)
    for i, r in enumerate(recs):
        by_author[r["author_login"]].append(i)
    out = {}
    for i in sample:
        r = recs[i]
        t0 = r["created_at"]
        f = {}
        # open at t0: created before t0 and not ended strictly before t0
        f["open_pr_count_t0"] = bisect.bisect_left(created, t0) - bisect.bisect_left(ends, t0)
        f["merge_velocity_7d_t0"] = bisect.bisect_left(merges, t0) - bisect.bisect_left(merges, t0 - timedelta(days=7))
        f["merge_velocity_30d_t0"] = bisect.bisect_left(merges, t0) - bisect.bisect_left(merges, t0 - timedelta(days=30))
        login = r["author_login"]
        mine = [recs[j] for j in by_author.get(login, []) if recs[j]["created_at"] < t0] if login else []
        f["author_pr_count"] = len(mine) if login else math.nan
        f["author_tenure_days"] = ((t0 - min(q["created_at"] for q in mine)).days if mine else (0 if login else math.nan))
        m_before = [q for q in mine if q["merged_at"] and q["merged_at"] < t0]
        c_before = [q for q in mine if not q["merged_at"] and q["closed_at"] and q["closed_at"] < t0]
        f["author_merge_rate"] = (len(m_before) / (len(m_before) + len(c_before))
                                  if (m_before or c_before) else math.nan)
        if m_before:
            f["author_avg_review_rounds_h"] = sum(q["n_it_human_pre"] for q in m_before) / len(m_before)
            v = sorted(q[tr_key] for q in m_before if q.get(tr_key) is not None)
            f["author_avg_tfr_human"] = (v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])) if v else math.nan
            lo = t0 - timedelta(days=90)
            f["author_commit_frequency_s"] = float(sum(q["commits_total"] for q in m_before if q["created_at"] >= lo))
        else:
            f["author_avg_review_rounds_h"] = math.nan
            f["author_avg_tfr_human"] = math.nan
            f["author_commit_frequency_s"] = 0.0 if login else math.nan
        out[i] = f
    return out


def thresholds_and_priors(recs, sample, clock):
    """Project-relative thresholds (median over sample PRs of the repo merged
    strictly before t0) and expanding label shares for the prior baseline."""
    tm_k, tr_k = f"t_m_h_{clock}", f"t_r_pre_h_{clock}"
    order = sorted(sample, key=lambda i: recs[i]["merged_at"])
    m_sorted = [recs[i]["merged_at"] for i in order]
    med_tm = _prefix_medians([recs[i][tm_k] for i in order])
    tr_vals, tr_pos = [], []
    for k, i in enumerate(order):
        v = recs[i][tr_k]
        if v is not None:
            tr_vals.append(v)
            tr_pos.append(k)
    med_tr = _prefix_medians(tr_vals)
    out = {}
    for i in sample:
        k = bisect.bisect_left(m_sorted, recs[i]["created_at"])
        j = bisect.bisect_left(tr_pos, k)
        out[i] = {"n_hist_tm": k, "thr_tm_h": med_tm[k], "n_hist_tr": j, "thr_tr_h": med_tr[j]}
    return out, order, m_sorted


def label_vector(r, thr, clock):
    """Binary labels of one PR given its thresholds (NaN = not evaluated)."""
    tm, tr = r[f"t_m_h_{clock}"], r[f"t_r_h_{clock}"]
    lab = {
        "IT": int(r["n_it_human"] >= 2),
        "ST": int(r[f"stale30_{clock}"]),
        "TM": (int(tm > thr["thr_tm_h"]) if thr["n_hist_tm"] >= N_MIN_HISTORY else math.nan),
        "TFR": (int(tr > thr["thr_tr_h"]) if (thr["n_hist_tr"] >= N_MIN_HISTORY and tr is not None) else math.nan),
        "TM24": int(tm > 24),
        "TFR24": (int(tr > 24) if tr is not None else math.nan),
    }
    return lab


# ── Driver ──────────────────────────────────────────────────────────────────

def load_repo_meta(raw_v2):
    meta = {}
    with open(os.path.join(raw_v2, "repos.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta[r["repo"]] = {"language": r["language"], "group": r["group"]}
    return meta


def iter_v2(raw_v2, repo):
    path = os.path.join(raw_v2, repo.replace("/", "_", 1), "prs.jsonl")
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


OUT_FIELDS = None


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    if isinstance(v, datetime):
        return v.strftime("%Y-%m-%dT%H:%M:%SZ")
    return v


def prep_repo(repo, meta, prs_iter):
    """Rows of the per-PR table for one repository (merged PRs only)."""
    recs, reverters, targets, nobase = [], [], {}, {}
    seen = set()
    for pr in prs_iter:
        n = int(pr["number"])
        if n in seen:
            continue
        seen.add(n)
        rec = pr_record(pr, repo)
        recs.append(rec)
        if rec["merged_at"]:
            oids = {(((c or {}).get("commit") or {}).get("oid") or "").lower()
                    for c in (pr.get("commits") or {}).get("nodes") or []}
            mo = merge_commit_oid(pr)
            if mo:
                oids.add(mo)
            oids.discard("")
            base = pr.get("baseRefName") or ""
            targets[n] = {"merged_at": rec["merged_at"], "title": pr.get("title") or "",
                          "oids": oids, "base": base}
            nums, quoted, shas = revert_signals(pr, repo)
            if nums or quoted or shas:
                reverters.append({"number": n, "merged_at": rec["merged_at"], "nums": nums,
                                  "quoted": quoted, "shas": shas, "base": base})
            rec["base_ref"] = base
            nobase[n] = hait_filter(pr, check_base=False)
    main_base_fallback(recs, nobase)
    rev, links = match_reverts(targets, reverters)
    merged_idx = [i for i, r in enumerate(recs) if r["merged_at"]]
    sample = [i for i in merged_idx if recs[i]["hait_filter"]]
    rows = []
    hist = {c: history_features(recs, merged_idx, c) for c in ("D", "A")}
    # thresholds over the analysis sample of each population
    thr = {}
    for pop, idx in (("main", [i for i in sample if recs[i]["in_main"]]), ("all", sample)):
        for c in ("D", "A"):
            thr[(pop, c)] = thresholds_and_priors(recs, idx, c)[0]
    for i in merged_idx:
        r = dict(recs[i])
        r.update({"language": meta["language"], "group": meta["group"]})
        r.update(rev[r["pr_number"]])
        for c in ("D", "A"):
            for k, v in hist[c][i].items():
                r[f"{k}" if c == "D" else f"{k}__A"] = v
        for pop in ("main", "all"):
            for c in ("D", "A"):
                t = thr[(pop, c)].get(i)
                if t is None:
                    continue
                suf = f"__{pop}_{c}"
                r.update({k + suf: v for k, v in t.items()})
                for lk, lv in label_vector(r, t, c).items():
                    r[f"y_{lk}{suf}"] = lv
        rows.append(r)
    for l in links:
        l.update({"repo": repo})
    return rows, links


def add_prior_shares(rows):
    """Expanding share of positive labels among earlier merged sample PRs of the
    same repository (merged strictly before t0): the 'project prior' baseline."""
    import numpy as np
    for pop in ("main", "all"):
        for c in ("D", "A"):
            suf = f"__{pop}_{c}"
            for tg in ("IT", "ST", "TM", "TFR"):
                col = f"y_{tg}{suf}"
                by_repo = defaultdict(list)
                for k, r in enumerate(rows):
                    if col in r and r[col] == r[col] and r[col] != "":
                        by_repo[r["repo"]].append(k)
                for repo, ks in by_repo.items():
                    ks_m = sorted(ks, key=lambda k: rows[k]["merged_at"])
                    m_sorted = [rows[k]["merged_at"] for k in ks_m]
                    cum = np.concatenate([[0.0], np.cumsum([rows[k][col] for k in ks_m])])
                    for k in ks:
                        j = bisect.bisect_left(m_sorted, rows[k]["created_at"])
                        rows[k][f"prior_{tg}{suf}"] = (cum[j] / j) if j else math.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["v2"], default="v2")
    ap.add_argument("--repos", default="", help="comma-separated owner/name (default: all in repos.csv)")
    ap.add_argument("--out", default="", help="output CSV (default: <R3_OUT>/pr_table_v2.csv)")
    args = ap.parse_args()
    P = r3_paths()
    meta = load_repo_meta(P["raw_v2"])
    repos = [x for x in args.repos.split(",") if x] or list(meta)
    out = args.out or os.path.join(P["out"], "pr_table_v2.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done_marker = out + ".done"
    if os.path.exists(done_marker):
        print(f"prep: {out} already complete, skipping")
        return
    part_dir = out + ".parts"
    os.makedirs(part_dir, exist_ok=True)
    t_start = time.time()
    for repo in repos:
        part = os.path.join(part_dir, repo.replace("/", "_", 1) + ".json")
        if os.path.exists(part):
            continue
        rows, links = prep_repo(repo, meta[repo], iter_v2(P["raw_v2"], repo))
        with open(part + ".tmp", "w") as f:
            json.dump({"rows": [{k: _fmt(v) for k, v in r.items()} for r in rows],
                       "revert_links": [{k: _fmt(v) for k, v in l.items()} for l in links]}, f)
        os.replace(part + ".tmp", part)
        print(f"  {repo}: {len(rows)} merged ({time.time() - t_start:.0f}s)", flush=True)
    rows, links = [], []
    for repo in repos:
        with open(os.path.join(part_dir, repo.replace("/", "_", 1) + ".json")) as f:
            z = json.load(f)
        rows.extend(z["rows"])
        links.extend(z["revert_links"])
    with open(os.path.join(os.path.dirname(out), "revert_links_v2.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["repo", "target", "reverter", "evidence",
                                          "target_merged_at", "reverter_merged_at"])
        w.writeheader()
        w.writerows(links)
    for r in rows:
        for k in ("created_at", "merged_at"):
            r[k] = parse_dt(r[k])
        for k, v in list(r.items()):
            if k.startswith("y_") and v == "":
                r[k] = math.nan
    add_prior_shares(rows)
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(out + ".tmp", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in fields})
    os.replace(out + ".tmp", out)
    open(done_marker, "w").write(time.strftime("%Y-%m-%dT%H:%M:%S"))
    print(f"prep done: {len(rows)} rows -> {out} ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
