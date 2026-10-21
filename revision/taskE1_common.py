#!/usr/bin/env python3
"""Shared helpers for ІКТ-2026 experiment E1 (reduced) — human reactions to assistant output.

Units: (a) a review thread started by an AI assistant in a MERGED pull request; (b) a review
submitted by an AI assistant (its `state` tells whether the assistant issued a verdict).
Thread features are restricted to what is known when the assistant's comment appears; replies,
resolution and later commits are outcomes.

Reads:  data/raw/<12 study repos>/merged_prs.json (via taskC_common.iter_raw)
Writes: nothing (callers write into <DISSERTATION_DIR>/conf_papers/october_2026/IKT2026/data/).
"""
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from taskC_common import DISSERTATION, iter_raw, is_ai, ai_tool_name, classify_login, parse_dt  # noqa: E402

OUT = os.path.join(DISSERTATION, "conf_papers", "october_2026", "IKT2026", "data")
THREADS_CSV = os.path.join(OUT, "E1-threads.csv")
REVIEWS_CSV = os.path.join(OUT, "E1-reviews.csv")
SEED = 20260907
N_JOBS = 4

# Repositories with >= 600 assistant-started threads (census of 2026-09-07) — predictability check.
BIG_REPOS = ["microsoft/vscode", "woocommerce/woocommerce", "vercel/next.js", "supabase/supabase",
             "grafana/grafana", "mattermost/mattermost"]
# Row order of Table 3 in the paper — Table 4 uses the same rows.
TABLE3_REPOS = ["microsoft/vscode", "woocommerce/woocommerce", "vercel/next.js", "supabase/supabase",
                "mattermost/mattermost", "grafana/grafana", "strapi/strapi", "discourse/discourse"]

FEATURES = ["tool", "ext", "has_suggestion", "body_len", "n_code_blocks", "severity", "ai_idx", "n_ai_threads",
            "pr_additions", "pr_deletions", "pr_changed_files", "author_is_bot", "hours_since_open",
            "hours_since_trigger", "commits_before"]
CATEGORICAL = ["tool", "ext"]

# Commit messages GitHub writes when a reviewer's suggestion is applied from the UI, or when a
# human commits with an assistant as co-author.
APPLY_RX = re.compile(
    r"apply suggestions? from code review|apply suggestion|"
    r"co-authored-by:\s*(copilot|coderabbit|gemini|claude|codex|cursor|graphite|vercel|diffray|greptile|devin)", re.I)

SEV2_RX = re.compile(r"Potential issue|_⚠️|\bCritical\b|\bMajor\b|!\[high\]|"
                     r"\b(critical|security|vulnerab\w*|crash|bug|error|leak|race condition)\b", re.I)
SEV1_RX = re.compile(r"[Nn]itpick|\bMinor\b|\bTrivial\b|Refactor suggestion|!\[medium\]|!\[low\]|"
                     r"\b(nit|minor|style|typo|cosmetic)\b", re.I)


def severity(body):
    b = body or ""
    if SEV2_RX.search(b):
        return 2
    if SEV1_RX.search(b):
        return 1
    return 0


def _login(node):
    return ((node or {}).get("author") or {}).get("login") or ""


def pr_commits(pr):
    """[(authored datetime, message)] for commits authored after the PR was opened, sorted by time."""
    created = parse_dt(pr.get("createdAt"))
    out = []
    for c in (pr.get("commits") or {}).get("nodes") or []:
        cc = (c or {}).get("commit") or {}
        d = parse_dt(cc.get("authoredDate"))
        if d and created and d > created:
            out.append((d, cc.get("message") or ""))
    out.sort(key=lambda x: x[0])
    return out


def thread_record(repo, pr, thread, ai_idx, n_ai, human_paths, commits):
    cs = (thread.get("comments") or {}).get("nodes") or []
    first = cs[0]
    login = _login(first)
    body = first.get("body") or ""
    t0 = parse_dt(first.get("createdAt"))
    created = parse_dt(pr.get("createdAt"))

    before = [d for d, _ in commits if t0 and d <= t0]
    after = [(d, m) for d, m in commits if t0 and d > t0]
    hours_open = (t0 - created).total_seconds() / 3600 if (t0 and created) else np.nan
    hours_commit = (t0 - before[-1]).total_seconds() / 3600 if before else np.nan
    hours_trigger = np.nanmin([hours_open, hours_commit]) if not (np.isnan(hours_open) and np.isnan(hours_commit)) else np.nan

    reply = next((c for c in cs[1:] if classify_login(_login(c)) == "human"), None)
    author_login = _login(pr)
    author_is_bot = int(((pr.get("author") or {}).get("__typename") == "Bot") or is_ai(author_login))
    path = thread.get("path") or ""
    has_sugg = int("```suggestion" in body)
    apply_after = int(any(APPLY_RX.search(m) for _, m in after))

    return {
        "repo": repo, "pr_number": pr.get("number"), "tool": ai_tool_name(login) or login,
        "created_at": first.get("createdAt") or "", "path": path,
        "ext": os.path.splitext(path)[1].lower().lstrip(".")[:10] or "none",
        # features known at comment time
        "has_suggestion": has_sugg, "body_len": len(body), "n_code_blocks": body.count("```") // 2,
        "severity": severity(body), "ai_idx": ai_idx, "n_ai_threads": n_ai,
        "pr_additions": pr.get("additions") or 0, "pr_deletions": pr.get("deletions") or 0,
        "pr_changed_files": pr.get("changedFiles") or 0, "author_is_bot": author_is_bot,
        "hours_since_open": round(hours_open, 3) if hours_open == hours_open else np.nan,
        "hours_since_trigger": round(float(hours_trigger), 3) if hours_trigger == hours_trigger else np.nan,
        "commits_before": len(before),
        # outcomes — never features
        "is_resolved": int(bool(thread.get("isResolved"))),
        "reacted": int(reply is not None),
        "reply_by_author": int(reply is not None and _login(reply) == author_login),
        "commits_after": len(after),
        "apply_commit_after": apply_after,
        "sugg_applied": int(has_sugg and apply_after and not author_is_bot),
        "human_same_file": int(path in human_paths),
    }


def iter_ai_threads():
    """One record per assistant-started thread; threads in time order within a PR."""
    for repo, pr in iter_raw("merged"):
        nodes = (pr.get("reviewThreads") or {}).get("nodes") or []
        ai, human_paths = [], set()
        for t in nodes:
            cs = (t.get("comments") or {}).get("nodes") or []
            if not cs:
                continue
            login = _login(cs[0])
            if is_ai(login):
                ai.append(t)
            elif classify_login(login) == "human":
                human_paths.add(t.get("path") or "")
        if not ai:
            continue
        ai.sort(key=lambda t: (((t.get("comments") or {}).get("nodes") or [{}])[0].get("createdAt") or ""))
        commits = pr_commits(pr)
        for i, t in enumerate(ai):
            yield thread_record(repo, pr, t, i, len(ai), human_paths, commits)


def iter_ai_reviews():
    """One record per review submitted by an assistant account: repo, pr_number, tool, state, submitted_at."""
    for repo, pr in iter_raw("merged"):
        for r in (pr.get("reviews") or {}).get("nodes") or []:
            login = _login(r)
            if is_ai(login):
                yield {"repo": repo, "pr_number": pr.get("number"), "tool": ai_tool_name(login) or login,
                       "state": r.get("state") or "", "submitted_at": r.get("submittedAt") or ""}


# ── modelling helpers (predictability check) ─────────────────────────────────
def chrono_split(df, test_frac=0.25):
    d = df.sort_values("created_at", kind="stable").reset_index(drop=True)
    n_test = max(1, int(round(len(d) * test_frac)))
    return d.iloc[:-n_test].copy(), d.iloc[-n_test:].copy()


def prep_X(df, cats=None):
    X = df[FEATURES].copy()
    out = {}
    for c in CATEGORICAL:
        categories = cats[c] if cats else sorted(X[c].astype(str).unique().tolist())
        X[c] = pd.Categorical(X[c].astype(str), categories=categories)
        out[c] = categories
    return X, out


def fit_lgbm(X, y, seed=SEED):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           random_state=seed, n_jobs=N_JOBS, verbose=-1)
    m.fit(X, y, categorical_feature=CATEGORICAL)
    return m


def ap(y, s):
    """Average precision; NaN when only one class is present."""
    from sklearn.metrics import average_precision_score
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, np.asarray(s, dtype=float)))
