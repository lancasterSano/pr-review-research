#!/usr/bin/env python3
"""Task C1 — enumerate every reviewer / review-thread commenter login in the corpus.

Produces <DISSERTATION_DIR>/journal_articles/HAIT_2026/revision_results/C-login-classification.csv
with, for every distinct login:

    login, n_prs, n_prs_as_review_author, n_prs_as_thread_commenter,
    n_review_events, n_thread_comments, n_repos, repos, class, tool, audit_flag

`class` comes from the explicit allow/block lists in taskC_common.py.
`audit_flag` marks logins that LOOK automated by name pattern but are not on
either list — those are the ones a human must eyeball. Nothing is classified by
substring matching.

Usage:
    python taskC1_login_scan.py            # write full CSV, print top 100
"""

import csv
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taskC_common import (  # noqa: E402
    OUT_DIR, iter_raw, pr_participants, classify_login, ai_tool_name,
)

# Name patterns that merely *suggest* automation. Used ONLY to raise an audit
# flag for manual inspection, never to classify.
BOTLIKE = re.compile(
    r"(\[bot\]$|(^|[-_])bot([-_]|$)|(^|[-_])ai([-_]|$)|copilot|coderabbit|greptile|"
    r"devin|cursor|codex|gpt|llm|assistant|reviewer[-_]?bot|agent)",
    re.IGNORECASE,
)


def main():
    n_prs_review = defaultdict(int)
    n_prs_thread = defaultdict(int)
    n_prs_any = defaultdict(int)
    n_review_events = defaultdict(int)
    n_thread_comments = defaultdict(int)
    repos_seen = defaultdict(set)

    totals = {"merged": 0, "open": 0}

    for pr_type in ("merged", "open"):
        for repo, pr in iter_raw(pr_type):
            totals[pr_type] += 1
            rev, thr = pr_participants(pr)
            for login in rev:
                n_prs_review[login] += 1
            for login in thr:
                n_prs_thread[login] += 1
            for login in rev | thr:
                n_prs_any[login] += 1
                repos_seen[login].add(repo)
            # event-level counts
            for r in pr.get("reviews", {}).get("nodes", []) or []:
                a = (r.get("author") or {}).get("login")
                if a:
                    n_review_events[a] += 1
            for t in pr.get("reviewThreads", {}).get("nodes", []) or []:
                for c in (t.get("comments", {}) or {}).get("nodes", []) or []:
                    a = (c.get("author") or {}).get("login")
                    if a:
                        n_thread_comments[a] += 1

    rows = []
    for login, n in n_prs_any.items():
        cls = classify_login(login)
        tool = ai_tool_name(login) or ""
        audit = ""
        if cls == "human" and BOTLIKE.search(login):
            audit = "REVIEW: bot-like name, not on either list"
        rows.append({
            "login": login,
            "n_prs": n,
            "n_prs_as_review_author": n_prs_review.get(login, 0),
            "n_prs_as_thread_commenter": n_prs_thread.get(login, 0),
            "n_review_events": n_review_events.get(login, 0),
            "n_thread_comments": n_thread_comments.get(login, 0),
            "n_repos": len(repos_seen[login]),
            "repos": "|".join(sorted(repos_seen[login])),
            "class": cls,
            "tool": tool,
            "audit_flag": audit,
        })

    rows.sort(key=lambda r: (-r["n_prs"], r["login"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "C-login-classification.csv")
    cols = ["rank", "login", "n_prs", "n_prs_as_review_author",
            "n_prs_as_thread_commenter", "n_review_events", "n_thread_comments",
            "n_repos", "repos", "class", "tool", "audit_flag"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"PRs scanned: merged={totals['merged']} open={totals['open']} "
          f"total={totals['merged'] + totals['open']}")
    print(f"Distinct participant logins: {len(rows)}")
    print(f"Wrote {out}")

    print("\n--- Top 100 by number of PRs participated in ---")
    print(f"{'#':>4} {'login':38s} {'nPR':>7} {'rev':>7} {'thr':>7} {'repos':>5}  class")
    for r in rows[:100]:
        print(f"{r['rank']:>4} {r['login'][:38]:38s} {r['n_prs']:>7} "
              f"{r['n_prs_as_review_author']:>7} {r['n_prs_as_thread_commenter']:>7} "
              f"{r['n_repos']:>5}  {r['class']}"
              + (f"  [{r['tool']}]" if r['tool'] else "")
              + (f"  <<{r['audit_flag']}" if r['audit_flag'] else ""))

    flagged = [r for r in rows if r["audit_flag"]]
    print(f"\n--- Bot-like names NOT on either list ({len(flagged)}) — manual audit ---")
    for r in flagged[:80]:
        print(f"  {r['login']:40s} nPR={r['n_prs']:>6}  repos={r['repos']}")

    print("\n--- Logins classified as ai_assistant (all) ---")
    for r in rows:
        if r["class"] == "ai_assistant":
            print(f"  {r['login']:40s} nPR={r['n_prs']:>6}  rev={r['n_prs_as_review_author']:>6} "
                  f"thr={r['n_prs_as_thread_commenter']:>6}  {r['tool']}")

    print("\n--- Logins classified as other_bot (all) ---")
    for r in rows:
        if r["class"] == "other_bot":
            print(f"  {r['login']:40s} nPR={r['n_prs']:>6}  {r['repos']}")

    print("\n--- Logins classified as uncertain (NOT counted as AI) ---")
    for r in rows:
        if r["class"] == "uncertain":
            print(f"  {r['login']:40s} nPR={r['n_prs']:>6}  {r['repos']}")

    n_by_class = defaultdict(int)
    prs_by_class = defaultdict(int)
    for r in rows:
        n_by_class[r["class"]] += 1
        prs_by_class[r["class"]] += r["n_prs"]
    print("\n--- Class totals ---")
    for k in ("human", "ai_assistant", "other_bot", "uncertain"):
        print(f"  {k:14s} logins={n_by_class[k]:>6}  sum(PR participations)={prs_by_class[k]:>8}")


if __name__ == "__main__":
    main()
