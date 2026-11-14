#!/usr/bin/env python3
"""Audit every participant login in corpus v2 (data/raw_v2, 40 repositories).

The allow/block lists in taskC_common.py were built on the 12-repository HAIT
corpus, and classify_login() treats every unlisted login as human. This script
finds the logins a human must eyeball before corpus v2 is used for any
"human events only" characteristic.

For every distinct login it records the account type GitHub reports
(__typename: User / Bot / ...), event counts by role, the repositories it
appears in and a few sample review / comment bodies, then raises audit flags:

    typename_bot   GitHub says Bot, but the login is on no list
    name_botlike   name pattern suggests automation, login on no list
    templated      most of its review / comment bodies share one opening
    new_repo_only  seen only in the 28 repositories added for corpus v2

Nothing is classified here; flags only select logins for manual review.

Outputs (git-ignored) in revision/output/corpus_v2/login_audit/:
    logins.csv        one row per login
    samples.jsonl     sample bodies for every flagged login

Usage:
    python taskV2_login_audit.py [--raw DIR] [--out DIR]
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taskC_common import ROOT, classify_login, ai_tool_name  # noqa: E402

RAW_V2 = os.path.join(ROOT, "data", "raw_v2")
OUT = os.path.join(ROOT, "revision", "output", "corpus_v2", "login_audit")

# The 12 repositories of the HAIT corpus (directory names in raw_v2).
HAIT_REPOS = {
    "microsoft_vscode", "woocommerce_woocommerce", "vercel_next.js",
    "supabase_supabase", "mattermost_mattermost", "grafana_grafana",
    "strapi_strapi", "discourse_discourse", "angular_angular",
    "nodejs_node", "facebook_react", "django_django",
}

# Wider than taskC1_login_scan.BOTLIKE: project bots in the new repositories
# are often named *-robot, *-ci, *-build, *-automation, *-release.
BOTLIKE = re.compile(
    r"(\[bot\]$|(^|[-_])bots?([-_]|$)|bot$|^bot|robot|(^|[-_])ci([-_]|$)|"
    r"(^|[-_])ai([-_]|$)|automation|automat|release|build|deploy|merge|"
    r"copilot|coderabbit|greptile|devin|cursor|codex|gpt|llm|claude|gemini|"
    r"assistant|agent|actions?$|-app$|-apps$|ops$)",
    re.IGNORECASE,
)

N_SAMPLES = 6
SAMPLE_CHARS = 400


def opening(body):
    """Normalised first 50 characters, used to spot templated output."""
    s = re.sub(r"\s+", " ", body or "").strip().lower()
    s = re.sub(r"\d+", "#", s)
    return s[:50]


def login_of(node):
    if not isinstance(node, dict):
        return None, None
    a = node.get("author") if "author" in node else node
    if not isinstance(a, dict) or not a.get("login"):
        return None, None
    return a["login"], a.get("__typename")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW_V2)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    typenames = defaultdict(Counter)
    counts = defaultdict(Counter)       # role -> events
    prs = defaultdict(set)              # (repo, number) where login takes part
    repo_prs = defaultdict(Counter)     # repo -> PRs
    openings = defaultdict(Counter)
    n_bodies = Counter()
    samples = defaultdict(list)
    sample_repos = defaultdict(set)

    def seen(login, typename, role, repo, number, body=None):
        typenames[login][typename or "?"] += 1
        counts[login][role] += 1
        key = (repo, number)
        if key not in prs[login]:
            prs[login].add(key)
            repo_prs[login][repo] += 1
        if body is not None and body.strip():
            n_bodies[login] += 1
            openings[login][opening(body)] += 1
            # Spread samples over repositories first, then fill up.
            if len(samples[login]) < N_SAMPLES and (
                repo not in sample_repos[login] or len(samples[login]) >= len(repo_prs[login])
            ):
                samples[login].append({"repo": repo, "pr": number, "role": role,
                                       "body": body[:SAMPLE_CHARS]})
                sample_repos[login].add(repo)

    repos = sorted(d for d in os.listdir(args.raw)
                   if os.path.isfile(os.path.join(args.raw, d, "prs.jsonl")))
    total = 0
    for repo in repos:
        n = 0
        with open(os.path.join(args.raw, repo, "prs.jsonl"), encoding="utf-8") as f:
            for line in f:
                pr = json.loads(line)
                num = pr.get("number")
                lg, tn = login_of(pr.get("author"))
                if lg:
                    seen(lg, tn, "pr_author", repo, num)
                lg, tn = login_of(pr.get("mergedBy"))
                if lg:
                    seen(lg, tn, "merged_by", repo, num)
                for r in (pr.get("reviews") or {}).get("nodes") or []:
                    lg, tn = login_of(r)
                    if lg:
                        seen(lg, tn, "review", repo, num, r.get("body"))
                for t in (pr.get("reviewThreads") or {}).get("nodes") or []:
                    for c in (t.get("comments") or {}).get("nodes") or []:
                        lg, tn = login_of(c)
                        if lg:
                            seen(lg, tn, "thread_comment", repo, num, c.get("body"))
                for it in (pr.get("timelineItems") or {}).get("nodes") or []:
                    if isinstance(it, dict):
                        lg, tn = login_of(it.get("actor"))
                        if lg:
                            seen(lg, tn, "timeline", repo, num)
                for rr in (pr.get("reviewRequests") or {}).get("nodes") or []:
                    rv = (rr or {}).get("requestedReviewer") or {}
                    if rv.get("login"):
                        seen(rv["login"], rv.get("__typename"), "requested", repo, num)
                n += 1
        total += n
        print(f"{repo}: {n} PRs", file=sys.stderr)

    rows = []
    for login in counts:
        cls = classify_login(login)
        listed = cls != "human"
        tn = typenames[login].most_common(1)[0][0]
        top_open, top_n = (openings[login].most_common(1)[0] if openings[login] else ("", 0))
        templ = top_n / n_bodies[login] if n_bodies[login] else 0.0
        repos_in = set(repo_prs[login])
        flags = []
        if not listed and tn == "Bot":
            flags.append("typename_bot")
        if not listed and BOTLIKE.search(login):
            flags.append("name_botlike")
        if not listed and n_bodies[login] >= 20 and templ >= 0.5:
            flags.append("templated")
        if repos_in and not (repos_in & HAIT_REPOS):
            flags.append("new_repo_only")
        c = counts[login]
        rows.append({
            "login": login,
            "typename": tn,
            "typenames": "|".join(f"{k}:{v}" for k, v in typenames[login].items()),
            "class": cls,
            "tool": ai_tool_name(login) or "",
            "n_prs": len(prs[login]),
            "n_review": c["review"],
            "n_thread_comment": c["thread_comment"],
            "n_timeline": c["timeline"],
            "n_pr_author": c["pr_author"],
            "n_merged_by": c["merged_by"],
            "n_requested": c["requested"],
            "n_repos": len(repos_in),
            "repos": "|".join(f"{r}:{k}" for r, k in repo_prs[login].most_common()),
            "templated_share": round(templ, 3),
            "top_opening": top_open,
            "flags": "|".join(f for f in flags if f != "new_repo_only"),
            "new_repo_only": int("new_repo_only" in flags),
        })
    rows.sort(key=lambda r: (-r["n_prs"], r["login"]))

    with open(os.path.join(args.out, "logins.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "samples.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            if r["flags"] or r["class"] != "human":
                f.write(json.dumps({"login": r["login"], "flags": r["flags"],
                                    "samples": samples[r["login"]]},
                                   ensure_ascii=False) + "\n")

    flagged = [r for r in rows if r["flags"]]
    print(f"PRs: {total}; logins: {len(rows)}; flagged: {len(flagged)}")
    print(Counter(r["class"] for r in rows))


if __name__ == "__main__":
    main()
