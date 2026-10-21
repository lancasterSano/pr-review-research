#!/usr/bin/env python3
"""Task C2 — descriptive prevalence of AI-assistant participation in code review.

Outputs (into <DISSERTATION_DIR>/journal_articles/HAIT_2026/revision_results/):
  C-ai-pr-flags.csv   one row per PR: repo, pr_number, pr_type, created_at,
                      in_analysis_set, has_ai_review, ai_tools, ai_authored
  C-ai-per-repo.csv   per-repository counts, Table-2 row order
  C-ai-by-year.csv    counts by year of PR creation
  C-ai-tool-by-repo.csv  which tool appears in which repository

Definitions
-----------
"AI review" on a PR  :=  at least one login on the AI allowlist appears as the
                         author of a review event or of a review-thread comment
                         on that PR, and that login is not the PR's own author.
                         (Self-review exclusion mirrors compute_features.py.)
"AI-authored PR"     :=  the PR's author login is on the AI allowlist.

Two corpora are reported side by side:
  raw       — every PR in data/raw for the 12 study repositories
  analysis  — only PRs that survive into all_{merged,open}_features.csv, i.e.
              the corpus behind Table 2 / Tables 3-4 of the article. The new
              Table 2 column must come from THIS corpus so it is consistent
              with the other columns of that table.

Known data asymmetry (reported, not hidden): open_prs.json does NOT contain a
reviewThreads field — only merged_prs.json does. For open PRs, AI participation
is therefore detectable only through review events, not through inline thread
comments. This makes the open-PR share a LOWER BOUND.
"""

import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taskC_common import (  # noqa: E402
    OUT_DIR, MERGED_CSV, OPEN_CSV, TABLE2_ORDER,
    iter_raw, pr_participants, is_ai, ai_tool_name, load_processed,
)


def ai_event_dates(pr):
    """Timestamps of AI review EVENTS on a PR (not the PR creation date).

    Needed because the year buckets elsewhere are keyed on PR creation, and a PR
    created in 2023 can receive its AI review in 2026. Reporting the event dates
    separately gives the honest adoption timeline.
    """
    author = (pr.get("author") or {}).get("login") or ""
    out = []
    for r in pr.get("reviews", {}).get("nodes", []) or []:
        a = (r.get("author") or {}).get("login")
        if a and is_ai(a) and a != author:
            out.append((ai_tool_name(a), r.get("submittedAt") or ""))
    for t in pr.get("reviewThreads", {}).get("nodes", []) or []:
        for c in (t.get("comments", {}) or {}).get("nodes", []) or []:
            a = (c.get("author") or {}).get("login")
            if a and is_ai(a) and a != author:
                out.append((ai_tool_name(a), c.get("createdAt") or ""))
    return out


def pct(a, b):
    return round(100.0 * a / b, 3) if b else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── which PRs survive into the modelling corpus ─────────────────────────
    in_analysis = {"merged": set(), "open": set()}
    for pr_type, path in (("merged", MERGED_CSV), ("open", OPEN_CSV)):
        for r in load_processed(path):
            in_analysis[pr_type].add((r["repo"], int(r["pr_number"])))

    # ── scan raw JSON ───────────────────────────────────────────────────────
    flags = []
    ev_by_tool_year = defaultdict(lambda: defaultdict(int))
    ev_first = {}
    for pr_type in ("merged", "open"):
        for repo, pr in iter_raw(pr_type):
            author = (pr.get("author") or {}).get("login") or ""
            rev, thr = pr_participants(pr)
            participants = rev | thr
            ai_logins = sorted(l for l in participants if is_ai(l) and l != author)
            tools = sorted({ai_tool_name(l) for l in ai_logins})
            for tool, ts in ai_event_dates(pr):
                if ts:
                    ev_by_tool_year[tool][ts[:4]] += 1
                    if tool not in ev_first or ts < ev_first[tool]:
                        ev_first[tool] = ts
            flags.append({
                "repo": repo,
                "pr_number": pr["number"],
                "pr_type": pr_type,
                "created_at": pr.get("createdAt") or "",
                "merged_at": pr.get("mergedAt") or "",
                "in_analysis_set": int((repo, int(pr["number"])) in in_analysis[pr_type]),
                "has_ai_review": int(bool(ai_logins)),
                "n_ai_logins": len(ai_logins),
                "ai_logins": "|".join(ai_logins),
                "ai_tools": "|".join(t for t in tools if t),
                "ai_authored": int(bool(author) and is_ai(author)),
            })

    with open(os.path.join(OUT_DIR, "C-ai-pr-flags.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flags[0].keys()))
        w.writeheader()
        w.writerows(flags)

    # ── (a) per repository ──────────────────────────────────────────────────
    def agg(rows, key_fn):
        out = defaultdict(lambda: defaultdict(int))
        for r in rows:
            k = key_fn(r)
            out[k][f"{r['pr_type']}_total"] += 1
            out[k][f"{r['pr_type']}_ai"] += r["has_ai_review"]
            out[k]["pooled_total"] += 1
            out[k]["pooled_ai"] += r["has_ai_review"]
            out[k]["ai_authored"] += r["ai_authored"]
        return out

    raw_repo = agg(flags, lambda r: r["repo"])
    ana_repo = agg([r for r in flags if r["in_analysis_set"]], lambda r: r["repo"])

    per_repo_rows = []
    for repo in TABLE2_ORDER:
        a, b = ana_repo[repo], raw_repo[repo]
        per_repo_rows.append({
            "repo": repo,
            # analysis corpus (= the corpus of Table 2)
            "merged_prs": a["merged_total"],
            "merged_ai": a["merged_ai"],
            "merged_ai_pct": pct(a["merged_ai"], a["merged_total"]),
            "open_prs": a["open_total"],
            "open_ai": a["open_ai"],
            "open_ai_pct": pct(a["open_ai"], a["open_total"]),
            "pooled_prs": a["pooled_total"],
            "pooled_ai": a["pooled_ai"],
            "pooled_ai_pct": pct(a["pooled_ai"], a["pooled_total"]),
            # raw corpus (pre-filter), for reference
            "raw_merged_prs": b["merged_total"],
            "raw_merged_ai": b["merged_ai"],
            "raw_merged_ai_pct": pct(b["merged_ai"], b["merged_total"]),
            "raw_open_prs": b["open_total"],
            "raw_open_ai": b["open_ai"],
            "raw_open_ai_pct": pct(b["open_ai"], b["open_total"]),
            "raw_pooled_prs": b["pooled_total"],
            "raw_pooled_ai": b["pooled_ai"],
            "raw_pooled_ai_pct": pct(b["pooled_ai"], b["pooled_total"]),
            "ai_authored_prs": b["ai_authored"],
        })

    def totals(rows, keys):
        t = {k: sum(r[k] for r in rows) for k in keys}
        return t

    cnt_keys = [k for k in per_repo_rows[0] if k not in ("repo",) and not k.endswith("_pct")]
    tot = totals(per_repo_rows, cnt_keys)
    tot["repo"] = "Pooled"
    for a, b in (("merged_ai", "merged_prs"), ("open_ai", "open_prs"), ("pooled_ai", "pooled_prs"),
                 ("raw_merged_ai", "raw_merged_prs"), ("raw_open_ai", "raw_open_prs"),
                 ("raw_pooled_ai", "raw_pooled_prs")):
        tot[a + "_pct"] = pct(tot[a], tot[b])
    per_repo_rows.append(tot)

    cols = list(per_repo_rows[0].keys())
    cols = ["repo"] + [c for c in cols if c != "repo"]
    with open(os.path.join(OUT_DIR, "C-ai-per-repo.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(per_repo_rows)

    # ── (b) by year of PR creation ──────────────────────────────────────────
    raw_year = agg(flags, lambda r: r["created_at"][:4] or "unknown")
    ana_year = agg([r for r in flags if r["in_analysis_set"]], lambda r: r["created_at"][:4] or "unknown")

    year_rows = []
    for y in sorted(set(raw_year) | set(ana_year)):
        a, b = ana_year[y], raw_year[y]
        year_rows.append({
            "year": y,
            "merged_prs": a["merged_total"], "merged_ai": a["merged_ai"],
            "merged_ai_pct": pct(a["merged_ai"], a["merged_total"]),
            "open_prs": a["open_total"], "open_ai": a["open_ai"],
            "open_ai_pct": pct(a["open_ai"], a["open_total"]),
            "pooled_prs": a["pooled_total"], "pooled_ai": a["pooled_ai"],
            "pooled_ai_pct": pct(a["pooled_ai"], a["pooled_total"]),
            "raw_pooled_prs": b["pooled_total"], "raw_pooled_ai": b["pooled_ai"],
            "raw_pooled_ai_pct": pct(b["pooled_ai"], b["pooled_total"]),
        })
    with open(os.path.join(OUT_DIR, "C-ai-by-year.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(year_rows[0].keys()))
        w.writeheader()
        w.writerows(year_rows)

    # ── (c) tool x repository ───────────────────────────────────────────────
    tool_repo = defaultdict(lambda: defaultdict(int))
    tool_repo_ana = defaultdict(lambda: defaultdict(int))
    tool_first = {}
    for r in flags:
        for t in [x for x in r["ai_tools"].split("|") if x]:
            tool_repo[t][r["repo"]] += 1
            if r["in_analysis_set"]:
                tool_repo_ana[t][r["repo"]] += 1
            d = r["created_at"][:10]
            if d and (t not in tool_first or d < tool_first[t]):
                tool_first[t] = d

    trows = []
    for t in sorted(tool_repo, key=lambda x: -sum(tool_repo[x].values())):
        trows.append({
            "tool": t,
            "prs_raw": sum(tool_repo[t].values()),
            "prs_analysis": sum(tool_repo_ana[t].values()),
            "n_repos": len(tool_repo[t]),
            "first_pr_created": tool_first.get(t, ""),
            "first_ai_event": ev_first.get(t, ""),
            "ai_events": sum(ev_by_tool_year[t].values()),
            "events_by_year": "; ".join(f"{y}={n}" for y, n in sorted(ev_by_tool_year[t].items())),
            "per_repo": "; ".join(f"{k}={v}" for k, v in
                                  sorted(tool_repo[t].items(), key=lambda x: -x[1])),
        })
    with open(os.path.join(OUT_DIR, "C-ai-tool-by-repo.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trows[0].keys()))
        w.writeheader()
        w.writerows(trows)

    # ── console report ──────────────────────────────────────────────────────
    print("=" * 96)
    print("(a) AI review prevalence per repository — ANALYSIS corpus (Table 2 rows)")
    print("=" * 96)
    print(f"{'repository':26s} {'merged':>7} {'AI':>6} {'%':>7} | {'open':>6} {'AI':>5} {'%':>7} "
          f"| {'pooled':>7} {'AI':>6} {'%':>7}")
    for r in per_repo_rows:
        print(f"{r['repo']:26s} {r['merged_prs']:>7} {r['merged_ai']:>6} {r['merged_ai_pct']:>7.3f} | "
              f"{r['open_prs']:>6} {r['open_ai']:>5} {r['open_ai_pct']:>7.3f} | "
              f"{r['pooled_prs']:>7} {r['pooled_ai']:>6} {r['pooled_ai_pct']:>7.3f}")

    print()
    print("=" * 96)
    print("(a') same, RAW corpus (before the merged-PR inclusion filters)")
    print("=" * 96)
    print(f"{'repository':26s} {'merged':>7} {'AI':>6} {'%':>7} | {'open':>6} {'AI':>5} {'%':>7} "
          f"| {'pooled':>7} {'AI':>6} {'%':>7}")
    for r in per_repo_rows:
        print(f"{r['repo']:26s} {r['raw_merged_prs']:>7} {r['raw_merged_ai']:>6} {r['raw_merged_ai_pct']:>7.3f} | "
              f"{r['raw_open_prs']:>6} {r['raw_open_ai']:>5} {r['raw_open_ai_pct']:>7.3f} | "
              f"{r['raw_pooled_prs']:>7} {r['raw_pooled_ai']:>6} {r['raw_pooled_ai_pct']:>7.3f}")

    print()
    print("=" * 96)
    print("(b) AI review prevalence by year of PR creation")
    print("=" * 96)
    print(f"{'year':>6} {'merged':>8} {'AI':>6} {'%':>7} | {'open':>6} {'AI':>5} {'%':>7} "
          f"| {'pooled':>8} {'AI':>6} {'%':>7} || raw pooled {'n':>7} {'AI':>6} {'%':>7}")
    def p(v):
        return f"{v:7.3f}" if v is not None else "    n/a"

    for r in year_rows:
        print(f"{r['year']:>6} {r['merged_prs']:>8} {r['merged_ai']:>6} {p(r['merged_ai_pct'])} | "
              f"{r['open_prs']:>6} {r['open_ai']:>5} {p(r['open_ai_pct'])} | "
              f"{r['pooled_prs']:>8} {r['pooled_ai']:>6} {p(r['pooled_ai_pct'])} || "
              f"{'':11s}{r['raw_pooled_prs']:>7} {r['raw_pooled_ai']:>6} {p(r['raw_pooled_ai_pct'])}")

    print()
    print("=" * 96)
    print("(c) which AI tool appears in which repository (raw corpus, PR counts)")
    print("=" * 96)
    for r in trows:
        print(f"{r['tool']:34s} PRs={r['prs_raw']:>6} (analysis {r['prs_analysis']:>6}) "
              f"repos={r['n_repos']:>2} first PR created={r['first_pr_created']} "
              f"first AI event={r['first_ai_event'][:10]}")
        print(f"    {r['per_repo']}")
        print(f"    AI events={r['ai_events']}  by event year: {r['events_by_year']}")

    n_ai_auth = sum(r["ai_authored"] for r in flags)
    print(f"\nPRs opened BY an AI account (reported separately, not part of the "
          f"'AI review' definition): {n_ai_auth}")
    print(f"\nWrote C-ai-pr-flags.csv, C-ai-per-repo.csv, C-ai-by-year.csv, "
          f"C-ai-tool-by-repo.csv into {OUT_DIR}")


if __name__ == "__main__":
    main()
