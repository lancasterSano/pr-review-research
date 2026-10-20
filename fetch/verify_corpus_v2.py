#!/usr/bin/env python3
"""
Verify one collected corpus v2 repository.

Checks:
  - count in prs.jsonl vs a fresh GitHub search (is:pr is:closed, and is:merged) in the window;
  - duplicates, dates outside the window, max gap between consecutive createdAt, PRs per month;
  - field fill rates (READY_FOR_REVIEW, REVIEW_REQUESTED, CI, ...) and completeness of
    nested connections (nodes == totalCount);
  - cost and speed from checkpoint.json and fetch.log;
  - optional regression check against the frozen HAIT data (data/raw/<dir>/merged_prs.json):
    reviews and commits per PR on the overlap.

Usage:
    .venv/bin/python fetch/verify_corpus_v2.py --repo django/django --compare-old
Writes data/raw_v2/<owner>_<name>/verify.json.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_prs_v2 as fp  # noqa: E402

OLD_RAW = os.path.join(fp.SCRIPTS_DIR, "data", "raw")

SEARCH_Q = """query($q: String!) { rateLimit { cost remaining resetAt limit }
  search(query: $q, type: ISSUE, first: 0) { issueCount } }"""


def load_prs(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def timeline_types(pr):
    # GraphQL may return null nodes alongside partial-data errors; count them separately.
    return Counter(n.get("__typename") if n else "<null>"
                   for n in (pr.get("timelineItems") or {}).get("nodes") or [])


def share(n, d):
    return round(n / d, 4) if d else None


def complete(conn, filtered=False):
    """No page left. For timelineItems with itemTypes, totalCount counts ALL event types
    (verified 14.10.2026), so only hasNextPage is meaningful."""
    if conn is None:
        return True
    if (conn.get("pageInfo") or {}).get("hasNextPage"):
        return False
    return filtered or len(conn.get("nodes", [])) >= (conn.get("totalCount") or 0)


def human(login_obj):
    return login_obj and login_obj.get("__typename") != "Bot" and \
        not (login_obj.get("login") or "").endswith("[bot]")


def page_log_stats(log_path):
    """Seconds per page and points per page from fetch.log lines."""
    pat = re.compile(r"page (\d+) size=(\d+) ci=(\w) \+(\d+) total=\d+ .* cost=(\d+)(?: pts=(\d+))? "
                     r"remaining=\S+ ([\d.]+)s")
    rows = []
    if os.path.exists(log_path):
        for line in open(log_path):
            m = pat.search(line)
            if m:
                rows.append({"page": int(m.group(1)), "size": int(m.group(2)), "ci": m.group(3) == "y",
                             "added": int(m.group(4)), "cost": int(m.group(5)),
                             "pts": int(m.group(6) or m.group(5)), "secs": float(m.group(7))})
    out = {}
    for ci in (True, False):
        sub = [r for r in rows if r["ci"] == ci]
        added = sum(r["added"] for r in sub)
        if sub and added:
            out["ci" if ci else "no_ci"] = {
                "pages": len(sub), "prs": added,
                "mean_page_cost_size25": round(sum(r["cost"] for r in sub if r["size"] == 25) /
                                               max(1, sum(1 for r in sub if r["size"] == 25)), 1),
                "points_per_pr_incl_followups": round(sum(r["pts"] for r in sub) / added, 3),
                "secs_per_pr_incl_followups": round(sum(r["secs"] for r in sub) / added, 3),
            }
    out["pages_size10"] = sum(1 for r in rows if r["size"] == 10)
    waits = 0.0
    if os.path.exists(log_path):
        for line in open(log_path):
            m = re.search(r"wait (\d+)s: rate limit", line)
            if m:
                waits += float(m.group(1))
    out["rate_limit_wait_s"] = waits
    return out


def verify(repo, raw_dir=fp.RAW_V2_DIR, compare_old=False, client=None):
    owner, name = repo.split("/")
    d = os.path.join(raw_dir, "%s_%s" % (owner, name))
    prs = load_prs(os.path.join(d, "prs.jsonl"))
    cp = json.load(open(os.path.join(d, "checkpoint.json")))
    res = {"repo": repo, "status": cp["status"], "collected": len(prs)}

    nums = [p["number"] for p in prs]
    res["duplicates"] = len(nums) - len(set(nums))
    res["outside_window"] = sum(1 for p in prs if fp.window_position(p["createdAt"]) != "in")
    res["states"] = dict(Counter(p["state"] for p in prs))

    created = sorted(fp.parse_dt(p["createdAt"]) for p in prs)
    if created:
        gaps = [(b - a).total_seconds() / 86400 for a, b in zip(created, created[1:])]
        res["first_created"] = fp.iso(created[0])
        res["last_created"] = fp.iso(created[-1])
        res["head_gap_days"] = round((created[0] - fp.parse_dt(fp.WINDOW_START)).total_seconds() / 86400, 2)
        res["tail_gap_days"] = round((fp.parse_dt(fp.WINDOW_END) - created[-1]).total_seconds() / 86400, 2)
        res["max_gap_days"] = round(max(gaps), 2) if gaps else 0
        months = Counter(c.strftime("%Y-%m") for c in created)
        res["per_month"] = dict(sorted(months.items()))
        res["months_with_zero"] = [m for m in month_range(fp.WINDOW_START, fp.WINDOW_END) if m not in months]

    # search counts
    if client is not None:
        q_closed = fp.search_query_closed(owner, name)
        q_merged = q_closed.replace("is:closed", "is:merged")
        res["search_closed"] = client.execute(SEARCH_Q, {"q": q_closed})["search"]["issueCount"]
        res["search_merged"] = client.execute(SEARCH_Q, {"q": q_merged})["search"]["issueCount"]
        res["diff_closed_vs_search"] = len(prs) - res["search_closed"]
        res["diff_merged_vs_search"] = res["states"].get("MERGED", 0) - res["search_merged"]

    # fill rates
    n = len(prs)
    tt = [timeline_types(p) for p in prs]
    merged = [p for p in prs if p["state"] == "MERGED"]
    ci_window = [p for p in prs if p["_meta"]["ci_window"]]
    ci_with_suites = [p for p in ci_window if any(
        (c["commit"].get("checkSuites") or {}).get("totalCount", 0) > 0 or
        ((c["commit"].get("status") or {}).get("contexts"))
        for c in (p.get("ciCommits") or {}).get("nodes", []))]
    res["fill"] = {
        "ready_for_review": share(sum(1 for t in tt if t["ReadyForReviewEvent"]), n),
        "convert_to_draft": share(sum(1 for t in tt if t["ConvertToDraftEvent"]), n),
        "review_requested": share(sum(1 for t in tt if t["ReviewRequestedEvent"]), n),
        "head_ref_force_pushed": share(sum(1 for t in tt if t["HeadRefForcePushedEvent"]), n),
        "issue_comment": share(sum(1 for t in tt if t["IssueComment"]), n),
        "merged_event_among_merged": share(sum(1 for p, t in zip(prs, tt) if p["state"] == "MERGED" and t["MergedEvent"]), len(merged)),
        "closed_event": share(sum(1 for t in tt if t["ClosedEvent"]), n),
        "merge_queue": share(sum(1 for t in tt if t["AddedToMergeQueueEvent"]), n),
        "has_review": share(sum(1 for p in prs if p["reviews"]["totalCount"] > 0), n),
        "has_human_review_by_other": share(sum(1 for p in prs if any(
            human(r.get("author")) and (r.get("author") or {}).get("login") != (p.get("author") or {}).get("login")
            for r in p["reviews"]["nodes"])), n),
        "merged_has_review": share(sum(1 for p in merged if p["reviews"]["totalCount"] > 0), len(merged)),
        "author_bot": share(sum(1 for p in prs if (p.get("author") or {}).get("__typename") == "Bot"), n),
        "is_draft_now": share(sum(1 for p in prs if p.get("isDraft")), n),
        "ci_window_share": share(len(ci_window), n),
        "ci_fetched_in_ci_window": share(sum(1 for p in ci_window if p["_meta"]["ci_fetched"]), len(ci_window)),
        "ci_any_suite_or_status_in_ci_window": share(len(ci_with_suites), len(ci_window)),
        "ci_head_runs_in_ci_window": share(sum(1 for p in ci_window if any(
            (s.get("checkRuns") or {}).get("totalCount", 0) > 0
            for c in (p.get("ciHead") or {}).get("nodes", [])
            for s in (c["commit"].get("checkSuites") or {}).get("nodes", []))), len(ci_window)),
        "commit_committedDate": share(sum(1 for p in prs if all(c["commit"].get("committedDate") for c in p["commits"]["nodes"])), n),
    }
    res["complete_nested"] = {k: share(sum(1 for p in prs if complete(p.get(k), k == "timelineItems")), n)
                              for k in fp.NESTED}
    res["commits_over_250"] = sum(1 for p in prs if p["commits"]["totalCount"] > 250)
    res["complete_thread_comments"] = share(sum(1 for p in prs if all(
        complete(t.get("comments")) for t in p["reviewThreads"]["nodes"])), n)

    # cost and speed
    hours = cp["elapsed_s"] / 3600 if cp["elapsed_s"] else None
    res["cost"] = {
        "pages": cp["pages"], "points": cp["points"], "followup_queries": cp["followup_queries"],
        "elapsed_s": round(cp["elapsed_s"], 1),
        "points_per_pr": round(cp["points"] / n, 3) if n else None,
        "prs_per_hour_wall": round(n / hours) if hours else None,
        "prs_per_hour_by_points": round(5000 / (cp["points"] / n)) if n and cp["points"] else None,
        "skipped_after_window": cp["skipped_after_window"],
    }
    res["cost"].update(page_log_stats(os.path.join(d, "fetch.log")))

    if compare_old:
        res["regression_vs_hait"] = compare_with_old(prs, "%s_%s" % (owner, name))

    fp.atomic_write_json(os.path.join(d, "verify.json"), res)
    return res


def month_range(start, end):
    s, e = fp.parse_dt(start), fp.parse_dt(end)
    y, m = s.year, s.month
    out = []
    while (y, m) <= (e.year, e.month):
        out.append("%04d-%02d" % (y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def compare_with_old(prs, old_dir):
    path = os.path.join(OLD_RAW, old_dir, "merged_prs.json")
    if not os.path.exists(path):
        return {"error": "no old data at %s" % path}
    old = {p["number"]: p for p in json.load(open(path))}
    new = {p["number"]: p for p in prs if p["state"] == "MERGED"}
    both = sorted(set(old) & set(new))
    out = {"overlap": len(both)}
    for key in ("reviews", "commits", "reviewThreads"):
        same_total = sum(1 for k in both if old[k][key]["totalCount"] == new[k][key]["totalCount"])
        new_more = sum(1 for k in both if new[k][key]["totalCount"] > old[k][key]["totalCount"])
        out[key] = {"same_totalCount": share(same_total, len(both)), "new_greater": new_more,
                    "new_smaller": len(both) - same_total - new_more,
                    "old_truncated_nodes": sum(1 for k in both if len(old[k][key]["nodes"]) < old[k][key]["totalCount"]),
                    "new_truncated_nodes": sum(1 for k in both if len(new[k][key]["nodes"]) < new[k][key]["totalCount"])}
    out["additions_deletions_same"] = share(sum(1 for k in both if (old[k]["additions"], old[k]["deletions"]) ==
                                                (new[k]["additions"], new[k]["deletions"])), len(both))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--raw-dir", default=fp.RAW_V2_DIR)
    ap.add_argument("--compare-old", action="store_true")
    ap.add_argument("--no-search", action="store_true")
    args = ap.parse_args()
    client = None if args.no_search else fp.GitHubClient(fp.load_token(log=lambda m: None))
    res = verify(args.repo, args.raw_dir, args.compare_old, client)
    print(json.dumps({k: v for k, v in res.items() if k != "per_month"}, indent=1, ensure_ascii=False))
    print("per_month:", " ".join("%s:%d" % kv for kv in res.get("per_month", {}).items()))


if __name__ == "__main__":
    main()
