"""Build the (PR x candidate) pair matrix for one repository.

STRICT features use only history strictly before the PR's creation time t0.
"""
import math
import sys
from pathlib import Path

import pandas as pd

from rank.events import parse_ts, positives, review_events, is_bot
from rank.history import AuthorHistory, ReviewerHistory
from rank.split import assign_splits

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent / "output"

REPOS = [
    "angular_angular", "discourse_discourse", "django_django", "facebook_react",
    "grafana_grafana", "mattermost_mattermost", "microsoft_vscode", "nodejs_node",
    "strapi_strapi", "supabase_supabase", "vercel_next.js", "woocommerce_woocommerce",
]

STRICT_FEATURES = [
    "act_7d", "act_30d", "act_90d", "act_total", "days_since_last",
    "workload", "act_30d_share",
    "lat_med_h", "fast_24h",
    "familiarity", "author_prior_prs", "author_med_first_review_h",
    "hour", "weekday",
]
CONTENT_FEATURES = ["additions", "deletions", "changed_files", "is_draft"]


def _first_review_at(pr):
    ts = [parse_ts(r.get("submittedAt")) for r in (pr.get("reviews") or {}).get("nodes") or []
          if (r.get("author") or {}).get("login") and (r.get("author") or {}).get("login") != (pr.get("author") or {}).get("login")
          and not is_bot((r.get("author") or {}).get("login"))]
    ts += [parse_ts(c.get("createdAt")) for th in (pr.get("reviewThreads") or {}).get("nodes") or []
           for c in (th.get("comments") or {}).get("nodes") or []
           if (c.get("author") or {}).get("login") and (c.get("author") or {}).get("login") != (pr.get("author") or {}).get("login")
           and not is_bot((c.get("author") or {}).get("login"))]
    ts = [t for t in ts if t]
    return min(ts) if ts else None


def build_repo(repo, prs, warmup_days=90, cand_window_days=180, log=None):
    prs = [p for p in prs if p.get("createdAt")]
    prs.sort(key=lambda p: p["createdAt"])
    for p in prs:
        p["_created"] = parse_ts(p["createdAt"])
        p["first_review_at"] = _first_review_at(p)
    events = review_events(prs)
    first_by_pr = {}
    for e in events:
        first_by_pr.setdefault(e["pr"], {})[e["reviewer"]] = e["first_at"]
    rh = ReviewerHistory(events)
    ah = AuthorHistory(prs)
    splits = assign_splits([p["_created"] for p in prs], warmup_days=warmup_days)

    pair_rows, pr_rows = [], []
    total_active_30d_cache = {}
    for i, (p, split) in enumerate(zip(prs, splits)):
        t0 = p["_created"]
        author = (p.get("author") or {}).get("login") or ""
        pos = positives(p)
        cands = [c for c in rh.candidates(t0, cand_window_days) if c != author]
        pr_rows.append({
            "repo": repo, "pr": p["number"], "created": t0, "split": split, "author": author,
            "n_candidates": len(cands), "n_positives": len(pos),
            "any_positive_in_pool": int(bool(pos & set(cands))),
            "all_positives_in_pool": int(bool(pos) and pos <= set(cands)),
            "first_review_h": ((p["first_review_at"] - t0).total_seconds() / 3600.0) if p["first_review_at"] else math.nan,
        })
        if split == "warmup" or not cands:
            continue
        a_prior = ah.prior_prs(author, t0)
        a_med = ah.median_hours_to_first_review(author, t0)
        # repo-level activity in the last 30 days, for the share feature
        repo_act_30d = sum(rh.activity(c, t0, 30) for c in cands) or 1
        for c in cands:
            act30 = rh.activity(c, t0, 30)
            fa = first_by_pr.get(p["number"], {}).get(c)
            pair_rows.append({
                "repo": repo, "pr": p["number"], "candidate": c, "label": int(c in pos), "split": split,
                "realized_lat_h": ((fa - t0).total_seconds() / 3600.0) if fa else math.nan,
                "act_7d": rh.activity(c, t0, 7), "act_30d": act30, "act_90d": rh.activity(c, t0, 90),
                "act_total": rh.total_activity(c, t0), "days_since_last": rh.days_since_last(c, t0),
                "workload": rh.workload(c, t0), "act_30d_share": act30 / repo_act_30d,
                "lat_med_h": rh.latency_median_hours(c, t0, 90), "fast_24h": rh.fast_share(c, t0, 90, 24),
                "familiarity": rh.familiarity(c, author, t0),
                "author_prior_prs": a_prior, "author_med_first_review_h": a_med,
                "hour": t0.hour, "weekday": t0.weekday(),
                "additions": p.get("additions", 0), "deletions": p.get("deletions", 0),
                "changed_files": p.get("changedFiles", 0), "is_draft": int(bool(p.get("isDraft"))),
            })
        if log and i % 1000 == 0:
            print(f"  {repo}: {i}/{len(prs)} PRs, {len(pair_rows)} pairs", file=log, flush=True)
    return pd.DataFrame(pair_rows), pd.DataFrame(pr_rows)


def main(repos=None):
    import json, time
    OUT_DIR.mkdir(exist_ok=True)
    for repo in repos or REPOS:
        t = time.time()
        prs = json.load(open(RAW_DIR / repo / "merged_prs.json"))
        pairs, prtab = build_repo(repo, prs, log=sys.stdout)
        pairs.to_parquet(OUT_DIR / f"pairs_{repo}.parquet", index=False)
        prtab.to_parquet(OUT_DIR / f"prs_{repo}.parquet", index=False)
        print(f"{repo}: {len(prtab)} PRs, {len(pairs)} pairs, {time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
