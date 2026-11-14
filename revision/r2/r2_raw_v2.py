#!/usr/bin/env python3
"""Chapter 2, corpus v2: quantities that the shared chapter 3 PR table lacks.

The shared table (pr_table_v2.csv, r3_v2_prep.py) already holds the
characteristics over human events (clock D from t_rd), labels, author groups,
language, project group and reverts. This script computes only what is missing
there, and reuses the collector functions unchanged (imported, not a second
adapter):
  - r3_v2_prep: typename_map, activity_trace_from, max_gap_days;
  - review_iterations: reviewer_events, count_iterations, RULE_LITERAL / RULE_CODE;
  - taskF1_duration_language.participant_class (classification that uses the
    GitHub account type, corpus v2 login audit of 29.10.2026);
  - taskE1_common.thread_record (assistant threads, as in the ICT 2026 paper).

PR-level sample: the chapter 3 main sample (pr_table_v2: in_main = 1,
hait_filter = 1). The ready-for-review moment t_rd and the groups are taken
from pr_table_v2, not computed again.

Inputs (read only):
  $R3_ROOT/v2/pr_table_v2.csv  chapter 3 PR table (revision/r3_v2_prep.py);
                               default revision/output/r3_b40d;
  $R3_RAW_V2                   corpus v2 raw data (default data/raw_v2):
                               repos.csv and <owner>_<repo>/prs.jsonl.

Output in $R2_DIR/v2/ (default revision/output/r2/v2, git-ignored):
  r2_pr_extra_v2.csv   PR level: verdict of the first human review, trace check
                       by the state automaton, deviation category, t_r over all
                       accounts (without reviews submitted in draft, as in
                       pr_table_v2 r3_b40d), class of the first reviewer,
                       assistant participation, iterations under other rules,
                       gap over all accounts, gap after approval;
  r2_threads_v2.csv    threads started by an assistant, a human reviewer or a
                       deterministic bot (trigger baseline), PRs of 2025-2026;
  r2_ai_reviews_v2.csv assistant reviews (verdict), PRs of 2025-2026;
  r2_all_prs_v2.csv    all collected PRs (any state): creation year, assistant
                       participation, author group, for shares by year.

Data are only read. Run from the repository root:
  PYTHONDONTWRITEBYTECODE=1 nice .venv/bin/python revision/r2/r2_raw_v2.py
A trial run takes a comma-separated list of repositories and writes to
$R2_DIR/v2_trial:
  ... revision/r2/r2_raw_v2.py vitejs/vite,django/django
"""
import csv
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)                      # revision/
ROOT = os.path.dirname(REV)                      # repository root
RAW_V2 = os.environ.get("R3_RAW_V2") or os.path.join(ROOT, "data", "raw_v2")
# chapter 3 run r3_b40d: the PR table without human reviews submitted in draft (before t_rd)
R3_ROOT = os.environ.get("R3_ROOT") or os.path.join(REV, "output", "r3_b40d")
R3_RUN = os.path.basename(os.path.normpath(R3_ROOT))
PR_TABLE = os.path.join(R3_ROOT, "v2", "pr_table_v2.csv")
R2_DIR = os.environ.get("R2_DIR") or os.path.join(REV, "output", "r2")
OUT = os.path.join(R2_DIR, "v2")
sys.path.insert(0, REV)

import r3_v2_prep as prep  # noqa: E402
import taskE1_common as e1  # noqa: E402
from review_iterations import RULE_CODE, RULE_LITERAL, count_iterations, login_of, parse_dt, reviewer_events  # noqa: E402
from taskC_common import ai_tool_name, author_group, classify_account  # noqa: E402
from taskF1_duration_language import participant_class  # noqa: E402

csv.field_size_limit(10 ** 9)


def load_main():
    """(repo, number) -> pr_table_v2 row for the main sample."""
    keep = ("repo", "pr_number", "in_main", "hait_filter", "t_rd", "author_group", "created_at",
            "merged_at")
    out = {}
    with open(PR_TABLE, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["in_main"] == "1" and r["hait_filter"] == "1":
                out[(r["repo"], int(r["pr_number"]))] = {k: r[k] for k in keep}
    return out


def hrs(a, b):
    return None if (a is None or b is None) else (b - a).total_seconds() / 3600.0


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ") if t else ""


def ai_events(pr, cls):
    """Times of reviews and thread comments by the ai_assistant class (not the author)."""
    author = login_of(pr)
    out = []
    for r in (pr.get("reviews") or {}).get("nodes") or []:
        w = login_of(r)
        if w != author and cls(w) == "ai_assistant":
            t = parse_dt((r or {}).get("submittedAt"))
            if t:
                out.append((t, w, "review"))
    for th in (pr.get("reviewThreads") or {}).get("nodes") or []:
        for c in (((th or {}).get("comments") or {}).get("nodes") or []):
            w = login_of(c)
            if w != author and cls(w) == "ai_assistant":
                t = parse_dt((c or {}).get("createdAt"))
                if t:
                    out.append((t, w, "comment"))
    out.sort(key=lambda e: e[0])
    return out


def longest_gap(events):
    """(length in days, start) of the longest gap in an activity trace."""
    best, start = 0.0, None
    for i in range(len(events) - 1):
        g = (events[i + 1] - events[i]).total_seconds() / 86400.0
        if g > best:
            best, start = g, events[i]
    return best, start


def replay(pr, merged, cls):
    """Trace check by the state automaton (transition table, appendix B.2) over human events.

    Merge is defined only from the "Approved" state (or from stale with the
    "Approved" memory); the timer does not change the substate, so a trace is
    accepted if and only if there is a human approval, not by the author, before
    the merge. A move to "Returned for rework" happens only before the first
    approval (after it, a change request is internal transition 11).
    Returns (accepted, visits_returned, category, first_approval).
    """
    author = login_of(pr)
    hum, bot_appr, dismissed = [], False, False
    for r in (pr.get("reviews") or {}).get("nodes") or []:
        w = login_of(r)
        if w == author:
            continue
        t = parse_dt((r or {}).get("submittedAt"))
        st = (r or {}).get("state")
        c = cls(w)
        if c == "human":
            if st == "DISMISSED":
                dismissed = True
            if t:
                hum.append((t, st))
        elif st == "APPROVED" and t and t <= merged:
            bot_appr = True
    hum.sort(key=lambda e: e[0])
    appr = [t for t, st in hum if st == "APPROVED"]
    accepted = any(t <= merged for t in appr)
    first_appr = min(appr) if appr else None
    visits_ret = any(st == "CHANGES_REQUESTED" and (first_appr is None or t < first_appr)
                     for t, st in hum)
    if accepted:
        cat = ""
    elif bot_appr:
        cat = "bot_approval"
    elif not hum:
        cat = "no_human_review"
    elif dismissed:
        cat = "approval_dismissed"
    elif appr:
        cat = "approval_after_merge"
    else:
        cat = "comment_or_changes_only"
    return int(accepted), int(visits_ret), cat, first_appr


def trigger_hours(t, created, t_rd, commit_times):
    """Hours since the nearest preceding trigger: creation, leaving draft
    (t_rd, if not later than t) or the last revision not later than t."""
    cands = [created]
    if t_rd and t_rd <= t:
        cands.append(t_rd)
    cands += [d for d in commit_times if d <= t]
    cands = [c for c in cands if c and c <= t]
    return min(hrs(c, t) for c in cands) if cands else None


PR_FIELDS = ["repo", "pr_number", "first_h_verdict", "last_h_verdict", "accepted", "reject_category",
             "visits_returned", "has_h_changes_requested", "n_h_reviews",
             "t_r_all_h_D", "first_reviewer_class", "first_reactor_any_class",
             "u_ai", "n_ai_reviews", "n_ai_comments", "t_ai_h_D", "t_ai_first", "ai_tools",
             "n_it_all_lit", "n_it_all_code", "n_it_h_code",
             "gap_all_days_D", "gap_h_days_D", "gap_h_start_after_appr",
             "n_commits", "n_commits_no_user", "ci_window"]
TH_FIELDS = ["repo", "pr_number", "starter_class", "tool", "created_at", "hours_since_trigger",
             "is_resolved", "reacted", "commits_after", "apply_commit_after", "has_suggestion",
             "sugg_applied", "author_group"]
AR_FIELDS = ["repo", "pr_number", "tool", "state", "submitted_at"]
ALL_FIELDS = ["repo", "pr_number", "state", "created_at", "author_group", "u_ai"]


def main():
    out = OUT + ("_trial" if len(sys.argv) > 1 else "")
    os.makedirs(out, exist_ok=True)
    main_rows = load_main()
    with open(os.path.join(RAW_V2, "repos.csv"), encoding="utf-8") as f:
        repos = sorted(r["repo"] for r in csv.DictReader(f))
    if len(sys.argv) > 1:  # trial run: comma-separated repositories, output to a folder with the _trial suffix
        repos = sys.argv[1].split(",")
    f_pr = open(os.path.join(out, "r2_pr_extra_v2.csv"), "w", newline="", encoding="utf-8")
    f_th = open(os.path.join(out, "r2_threads_v2.csv"), "w", newline="", encoding="utf-8")
    f_ar = open(os.path.join(out, "r2_ai_reviews_v2.csv"), "w", newline="", encoding="utf-8")
    f_all = open(os.path.join(out, "r2_all_prs_v2.csv"), "w", newline="", encoding="utf-8")
    w_pr = csv.DictWriter(f_pr, PR_FIELDS)
    w_th = csv.DictWriter(f_th, TH_FIELDS)
    w_ar = csv.DictWriter(f_ar, AR_FIELDS)
    w_all = csv.DictWriter(f_all, ALL_FIELDS)
    for w in (w_pr, w_th, w_ar, w_all):
        w.writeheader()
    n_done = 0
    for repo in repos:
        seen = set()
        for pr in prep.iter_v2(RAW_V2, repo):
            num = int(pr["number"])
            if num in seen:
                continue
            seen.add(num)
            tm = prep.typename_map(pr)

            def cls(w, tm=tm):
                return participant_class(w, tm.get(w))
            a_login = login_of(pr)
            a_type = ((pr.get("author") or {}).get("__typename"))
            created = parse_dt(pr.get("createdAt"))
            aev = ai_events(pr, cls)
            w_all.writerow({"repo": repo, "pr_number": num, "state": pr.get("state"),
                            "created_at": pr.get("createdAt"),
                            "author_group": author_group(a_login, a_type), "u_ai": int(bool(aev))})
            key = (repo, num)
            if key not in main_rows:
                continue
            row = main_rows[key]
            merged = parse_dt(pr.get("mergedAt"))
            t_rd = parse_dt(row["t_rd"]) or created
            human = (lambda w, cls=cls: cls(w) == "human")
            ev_h = reviewer_events(pr, human)
            # draft rule for all accounts: reviews submitted in draft (before t_rd) are not
            # events of t_r or iterations, as in pr_table_v2 (r3_b40d)
            pr_d = prep.reviews_from(pr, t_rd)
            ev_all = reviewer_events(pr_d, lambda _w: True)
            accepted, visits_ret, cat, first_appr = replay(pr, merged, cls)
            # t_r over all accounts from t_rd (negative -> 0, as t_r_h_D)
            t_r_all = None
            if ev_all:
                t_r_all = max(0.0, hrs(t_rd, ev_all[0][0]))
            # first reaction by any account: a review or a thread comment
            reacts = [(t, w) for t, w, _ in ev_all]
            for th in (pr.get("reviewThreads") or {}).get("nodes") or []:
                for c in (((th or {}).get("comments") or {}).get("nodes") or []):
                    w = login_of(c)
                    t = parse_dt((c or {}).get("createdAt"))
                    if w != a_login and t:
                        reacts.append((t, w))
            reacts.sort(key=lambda e: e[0])
            tr_h = prep.activity_trace_from(pr, human, t_rd)
            tr_all = prep.activity_trace_from(pr, lambda _w: True, t_rd)
            g_h, g_start = longest_gap(tr_h) if tr_h else (None, None)
            g_start_dt = g_start.replace(tzinfo=timezone.utc) if g_start else None
            fa = first_appr.astimezone(timezone.utc) if first_appr else None
            commits = (pr.get("commits") or {}).get("nodes") or []
            n_no_user = sum(1 for c in commits
                            if not ((((c or {}).get("commit") or {}).get("author") or {}).get("user")))
            ai_rev = [e for e in aev if e[2] == "review"]
            tools = sorted({ai_tool_name(w) or w for _, w, _ in aev})
            w_pr.writerow({
                "repo": repo, "pr_number": num,
                "first_h_verdict": ev_h[0][2] if ev_h else "",
                "last_h_verdict": ev_h[-1][2] if ev_h else "",
                "accepted": accepted, "reject_category": cat, "visits_returned": visits_ret,
                "has_h_changes_requested": int(any(s == "CHANGES_REQUESTED" for _, _, s in ev_h)),
                "n_h_reviews": len(ev_h),
                "t_r_all_h_D": "" if t_r_all is None else round(t_r_all, 4),
                "first_reviewer_class": cls(ev_all[0][1]) if ev_all else "",
                "first_reactor_any_class": cls(reacts[0][1]) if reacts else "",
                "u_ai": int(bool(aev)), "n_ai_reviews": len(ai_rev),
                "n_ai_comments": len(aev) - len(ai_rev),
                "t_ai_h_D": "" if not aev else round(max(0.0, hrs(t_rd, aev[0][0])), 4),
                "t_ai_first": iso(aev[0][0]) if aev else "",
                "ai_tools": "|".join(tools),
                "n_it_all_lit": count_iterations(pr_d, lambda _w: True, rule=RULE_LITERAL),
                "n_it_all_code": count_iterations(pr_d, lambda _w: True, rule=RULE_CODE),
                "n_it_h_code": count_iterations(pr_d, human, rule=RULE_CODE),
                "gap_all_days_D": "" if not tr_all else round(prep.max_gap_days(tr_all), 5),
                "gap_h_days_D": "" if g_h is None else round(g_h, 5),
                "gap_h_start_after_appr": int(bool(fa and g_start_dt and g_start_dt >= fa.replace(microsecond=0))),
                "n_commits": len(commits), "n_commits_no_user": n_no_user,
                "ci_window": int(bool((pr.get("_meta") or {}).get("ci_window"))),
            })
            # assistant threads and reviews: PRs created in 2025-2026
            if created and created.year >= 2025:
                orig_cl = e1.classify_login
                e1.classify_login = lambda w, tm=tm: classify_account(w, tm.get(w))
                try:
                    commit_pairs = e1.pr_commits(pr)
                    ctimes = [d for d, _ in commit_pairs]
                    nodes = (pr.get("reviewThreads") or {}).get("nodes") or []
                    ai_threads = []
                    for th in nodes:
                        cs = (th.get("comments") or {}).get("nodes") or []
                        if not cs:
                            continue
                        w0 = login_of(cs[0])
                        if w0 == a_login:
                            continue
                        c0 = cls(w0)
                        if c0 == "ai_assistant":
                            ai_threads.append(th)
                        t0 = parse_dt(cs[0].get("createdAt"))
                        if c0 in ("human", "other_bot") and t0:
                            th_h = trigger_hours(t0, created, t_rd, ctimes)
                            w_th.writerow({"repo": repo, "pr_number": num, "starter_class": c0,
                                           "tool": "", "created_at": cs[0].get("createdAt"),
                                           "hours_since_trigger": "" if th_h is None else round(th_h, 4),
                                           "author_group": row["author_group"]})
                    ai_threads.sort(key=lambda t: (((t.get("comments") or {}).get("nodes") or [{}])[0]
                                                   .get("createdAt") or ""))
                    for i, th in enumerate(ai_threads):
                        rec = e1.thread_record(repo, pr, th, i, len(ai_threads), set(), commit_pairs)
                        t0 = parse_dt(rec["created_at"])
                        th_h = trigger_hours(t0, created, t_rd, ctimes) if t0 else None
                        w_th.writerow({
                            "repo": repo, "pr_number": num, "starter_class": "ai_assistant",
                            "tool": rec["tool"], "created_at": rec["created_at"],
                            "hours_since_trigger": "" if th_h is None else round(th_h, 4),
                            "is_resolved": rec["is_resolved"], "reacted": rec["reacted"],
                            "commits_after": rec["commits_after"],
                            "apply_commit_after": rec["apply_commit_after"],
                            "has_suggestion": rec["has_suggestion"],
                            "sugg_applied": int(rec["has_suggestion"] and rec["apply_commit_after"]
                                                and row["author_group"] == "human"),
                            "author_group": row["author_group"]})
                finally:
                    e1.classify_login = orig_cl
                for r in (pr.get("reviews") or {}).get("nodes") or []:
                    w = login_of(r)
                    if w != a_login and cls(w) == "ai_assistant":
                        w_ar.writerow({"repo": repo, "pr_number": num, "tool": ai_tool_name(w) or w,
                                       "state": r.get("state") or "",
                                       "submitted_at": r.get("submittedAt") or ""})
            n_done += 1
        print(repo, len(seen), n_done, datetime.now().strftime("%H:%M:%S"), flush=True)
    for f in (f_pr, f_th, f_ar, f_all):
        f.close()


if __name__ == "__main__":
    main()
