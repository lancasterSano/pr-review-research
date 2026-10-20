#!/usr/bin/env python3
"""
Status of the corpus v2 collection: progress per repository, process liveness, ETA.

Usage:
    .venv/bin/python fetch/corpus_v2_status.py          # table + ETA
    .venv/bin/python fetch/corpus_v2_status.py --tail 20

Expected totals come from repos.csv (`search_closed_in_window` and its part created
on/after CI_SINCE, GitHub search at metadata time). For repositories that search does
not index (facebook/react returns 0) a rough estimate is used.

ETA: remaining PRs are split into the CI part (pages with check suites, slower and
dearer) and the older part. For each part the per-PR work time and points (including
follow-up queries) are measured from all fetch.log page lines. Duration =
max(work time, points / 5000 per hour), because the collector waits when the hourly
GraphQL budget is spent.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_prs_v2 as fp  # noqa: E402

# search does not index these repositories; rough closed-in-window estimates
EXPECTED_FALLBACK = {"facebook/react": (3000, 1200)}   # (whole window, CI part)
# defaults before any page of a kind was seen (trial K0, 14.10.2026): (secs, points) per PR
DEFAULT_RATES = {True: (0.75, 1.15), False: (0.30, 0.62)}
POINTS_PER_HOUR = 5000
PAGE_RE = re.compile(r"page \d+ size=\d+ ci=(\w) \+(\d+) total=\d+ .* cost=(\d+)(?: pts=(\d+))? "
                     r"remaining=\S+ ([\d.]+)s")


def measured_rates(raw_dir):
    """Per-PR (seconds, points) for CI and non-CI pages across all fetch.log files."""
    acc = {True: [0, 0.0, 0], False: [0, 0.0, 0]}   # prs, secs, points
    for d in os.listdir(raw_dir):
        path = os.path.join(raw_dir, d, "fetch.log")
        if not os.path.isfile(path):
            continue
        for line in open(path):
            m = PAGE_RE.search(line)
            if not m or int(m.group(2)) == 0:
                continue
            a = acc[m.group(1) == "y"]
            a[0] += int(m.group(2))
            a[1] += float(m.group(5))
            a[2] += int(m.group(4) or m.group(3))
    rates = {}
    for ci, (n, secs, pts) in acc.items():
        rates[ci] = (secs / n, pts / n, n) if n >= 200 else DEFAULT_RATES[ci] + (0,)
    return rates


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pid_status(raw_dir):
    path = os.path.join(raw_dir, "run.pid")
    if not os.path.exists(path):
        return None, False
    try:
        pid = int(open(path).read().strip())
        os.kill(pid, 0)
        return pid, True
    except (ValueError, OSError):
        return None, False


def tail(path, n):
    if not os.path.exists(path):
        return []
    out = subprocess.run(["tail", "-n", str(n), path], capture_output=True, text=True)
    return out.stdout.rstrip("\n").splitlines()


def fmt_h(hours):
    if hours is None:
        return "?"
    return "%dг %02dхв" % (int(hours), int(round((hours - int(hours)) * 60)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=fp.RAW_V2_DIR)
    ap.add_argument("--tail", type=int, default=5)
    args = ap.parse_args()
    raw = args.raw_dir

    repos = read_csv(os.path.join(raw, "repos.csv"))
    progress = {r["repo"]: r for r in read_csv(os.path.join(raw, "progress.csv"))}
    pid, alive = pid_status(raw)

    print("Процес: %s" % ("працює, PID %d" % pid if alive else "не працює"))
    if os.path.exists(os.path.join(raw, "STOP")):
        print("Файл STOP існує: збирач зупиниться після поточної сторінки")
    print()
    print("%-3s %-34s %-7s %-9s %8s %8s %6s  %-10s %s"
          % ("#", "repo", "tier", "status", "зібрано", "очік.", "%", "до дати", "год"))

    tot_expected = tot_done_prs = 0
    remaining = 0
    rem_ci = rem_old = 0
    for r in repos:
        repo = r["repo"]
        exp = int(r.get("search_closed_in_window") or 0)
        exp_ci = int(r.get("search_closed_ci_part") or 0)
        if not exp and repo in EXPECTED_FALLBACK:
            exp, exp_ci = EXPECTED_FALLBACK[repo]
        if not exp_ci:
            exp_ci = exp // 2
        p = progress.get(repo, {})
        status = p.get("status") or "queued"
        got = int(p.get("collected") or 0)
        el = float(p.get("elapsed_s") or 0)
        tot_expected += exp
        tot_done_prs += got
        if status != "done":
            remaining += max(exp - got, 0)
            last = p.get("last_created") or ""
            if not last or last >= fp.CI_SINCE:
                rem_ci += max(exp_ci - got, 0)
                rem_old += exp - exp_ci
            else:
                rem_old += max(exp - got, 0)
        pct = "%.0f" % (100.0 * got / exp) if exp else "-"
        print("%-3s %-34s %-7s %-9s %8d %8d %6s  %-10s %.2f"
              % (r["order"], repo, r["tier"], status, got, exp, pct,
                 (p.get("last_created") or "")[:10], el / 3600))

    rates = measured_rates(raw)
    work_h = (rem_ci * rates[True][0] + rem_old * rates[False][0]) / 3600
    points = rem_ci * rates[True][1] + rem_old * rates[False][1]
    eta_h = max(work_h, points / POINTS_PER_HOUR)
    print()
    print("Разом: зібрано %d з ~%d (%.1f%%); залишилось ~%d запитів (CI-частина %d, старіша %d)"
          % (tot_done_prs, tot_expected, 100.0 * tot_done_prs / tot_expected if tot_expected else 0,
             remaining, rem_ci, rem_old))
    for ci, label in ((True, "CI"), (False, "без CI")):
        sec, pts, n = rates[ci]
        print("  %s: %.2f с і %.2f бала на запит (%s)"
              % (label, sec, pts, "виміряно на %d запитах" % n if n else "типове значення"))
    if remaining:
        finish = datetime.now(timezone.utc) + timedelta(hours=eta_h)
        print("ETA: %s (робота %s, ліміт балів %s) → ~%s UTC (~%s за Києвом)"
              % (fmt_h(eta_h), fmt_h(work_h), fmt_h(points / POINTS_PER_HOUR),
                 finish.strftime("%Y-%m-%d %H:%M"),
                 (finish + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")))
    errors = [r for r in progress.values() if r.get("status") == "error"]
    for e in errors:
        print("ПОМИЛКА %s: %s" % (e["repo"], e.get("message")))

    for name in ("run.log",):
        lines = tail(os.path.join(raw, name), args.tail)
        if lines:
            print("\n── %s (останні %d) ──" % (name, len(lines)))
            print("\n".join(lines))


if __name__ == "__main__":
    main()
