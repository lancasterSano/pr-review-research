#!/usr/bin/env python3
"""
Corpus v2 orchestrator: collects 40 repositories sequentially with fetch_prs_v2.

Collection order (tiers): K0 trial (eslint, django) -> P1 new repositories from small
to large -> R12 re-collection of the remaining HAIT 2026 repositories -> P2.
One process, one token, resumable.

Before collecting, it fetches repository metadata (GitHub language byte distribution,
archive/mirror flags, search count of closed PRs in the window) and writes
data/raw_v2/repos.csv plus data/raw_v2/<owner>_<name>/repo_meta.json.

Usage:
    .venv/bin/python fetch/run_corpus_v2.py                 # everything
    .venv/bin/python fetch/run_corpus_v2.py --only eslint/eslint,django/django
    .venv/bin/python fetch/run_corpus_v2.py --meta-only
    .venv/bin/python fetch/run_corpus_v2.py --list

Stop gracefully: `touch data/raw_v2/STOP` (finishes the current page)
or `kill $(cat data/raw_v2/run.pid)` (SIGTERM, same effect).
Resume: run the same command again; finished repositories are skipped.
"""

import argparse
import csv
import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_prs_v2 as fp  # noqa: E402

COMPANY, COMMUNITY = "company", "community"

# Who governs the project, by owner type only (no owner names). Written to
# repos.csv as `group_rationale`; `group` is the two-way split used in the analysis.
OWNER_COMPANY, OWNER_FOUNDATION, OWNER_COMMUNITY = "company", "foundation", "community"

# (repo, tier, group, owner kind, hait_language).
REPO_PLAN = [
    # K0 trial
    ("eslint/eslint", "K0+P1", COMMUNITY, OWNER_FOUNDATION, ""),
    ("django/django", "K0+R12", COMMUNITY, OWNER_FOUNDATION, "Python"),
    # Step 1: P1, small to large
    ("matomo-org/matomo", "P1", COMPANY, OWNER_COMPANY, ""),
    ("mautic/mautic", "P1", COMMUNITY, OWNER_COMMUNITY, ""),
    ("mozilla/pdf.js", "P1", COMPANY, OWNER_COMPANY, ""),
    ("PrestaShop/PrestaShop", "P1", COMPANY, OWNER_COMPANY, ""),
    ("vitejs/vite", "P1", COMMUNITY, OWNER_COMMUNITY, ""),
    ("scikit-learn/scikit-learn", "P1", COMMUNITY, OWNER_FOUNDATION, ""),
    ("saleor/saleor", "P1", COMPANY, OWNER_COMPANY, ""),
    ("prometheus/prometheus", "P1", COMMUNITY, OWNER_FOUNDATION, ""),
    ("chatwoot/chatwoot", "P1", COMPANY, OWNER_COMPANY, ""),
    ("decidim/decidim", "P1", COMMUNITY, OWNER_FOUNDATION, ""),
    ("go-gitea/gitea", "P1", COMMUNITY, OWNER_COMMUNITY, ""),
    ("Homebrew/brew", "P1", COMMUNITY, OWNER_COMMUNITY, ""),
    ("glpi-project/glpi", "P1", COMPANY, OWNER_COMPANY, ""),
    ("hashicorp/terraform-provider-aws", "P1", COMPANY, OWNER_COMPANY, ""),
    ("huggingface/transformers", "P1", COMPANY, OWNER_COMPANY, ""),
    ("opf/openproject", "P1", COMPANY, OWNER_COMPANY, ""),
    ("WordPress/gutenberg", "P1", COMMUNITY, OWNER_COMMUNITY, ""),
    ("cilium/cilium", "P1", COMPANY, OWNER_COMPANY, ""),
    ("apache/airflow", "P1", COMMUNITY, OWNER_FOUNDATION, ""),
    # Step 2: re-collection of the HAIT repositories (django is collected in K0)
    ("strapi/strapi", "R12", COMPANY, OWNER_COMPANY, "TypeScript"),
    ("facebook/react", "R12", COMPANY, OWNER_COMPANY, "JavaScript"),
    ("angular/angular", "R12", COMPANY, OWNER_COMPANY, "TypeScript"),
    ("nodejs/node", "R12", COMMUNITY, OWNER_FOUNDATION, "JavaScript/C++"),
    ("mattermost/mattermost", "R12", COMPANY, OWNER_COMPANY, "Go"),
    ("woocommerce/woocommerce", "R12", COMPANY, OWNER_COMPANY, "PHP"),
    ("supabase/supabase", "R12", COMPANY, OWNER_COMPANY, "TypeScript"),
    ("vercel/next.js", "R12", COMPANY, OWNER_COMPANY, "TypeScript"),
    ("discourse/discourse", "R12", COMPANY, OWNER_COMPANY, "Ruby"),
    ("grafana/grafana", "R12", COMPANY, OWNER_COMPANY, "Go"),
    ("microsoft/vscode", "R12", COMPANY, OWNER_COMPANY, "TypeScript"),
    # Step 3: P2
    ("getsentry/relay", "P2", COMPANY, OWNER_COMPANY, ""),
    ("vectordotdev/vector", "P2", COMPANY, OWNER_COMPANY, ""),
    ("apache/iceberg", "P2", COMMUNITY, OWNER_FOUNDATION, ""),
    ("apache/kafka", "P2", COMMUNITY, OWNER_FOUNDATION, ""),
    ("bevyengine/bevy", "P2", COMMUNITY, OWNER_FOUNDATION, ""),
    ("quarkusio/quarkus", "P2", COMPANY, OWNER_COMPANY, ""),
    ("keycloak/keycloak", "P2", COMPANY, OWNER_COMPANY, ""),
    ("servo/servo", "P2", COMMUNITY, OWNER_FOUNDATION, ""),
]

REPOS_CSV_FIELDS = ["order", "repo", "tier", "group", "group_rationale", "language",
                    "primary_share", "mixed", "language_shares", "hait_language",
                    "github_primary_language", "search_closed_in_window", "search_closed_ci_part", "is_archived",
                    "is_mirror", "owner_type", "meta_fetched_at"]

STOP = {"flag": False}


def repo_dir(raw_dir, repo):
    owner, name = repo.split("/")
    return os.path.join(raw_dir, "%s_%s" % (owner, name))


def log(msg):
    print("%s [run] %s" % (fp.iso(fp.utcnow()), msg), flush=True)


# ── PID lock and stop handling ──────────────────────────────────────────────

def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def acquire_pid(raw_dir):
    path = os.path.join(raw_dir, "run.pid")
    if os.path.exists(path):
        try:
            old = int(open(path).read().strip())
        except ValueError:
            old = None
        if old and old != os.getpid() and pid_alive(old):
            raise SystemExit("another collector is running (PID %d); refusing to start" % old)
    with open(path, "w") as f:
        f.write(str(os.getpid()))
    return path


def release_pid(path):
    try:
        if int(open(path).read().strip()) == os.getpid():
            os.remove(path)
    except (OSError, ValueError):
        pass


def install_signal_handlers():
    def handler(signum, frame):
        if STOP["flag"]:
            log("second signal %d: exiting now" % signum)
            raise SystemExit(1)
        STOP["flag"] = True
        log("signal %d: will stop after the current page" % signum)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def stop_requested(raw_dir):
    return STOP["flag"] or os.path.exists(os.path.join(raw_dir, "STOP"))


# ── Metadata ────────────────────────────────────────────────────────────────

def format_shares(shares):
    return "|".join("%s:%.4f" % (n, s) for n, _, s in shares if s >= 0.0005)


def ensure_metadata(client, raw_dir, plan, refresh=False):
    for repo, tier, group, rationale, hait in plan:
        d = repo_dir(raw_dir, repo)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "repo_meta.json")
        if os.path.exists(path) and not refresh:
            continue
        owner, name = repo.split("/")
        meta = fp.fetch_repo_meta(client, owner, name)
        meta.update({"tier": tier, "group": group, "group_rationale": rationale,
                     "hait_language": hait, "mixed_threshold": fp.MIXED_THRESHOLD,
                     "window": {"start": fp.WINDOW_START, "end": fp.WINDOW_END}})
        fp.atomic_write_json(path, meta)
        log("meta %s: %s %.0f%% mixed=%s closed-in-window=%s archived=%s mirror=%s"
            % (repo, meta["languages"]["primary"], 100 * (meta["languages"]["primary_share"] or 0),
               meta["languages"]["mixed"], meta["search_closed_in_window"],
               meta["is_archived"], meta["is_mirror"]))
    write_repos_csv(raw_dir)


def write_repos_csv(raw_dir, plan=REPO_PLAN):
    rows = []
    for i, (repo, tier, group, rationale, hait) in enumerate(plan, 1):
        path = os.path.join(repo_dir(raw_dir, repo), "repo_meta.json")
        if not os.path.exists(path):
            continue
        meta = json.load(open(path))
        langs = meta["languages"]
        rows.append({
            "order": i, "repo": repo, "tier": tier, "group": group,
            "group_rationale": rationale, "language": langs["primary"],
            "primary_share": langs["primary_share"], "mixed": langs["mixed"],
            "language_shares": format_shares(langs["shares"]), "hait_language": hait,
            "github_primary_language": meta.get("github_primary_language"),
            "search_closed_in_window": meta.get("search_closed_in_window"),
            "search_closed_ci_part": meta.get("search_closed_ci_part"),
            "is_archived": meta.get("is_archived"), "is_mirror": meta.get("is_mirror"),
            "owner_type": meta.get("owner_type"), "meta_fetched_at": meta.get("fetched_at"),
        })
    tmp = os.path.join(raw_dir, "repos.csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REPOS_CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, os.path.join(raw_dir, "repos.csv"))
    return rows


# ── Collection loop ─────────────────────────────────────────────────────────

def is_done(raw_dir, repo):
    cp = os.path.join(repo_dir(raw_dir, repo), "checkpoint.json")
    if not os.path.exists(cp):
        return False
    return json.load(open(cp)).get("phase") == "done"


def collect(client, raw_dir, plan, max_passes=3, retry_pause_s=900, sleep=time.sleep,
            fetcher_cls=fp.RepoFetcher):
    for attempt in range(1, max_passes + 1):
        pending = [p for p in plan if not is_done(raw_dir, p[0])]
        if not pending:
            break
        log("pass %d: %d repositories pending" % (attempt, len(pending)))
        for repo, *_ in pending:
            if stop_requested(raw_dir):
                log("stop requested; exiting before %s" % repo)
                return False
            owner, name = repo.split("/")
            try:
                cp = fetcher_cls(client, owner, name, raw_dir=raw_dir,
                                 stop_requested=lambda: stop_requested(raw_dir)).run()
            except (fp.GraphQLError, fp.TransientError, OSError, KeyError, TypeError) as e:
                log("%s failed: %s: %s (will retry in a later pass)" % (repo, type(e).__name__, e))
                continue
            if cp.get("status") == "stopped":
                return False
        if all(is_done(raw_dir, p[0]) for p in plan):
            break
        if attempt < max_passes:
            log("some repositories failed; pausing %ds before the next pass" % retry_pause_s)
            sleep(retry_pause_s)
    done = [p[0] for p in plan if is_done(raw_dir, p[0])]
    log("finished: %d/%d repositories done" % (len(done), len(plan)))
    return len(done) == len(plan)


def select_plan(only=None, tiers=None):
    plan = REPO_PLAN
    if only:
        wanted = [s.strip() for s in only.split(",") if s.strip()]
        by_repo = {p[0]: p for p in REPO_PLAN}
        missing = [w for w in wanted if w not in by_repo]
        if missing:
            raise SystemExit("not in REPO_PLAN: %s" % ", ".join(missing))
        plan = [by_repo[w] for w in wanted]
    if tiers:
        ts = set(tiers.split(","))
        plan = [p for p in plan if any(t in ts for t in p[1].split("+"))]
    return plan


def main():
    ap = argparse.ArgumentParser(description="Corpus v2 orchestrator")
    ap.add_argument("--only", help="comma-separated owner/name list (subset of the plan)")
    ap.add_argument("--tiers", help="comma-separated tiers: K0,P1,R12,P2")
    ap.add_argument("--raw-dir", default=fp.RAW_V2_DIR)
    ap.add_argument("--meta-only", action="store_true")
    ap.add_argument("--refresh-meta", action="store_true")
    ap.add_argument("--list", action="store_true", help="print the queue and exit")
    ap.add_argument("--max-passes", type=int, default=3)
    args = ap.parse_args()

    plan = select_plan(args.only, args.tiers)
    if args.list:
        for i, p in enumerate(plan, 1):
            print("%2d %-35s %-7s %s" % (i, p[0], p[1], p[2]))
        return

    os.makedirs(args.raw_dir, exist_ok=True)
    pid_path = acquire_pid(args.raw_dir)
    install_signal_handlers()
    stop_file = os.path.join(args.raw_dir, "STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)
        log("removed a STOP file left from the previous run")
    try:
        log("start PID %d, %d repositories, window %s..%s, CI since %s"
            % (os.getpid(), len(plan), fp.WINDOW_START, fp.WINDOW_END, fp.CI_SINCE))
        client = fp.GitHubClient(fp.load_token(log=log), log=log)
        # metadata for the whole plan first (repos.csv covers all 40)
        ensure_metadata(client, args.raw_dir, REPO_PLAN, refresh=args.refresh_meta)
        if args.meta_only:
            return
        ok = collect(client, args.raw_dir, plan, max_passes=args.max_passes)
        log("exit: %s; points used this run %d, waited %.0fs for limits"
            % ("all done" if ok else "incomplete", client.points_used, client.waited_s))
    finally:
        release_pid(pid_path)


if __name__ == "__main__":
    main()
