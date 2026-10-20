#!/usr/bin/env python3
"""
Corpus v2 collector: closed and merged pull requests in a fixed creation window.

Differences from fetch_prs.py (the frozen HAIT collector, left untouched):
  - fixed window on createdAt (UTC), exact stop on the window start date, no count limit;
  - states MERGED + CLOSED (abandoned PRs), no OPEN snapshot;
  - extended fields: timeline events, per-commit committedDate/additions/deletions,
    mergedBy, authorAssociation, review commit oid, CI check suites (recent PRs only);
  - nested connections (commits, reviews, threads, thread comments, timeline, labels,
    review requests) are paginated to completion with follow-up queries;
  - checkpoint after every page, append-only JSONL output, rate-limit aware waiting,
    retries with backoff, page size 25 -> 10 -> 5 -> 2 -> 1 on repeated 502/504;
    a single PR that still times out is fetched without per-commit stats, which
    are then requested commit by commit (missing ones stay null); if it still fails,
    thread comments are fetched without their review link (`_meta.thread_review_link_missing`).

Output (per repository, under data/raw_v2/<owner>_<name>/):
  prs.jsonl         one PR node per line (GraphQL shape + "_meta")
  checkpoint.json   cursor, counters, status
  fetch.log         timestamped progress log
and the shared data/raw_v2/progress.csv.

Usage (single repository; the orchestrator is run_corpus_v2.py):
    .venv/bin/python fetch/fetch_prs_v2.py --repo eslint/eslint

The token is read from $GITHUB_TOKEN or the project .env and is never printed.
"""

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime, timezone

import requests

# ── Constants ───────────────────────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
PROJECT_ROOT = SCRIPTS_DIR
RAW_V2_DIR = os.path.join(SCRIPTS_DIR, "data", "raw_v2")
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

WINDOW_START = "2024-10-01T00:00:00Z"   # inclusive
WINDOW_END = "2026-08-31T23:59:59Z"     # inclusive
# CI check suites are retained by GitHub for ~13 months. CI is requested only for
# PRs created on/after this date (the last ~12 months of the window, rounded to a
# month boundary). Fixed, so that every repo uses the same cut.
CI_SINCE = "2025-10-01T00:00:00Z"

DEFAULT_PAGE_SIZE = 25      # 50 times out (HTTP 502) with the extended fields
# Steps after repeated 502/504 on the same page. Some repositories (openproject,
# transformers) time out at 10 PRs even without CI fields, while one PR takes 1-7 s.
FALLBACK_PAGE_SIZES = (10, 5, 2, 1)
RESTORE_AFTER_PAGES = 8     # successful reduced pages before trying one step larger
MIN_REMAINING = 150         # wait for reset when fewer points remain
RESET_MARGIN_S = 30         # extra seconds after resetAt
HTTP_TIMEOUT_S = 90
COLLECTOR_VERSION = "v2.2"

PR_STATES = ["MERGED", "CLOSED"]

PROGRESS_FIELDS = ["repo", "status", "collected", "last_created", "pages", "points",
                   "skipped_after_window", "page_size", "started_at", "updated_at",
                   "elapsed_s", "message"]

# ── GraphQL fragments ───────────────────────────────────────────────────────

ACTOR = "login __typename"

# Per-commit diff stats are computed by GitHub on request; one huge commit (measured
# 16.10.2026 on chatwoot #13672) makes even a one-PR page time out, while the same
# page without these fields takes ~1 s.
COMMIT_STATS = "additions deletions changedFilesIfAvailable"
COMMIT_NODE = """
  commit {
    oid message authoredDate committedDate
    %s
    parents { totalCount }
    author { user { login } }
  }
""" % COMMIT_STATS
COMMIT_NODE_NO_STATS = COMMIT_NODE.replace(COMMIT_STATS, "")

REVIEW_NODE = """
  id author { %s } authorAssociation authorCanPushToRepository
  state submittedAt createdAt body commit { oid }
""" % ACTOR

# The review link of a thread comment can fail on GitHub's side: measured 19.10.2026
# on vscode #322121, a Copilot comment whose `pullRequestReview` makes every query
# with this field fail at once ("Something went wrong"), at any page size.
REVIEW_LINK = "pullRequestReview { id }"
COMMENT_NODE = """
  author { %s } authorAssociation body createdAt
  originalCommit { oid } %s
""" % (ACTOR, REVIEW_LINK)
COMMENT_NODE_NO_REVIEW_LINK = COMMENT_NODE.replace(REVIEW_LINK, "")

THREAD_NODE_TEMPLATE = """
  id isResolved isOutdated path line originalLine resolvedBy { login }
  comments(first: 20) {
    totalCount pageInfo { hasNextPage endCursor }
    nodes { %s }
  }
"""
THREAD_NODE = THREAD_NODE_TEMPLATE % COMMENT_NODE
THREAD_NODE_NO_REVIEW_LINK = THREAD_NODE_TEMPLATE % COMMENT_NODE_NO_REVIEW_LINK

REQUESTED_REVIEWER = """
  requestedReviewer {
    __typename
    ... on User { login }
    ... on Bot { login }
    ... on Mannequin { login }
    ... on Team { slug }
  }
"""

TIMELINE_TYPES = [
    "READY_FOR_REVIEW_EVENT", "CONVERT_TO_DRAFT_EVENT",
    "REVIEW_REQUESTED_EVENT", "REVIEW_REQUEST_REMOVED_EVENT",
    "HEAD_REF_FORCE_PUSHED_EVENT", "ISSUE_COMMENT", "PULL_REQUEST_COMMIT",
    "MERGED_EVENT", "CLOSED_EVENT", "REOPENED_EVENT",
    "AUTO_MERGE_ENABLED_EVENT", "AUTO_MERGE_DISABLED_EVENT",
    "ADDED_TO_MERGE_QUEUE_EVENT", "REMOVED_FROM_MERGE_QUEUE_EVENT",
    "REVIEW_DISMISSED_EVENT",
]

TIMELINE_NODE = """
  __typename
  ... on ReadyForReviewEvent { createdAt actor { %(a)s } }
  ... on ConvertToDraftEvent { createdAt actor { %(a)s } }
  ... on ReviewRequestedEvent { createdAt actor { %(a)s } %(rr)s }
  ... on ReviewRequestRemovedEvent { createdAt actor { %(a)s } %(rr)s }
  ... on HeadRefForcePushedEvent { createdAt actor { %(a)s } beforeCommit { oid } afterCommit { oid } }
  ... on IssueComment { createdAt author { %(a)s } authorAssociation }
  ... on PullRequestCommit { commit { oid authoredDate committedDate } }
  ... on MergedEvent { createdAt actor { %(a)s } commit { oid } }
  ... on ClosedEvent { createdAt actor { %(a)s } }
  ... on ReopenedEvent { createdAt actor { %(a)s } }
  ... on AutoMergeEnabledEvent { createdAt actor { %(a)s } }
  ... on AutoMergeDisabledEvent { createdAt actor { %(a)s } }
  ... on AddedToMergeQueueEvent { createdAt actor { %(a)s } }
  ... on RemovedFromMergeQueueEvent { createdAt actor { %(a)s } }
  ... on ReviewDismissedEvent { createdAt actor { %(a)s } review { id } }
""" % {"a": ACTOR, "rr": REQUESTED_REVIEWER}

TIMELINE_TYPES_GQL = "[" + ", ".join(TIMELINE_TYPES) + "]"

# CI, part 1 (inside the page query): check suites of the last 30 commits (push time ~
# suite createdAt, status, conclusion, app) and legacy commit statuses (a plain list,
# no extra connection cost).
# CI, part 2 (separate CI_DETAIL_QUERY over nodes(ids), chunks of 10 PRs): check runs
# and workflow names for the head commit's suites.
# Why split (measured 14.10.2026 on eslint, page of 25): no CI 14 points / ~5 s;
# suites 22 points / ~5.6 s (first: 5, 10 or 20 cost the same); + workflowRun ~7-9 s;
# + head check runs inside the page -> HTTP 502 or RESOURCE_LIMITS_EXCEEDED with
# partial data. Check runs under every commit would be ~+150 points per page.
CI_FIELDS = """
  ciCommits: commits(last: 30) {
    totalCount
    nodes {
      commit {
        oid committedDate
        checkSuites(first: 20) {
          totalCount
          nodes { createdAt updatedAt status conclusion app { slug } }
        }
        status { state contexts { context state createdAt } }
      }
    }
  }
"""

CI_HEAD_FIELDS = """
  ciHead: commits(last: 1) {
    nodes {
      commit {
        oid
        checkSuites(first: 20) {
          totalCount
          nodes {
            createdAt status conclusion
            app { slug }
            workflowRun { workflow { name } }
            checkRuns(first: 50) {
              totalCount
              nodes { name status conclusion startedAt completedAt }
            }
          }
        }
      }
    }
  }
"""
CI_DETAIL_CHUNK = 10

# Nested connections that are paginated to completion: key -> (args, node fields)
NESTED = {
    "labels": ("first: 20", "name"),
    "commits": ("first: 100", COMMIT_NODE),
    "reviews": ("first: 50", REVIEW_NODE),
    "reviewThreads": ("first: 50", THREAD_NODE),
    "reviewRequests": ("first: 10", REQUESTED_REVIEWER),
    "timelineItems": ("first: 100, itemTypes: " + TIMELINE_TYPES_GQL, TIMELINE_NODE),
}


def _connection(key, args, node_fields):
    return """
    %s(%s) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes { %s }
    }""" % (key, args, node_fields)


PR_SCALARS = """
  id number title body url
  createdAt updatedAt mergedAt closedAt state isDraft
  additions deletions changedFiles
  baseRefName headRefName baseRefOid headRefOid isCrossRepository
  authorAssociation totalCommentsCount
  author { %(a)s }
  mergedBy { %(a)s }
  milestone { title }
""" % {"a": ACTOR}

RATE_LIMIT = "rateLimit { cost remaining resetAt limit }"


def _nested_node(key, commit_stats=True, review_link=True):
    args, node = NESTED[key]
    if key == "commits" and not commit_stats:
        node = COMMIT_NODE_NO_STATS
    if key == "reviewThreads" and not review_link:
        node = THREAD_NODE_NO_REVIEW_LINK
    return args, node


def build_pr_fields(with_ci, commit_stats=True, review_link=True):
    parts = [PR_SCALARS]
    for key in NESTED:
        parts.append(_connection(key, *_nested_node(key, commit_stats, review_link)))
    if with_ci:
        parts.append(CI_FIELDS)
    return "\n".join(parts)


def build_page_query(with_ci, commit_stats=True, review_link=True):
    """Main page query. Page size and cursor are variables."""
    return """
query($owner: String!, $name: String!, $cursor: String, $pageSize: Int!) {
  %s
  repository(owner: $owner, name: $name) {
    pullRequests(first: $pageSize, after: $cursor, states: [%s],
                 orderBy: {field: CREATED_AT, direction: DESC}) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes { %s }
    }
  }
}""" % (RATE_LIMIT, ", ".join(PR_STATES), build_pr_fields(with_ci, commit_stats, review_link))


LOCATE_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  %s
  repository(owner: $owner, name: $name) {
    pullRequests(first: 100, after: $cursor, states: [%s],
                 orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      edges { cursor node { number createdAt } }
    }
  }
}""" % (RATE_LIMIT, ", ".join(PR_STATES))


def build_more_query(key, commit_stats=True, review_link=True):
    """Follow-up query for one nested connection of one pull request."""
    args, node = _nested_node(key, commit_stats, review_link)
    # replace "first: N" by first: 100 and add the cursor
    rest = args.split(",", 1)[1] if "," in args else ""
    new_args = "first: 100, after: $cursor" + ("," + rest if rest else "")
    return """
query($id: ID!, $cursor: String) {
  %s
  node(id: $id) { ... on PullRequest { %s } }
}""" % (RATE_LIMIT, _connection(key, new_args, node))


THREAD_COMMENTS_QUERY_TEMPLATE = """
query($id: ID!, $cursor: String) {
  %s
  node(id: $id) {
    ... on PullRequestReviewThread {
      comments(first: 100, after: $cursor) {
        totalCount pageInfo { hasNextPage endCursor }
        nodes { %s }
      }
    }
  }
}"""
THREAD_COMMENTS_QUERY = THREAD_COMMENTS_QUERY_TEMPLATE % (RATE_LIMIT, COMMENT_NODE)
THREAD_COMMENTS_QUERY_NO_REVIEW_LINK = THREAD_COMMENTS_QUERY_TEMPLATE % (
    RATE_LIMIT, COMMENT_NODE_NO_REVIEW_LINK)

COMMIT_STATS_QUERY = """
query($owner: String!, $name: String!, $oid: GitObjectID!) {
  %s
  repository(owner: $owner, name: $name) { object(oid: $oid) { ... on Commit { %s } } }
}""" % (RATE_LIMIT, COMMIT_STATS)

CI_DETAIL_QUERY = """
query($ids: [ID!]!) {
  %s
  nodes(ids: $ids) { ... on PullRequest { id %s } }
}""" % (RATE_LIMIT, CI_HEAD_FIELDS)

REPO_META_QUERY = """
query($owner: String!, $name: String!, $search: String!, $searchCi: String!) {
  %s
  repository(owner: $owner, name: $name) {
    nameWithOwner isArchived isMirror isFork createdAt stargazerCount
    owner { __typename login }
    primaryLanguage { name }
    languages(first: 100, orderBy: {field: SIZE, direction: DESC}) {
      totalSize totalCount
      edges { size node { name } }
    }
  }
  search(query: $search, type: ISSUE, first: 0) { issueCount }
  ciPart: search(query: $searchCi, type: ISSUE, first: 0) { issueCount }
}""" % RATE_LIMIT


def search_query_closed(owner, name, start=WINDOW_START, end=WINDOW_END):
    return "repo:%s/%s is:pr is:closed created:%s..%s" % (owner, name, start, end)


# ── Time helpers ────────────────────────────────────────────────────────────

def parse_dt(s):
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def utcnow():
    return datetime.now(timezone.utc)


def iso(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def window_position(created_at, start=WINDOW_START, end=WINDOW_END):
    """'after' (newer than window end), 'in', or 'before' (older than window start)."""
    c = parse_dt(created_at)
    if c > parse_dt(end):
        return "after"
    if c < parse_dt(start):
        return "before"
    return "in"


# ── Languages ───────────────────────────────────────────────────────────────

MIXED_THRESHOLD = 0.60   # primary language share below this -> "mixed"


def summarize_languages(languages, threshold=MIXED_THRESHOLD):
    """Byte distribution from a GraphQL `languages` connection.

    Returns dict(primary, primary_share, mixed, total_bytes, shares=[(name, bytes, share)]).
    """
    total = (languages or {}).get("totalSize") or 0
    edges = (languages or {}).get("edges") or []
    shares = []
    for e in edges:
        size = e.get("size") or 0
        shares.append((e["node"]["name"], size, (size / total) if total else 0.0))
    shares.sort(key=lambda t: -t[1])
    if not shares or total == 0:
        return {"primary": None, "primary_share": None, "mixed": None,
                "total_bytes": total, "shares": shares}
    top = shares[0]
    return {"primary": top[0], "primary_share": round(top[2], 4),
            "mixed": top[2] < threshold, "total_bytes": total, "shares": shares}


# ── Token ───────────────────────────────────────────────────────────────────

def _token_from_env_file(env_path):
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    line = line[len("export "):]
                if line.startswith("GITHUB_TOKEN="):
                    return line.split("=", 1)[1].strip().strip("'\"") or None
    return None


def _token_from_gh_cli():
    import subprocess
    try:
        out = subprocess.run(["gh", "auth", "token", "-h", "github.com"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def token_candidates(env_path=ENV_PATH):
    """(source_name, token) in priority order; values are never logged."""
    yield "env:GITHUB_TOKEN", (os.environ.get("GITHUB_TOKEN") or "").strip() or None
    yield ".env:GITHUB_TOKEN", _token_from_env_file(env_path)
    yield "gh auth token (github.com)", _token_from_gh_cli()


def token_is_valid(token, session=None):
    s = session or requests
    try:
        r = s.post(GitHubClient.URL, json={"query": "{ rateLimit { remaining } }"},
                   headers={"Authorization": "bearer " + token}, timeout=30)
    except requests.RequestException:
        return False
    return r.status_code == 200


def load_token(env_path=ENV_PATH, validate=True, log=print):
    """First valid token from $GITHUB_TOKEN, .env, then the gh CLI keyring."""
    for source, tok in token_candidates(env_path):
        if not tok:
            continue
        if not validate or token_is_valid(tok):
            log("token source: %s" % source)
            return tok
        log("token source %s rejected (HTTP 401 or unreachable)" % source)
    raise RuntimeError("no valid GitHub token (checked $GITHUB_TOKEN, .env, gh auth token)")


# ── GraphQL client ──────────────────────────────────────────────────────────

class GraphQLError(Exception):
    """Non-retryable error (bad query, not found, auth)."""


class TransientError(Exception):
    """Retryable error that persisted after all attempts."""


class ServerTimeoutError(TransientError):
    """502/504 or a GraphQL 'timeout' / 'something went wrong' error."""


def _is_timeout_message(msg):
    m = (msg or "").lower()
    return any(s in m for s in ("timeout", "timed out", "something went wrong",
                                "couldn't respond", "loading"))


class GitHubClient:
    URL = "https://api.github.com/graphql"

    def __init__(self, token, session=None, sleep=time.sleep, clock=utcnow,
                 log=None, min_remaining=MIN_REMAINING, timeout=HTTP_TIMEOUT_S):
        self._token = token
        self.session = session or requests.Session()
        self.sleep = sleep
        self.clock = clock
        self.log = log or (lambda msg: None)
        self.min_remaining = min_remaining
        self.timeout = timeout
        self.last_rate = None     # {"cost", "remaining", "resetAt", "limit"}
        self.points_used = 0
        self.calls = 0
        self.waited_s = 0.0
        self.partial_errors = 0

    def __repr__(self):  # never expose the token
        return "GitHubClient(calls=%d, points=%d)" % (self.calls, self.points_used)

    def _headers(self):
        return {"Authorization": "bearer " + self._token,
                "Accept": "application/vnd.github+json",
                "User-Agent": "phd-corpus-v2-collector"}

    def _sleep(self, seconds, why):
        seconds = max(0.0, float(seconds))
        if seconds > 0:
            self.log("wait %.0fs: %s" % (seconds, why))
            self.waited_s += seconds
            self.sleep(seconds)

    def _wait_reset(self, reset_at, why):
        if reset_at is None:
            self._sleep(60, why + " (reset unknown)")
            return
        wait = (reset_at - self.clock()).total_seconds() + RESET_MARGIN_S
        self._sleep(max(wait, 5), why)

    @staticmethod
    def _backoff(attempt):
        return min(300.0, 5.0 * (3 ** (attempt - 1))) + random.uniform(0, 2)

    def execute(self, query, variables, max_attempts=6):
        """Run a query; returns `data`. Rate-limit waits do not consume attempts."""
        attempt = 0
        last_exc = None
        rate_waits = 0
        while attempt < max_attempts:
            try:
                resp = self.session.post(self.URL, json={"query": query, "variables": variables},
                                         headers=self._headers(), timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as e:
                attempt += 1
                last_exc = ServerTimeoutError("network: %s" % type(e).__name__)
                self.log("attempt %d/%d network error %s" % (attempt, max_attempts, type(e).__name__))
                if attempt < max_attempts:
                    self._sleep(self._backoff(attempt), "retry after network error")
                continue

            self.calls += 1
            status = resp.status_code
            if status in (403, 429):
                text = (resp.text or "")[:300].lower()
                if status == 429 or "rate limit" in text or "abuse" in text:
                    rate_waits += 1
                    if rate_waits > 20:
                        raise TransientError("rate limited too many times")
                    retry_after = resp.headers.get("Retry-After")
                    reset_hdr = resp.headers.get("X-RateLimit-Reset")
                    if retry_after:
                        self._sleep(int(retry_after) + 5, "HTTP %d Retry-After" % status)
                    elif reset_hdr and resp.headers.get("X-RateLimit-Remaining") == "0":
                        self._wait_reset(datetime.fromtimestamp(int(reset_hdr), timezone.utc),
                                         "HTTP %d primary rate limit" % status)
                    else:
                        self._sleep(60 * min(rate_waits, 5), "HTTP %d secondary rate limit" % status)
                    continue
                raise GraphQLError("HTTP %d: %s" % (status, text[:200]))
            if status == 401:
                raise GraphQLError("HTTP 401: bad credentials")
            if status >= 500:
                attempt += 1
                cls = ServerTimeoutError if status in (502, 504) else TransientError
                last_exc = cls("HTTP %d" % status)
                self.log("attempt %d/%d HTTP %d" % (attempt, max_attempts, status))
                if attempt < max_attempts:
                    self._sleep(self._backoff(attempt), "retry after HTTP %d" % status)
                continue
            if status != 200:
                raise GraphQLError("HTTP %d: %s" % (status, (resp.text or "")[:200]))

            try:
                payload = resp.json()
            except ValueError:
                # seen in practice: HTTP 200 with a truncated body when the server times out
                attempt += 1
                last_exc = ServerTimeoutError("invalid JSON (HTTP 200, %d bytes)" % len(resp.text or ""))
                self.log("attempt %d/%d invalid JSON body (%d bytes)" % (attempt, max_attempts, len(resp.text or "")))
                if attempt < max_attempts:
                    self._sleep(self._backoff(attempt), "retry after invalid JSON")
                continue

            data = payload.get("data") or {}
            rate = data.get("rateLimit") if isinstance(data, dict) else None
            if rate:
                self.last_rate = rate
                self.points_used += rate.get("cost") or 0

            errors = payload.get("errors")
            if errors:
                types = {e.get("type") for e in errors}
                msgs = "; ".join(str(e.get("message"))[:200] for e in errors[:3])
                if "RATE_LIMITED" in types or "rate limit" in msgs.lower():
                    rate_waits += 1
                    if rate_waits > 20:
                        raise TransientError("rate limited too many times")
                    reset = parse_dt(rate["resetAt"]) if rate else None
                    self._wait_reset(reset, "GraphQL RATE_LIMITED")
                    continue
                # Partial data with nested FORBIDDEN/NOT_FOUND (e.g. a secret team as a
                # requested reviewer): keep the data, the field is simply null.
                nested_only = all(len(e.get("path") or []) > 2 for e in errors)
                if data and types <= {"FORBIDDEN", "NOT_FOUND"} and nested_only:
                    self.partial_errors += 1
                    self.log("partial data accepted (%s): %s" % (",".join(sorted(types)), msgs[:150]))
                    errors = None
                elif _is_timeout_message(msgs) or types & {"INTERNAL", "TIMEOUT", "SERVICE_UNAVAILABLE",
                                                            "RESOURCE_LIMITS_EXCEEDED"}:
                    attempt += 1
                    last_exc = ServerTimeoutError("GraphQL: " + msgs)
                    self.log("attempt %d/%d GraphQL error: %s" % (attempt, max_attempts, msgs))
                    if attempt < max_attempts:
                        self._sleep(self._backoff(attempt), "retry after GraphQL error")
                    continue
                else:
                    raise GraphQLError("GraphQL: " + msgs)

            if rate and rate.get("remaining") is not None and rate["remaining"] < self.min_remaining:
                self._wait_reset(parse_dt(rate["resetAt"]),
                                 "rate limit low (remaining %s)" % rate["remaining"])
            return data
        raise last_exc or TransientError("failed")


# ── File helpers ────────────────────────────────────────────────────────────

def atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def repair_and_index_jsonl(path):
    """Drop a trailing partial line (crash mid-write); return {pr_number: createdAt}."""
    seen = {}
    if not os.path.exists(path):
        return seen
    with open(path, "rb") as f:
        raw = f.read()
    if raw and not raw.endswith(b"\n"):
        cut = raw.rfind(b"\n") + 1
        with open(path, "r+b") as f:
            f.truncate(cut)
        raw = raw[:cut]
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        seen[rec["number"]] = rec.get("createdAt")
    return seen


def append_jsonl(path, records):
    if not records:
        return
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def update_progress(raw_dir, row):
    path = os.path.join(raw_dir, "progress.csv")
    rows = {}
    order = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows[r["repo"]] = r
                order.append(r["repo"])
    if row["repo"] not in rows:
        order.append(row["repo"])
    merged = dict(rows.get(row["repo"], {}))
    merged.update({k: ("" if v is None else v) for k, v in row.items()})
    rows[row["repo"]] = merged
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PROGRESS_FIELDS, extrasaction="ignore")
        w.writeheader()
        for repo in order:
            w.writerow(rows[repo])
    os.replace(tmp, path)


class RepoLogger:
    def __init__(self, path, echo=True):
        self.path = path
        self.echo = echo
        self.prefix = ""

    def __call__(self, msg):
        line = "%s %s%s" % (utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), self.prefix, msg)
        with open(self.path, "a") as f:
            f.write(line + "\n")
        if self.echo:
            print(line, flush=True)


# ── Repository fetcher ──────────────────────────────────────────────────────

class RepoFetcher:
    def __init__(self, client, owner, name, raw_dir=RAW_V2_DIR,
                 window_start=WINDOW_START, window_end=WINDOW_END, ci_since=CI_SINCE,
                 page_size=DEFAULT_PAGE_SIZE, fallback_page_sizes=FALLBACK_PAGE_SIZES,
                 stop_requested=None, echo=True, clock=utcnow):
        self.client = client
        self.owner, self.name = owner, name
        self.repo = "%s/%s" % (owner, name)
        self.raw_dir = raw_dir
        self.dir = os.path.join(raw_dir, "%s_%s" % (owner, name))
        os.makedirs(self.dir, exist_ok=True)
        self.jsonl = os.path.join(self.dir, "prs.jsonl")
        self.cp_path = os.path.join(self.dir, "checkpoint.json")
        self.log = RepoLogger(os.path.join(self.dir, "fetch.log"), echo=echo)
        self.log.prefix = "[%s] " % self.repo
        self.window_start, self.window_end, self.ci_since = window_start, window_end, ci_since
        self.default_page_size = page_size
        self.page_sizes = [page_size] + sorted(
            (s for s in fallback_page_sizes if s < page_size), reverse=True)
        self.stop_requested = stop_requested or (lambda: False)
        self.clock = clock
        if client is not None:
            client.log = self.log

    def smaller_page_size(self, size):
        """Next step below `size`, or None at the smallest step."""
        return next((s for s in self.page_sizes if s < size), None)

    def larger_page_size(self, size):
        """Next step above `size`, capped at the default."""
        return next((s for s in reversed(self.page_sizes) if s > size), self.default_page_size)

    # checkpoint ------------------------------------------------------------
    def _new_checkpoint(self):
        now = iso(self.clock())
        return {"repo": self.repo, "phase": "locate", "cursor": None, "locate_cursor": None,
                "page_size": self.default_page_size, "small_pages_ok": 0, "pages": 0,
                "n_written": 0, "last_created": None, "first_created": None,
                "skipped_after_window": 0, "points": 0, "followup_queries": 0,
                "window": {"start": self.window_start, "end": self.window_end},
                "ci_since": self.ci_since, "states": PR_STATES,
                "collector": COLLECTOR_VERSION, "status": "running",
                "started_at": now, "updated_at": now, "elapsed_s": 0.0}

    def corpus_params(self):
        """Parameters that define what a record means; a resume must not change them."""
        return {"window": {"start": self.window_start, "end": self.window_end},
                "ci_since": self.ci_since, "states": PR_STATES, "collector": COLLECTOR_VERSION}

    def load_checkpoint(self):
        if os.path.exists(self.cp_path):
            with open(self.cp_path) as f:
                cp = json.load(f)
            if cp.get("window") != {"start": self.window_start, "end": self.window_end}:
                raise GraphQLError("checkpoint window %s differs from %s..%s; refusing to mix"
                                   % (cp.get("window"), self.window_start, self.window_end))
            # Older checkpoints may lack a field; only stored fields are compared.
            diffs = ["%s: checkpoint %s, now %s" % (key, json.dumps(cp[key]), json.dumps(value))
                     for key, value in self.corpus_params().items()
                     if key in cp and cp[key] != value]
            if diffs:
                raise GraphQLError("checkpoint %s was written with other corpus parameters (%s); "
                                   "refusing to mix. Collect into a new --raw-dir instead."
                                   % (self.cp_path, "; ".join(diffs)))
            return cp
        return self._new_checkpoint()

    def save_checkpoint(self, cp):
        cp["updated_at"] = iso(self.clock())
        atomic_write_json(self.cp_path, cp)
        update_progress(self.raw_dir, {
            "repo": self.repo, "status": cp["status"], "collected": cp["n_written"],
            "last_created": cp["last_created"], "pages": cp["pages"], "points": cp["points"],
            "skipped_after_window": cp["skipped_after_window"], "page_size": cp["page_size"],
            "started_at": cp["started_at"], "updated_at": cp["updated_at"],
            "elapsed_s": round(cp["elapsed_s"], 1), "message": cp.get("message", "")})

    # locate ----------------------------------------------------------------
    def locate_start(self, cp):
        """Cheap scan (100 per page, ~1 point) to the first PR created <= window end.

        Sets cp['cursor'] to the cursor *just before* that PR (None = from the top).
        """
        cursor = cp.get("locate_cursor")
        prev_cursor = cursor
        while True:
            data = self.client.execute(LOCATE_QUERY, {"owner": self.owner, "name": self.name,
                                                      "cursor": cursor})
            self._check_repo(data)
            conn = data["repository"]["pullRequests"]
            cp["points"] += (data.get("rateLimit") or {}).get("cost") or 0
            for edge in conn["edges"]:
                if window_position(edge["node"]["createdAt"], self.window_start,
                                   self.window_end) != "after":
                    cp["cursor"] = prev_cursor
                    cp["phase"] = "fetch"
                    self.log("located window end: first in-window PR #%s created %s"
                             % (edge["node"]["number"], edge["node"]["createdAt"]))
                    return
                cp["skipped_after_window"] += 1
                prev_cursor = edge["cursor"]
            if not conn["pageInfo"]["hasNextPage"]:
                cp["phase"] = "done"
                cp["status"] = "done"
                self.log("no PR at or before window end")
                return
            cursor = conn["pageInfo"]["endCursor"]
            cp["locate_cursor"] = cursor
            self.save_checkpoint(cp)

    def _check_repo(self, data):
        if not data or data.get("repository") is None:
            raise GraphQLError("repository %s not found" % self.repo)

    # nested pagination ---------------------------------------------------------
    def complete_nested(self, pr, cp, commit_stats=True, review_link=True):
        for key in NESTED:
            conn = pr.get(key)
            if not conn:
                continue
            while conn["pageInfo"]["hasNextPage"]:
                data = self.client.execute(build_more_query(key, commit_stats, review_link),
                                           {"id": pr["id"], "cursor": conn["pageInfo"]["endCursor"]})
                cp["followup_queries"] += 1
                cp["points"] += (data.get("rateLimit") or {}).get("cost") or 0
                more = data["node"][key]
                conn["nodes"].extend(more["nodes"])
                conn["pageInfo"] = more["pageInfo"]
        for thread in (pr.get("reviewThreads") or {}).get("nodes", []):
            comments = thread.get("comments")
            if not comments:
                continue
            while comments["pageInfo"]["hasNextPage"]:
                data = self.client.execute(THREAD_COMMENTS_QUERY if review_link
                                           else THREAD_COMMENTS_QUERY_NO_REVIEW_LINK,
                                           {"id": thread["id"],
                                            "cursor": comments["pageInfo"]["endCursor"]})
                cp["followup_queries"] += 1
                cp["points"] += (data.get("rateLimit") or {}).get("cost") or 0
                more = data["node"]["comments"]
                comments["nodes"].extend(more["nodes"])
                comments["pageInfo"] = more["pageInfo"]
        return pr

    def fill_commit_stats(self, pr, cp):
        """Request diff stats commit by commit; return how many stayed null."""
        missing = 0
        for node in pr["commits"]["nodes"]:
            commit = node["commit"]
            try:
                data = self.client.execute(COMMIT_STATS_QUERY,
                                           {"owner": self.owner, "name": self.name,
                                            "oid": commit["oid"]}, max_attempts=2)
                stats = (data["repository"] or {}).get("object") or {}
                cp["points"] += (data.get("rateLimit") or {}).get("cost") or 0
            except ServerTimeoutError:
                stats = {}
            cp["followup_queries"] += 1
            for field in COMMIT_STATS.split():
                commit[field] = stats.get(field)
            if not stats:
                missing += 1
                self.log("PR #%s commit %s: diff stats unavailable" % (pr["number"], commit["oid"][:10]))
        return missing

    # CI details (head commit check runs) ---------------------------------------
    def fetch_ci_details(self, records, cp, chunk=CI_DETAIL_CHUNK):
        """Attach `ciHead` to records via nodes(ids); split chunks on timeouts."""
        todo = [r for r in records if r["_meta"]["ci_fetched"]]
        by_id = {r["id"]: r for r in todo}
        queue = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
        while queue:
            part = queue.pop(0)
            try:
                data = self.client.execute(CI_DETAIL_QUERY, {"ids": [r["id"] for r in part]},
                                           max_attempts=2 if len(part) > 1 else 6)
            except ServerTimeoutError:
                if len(part) == 1:
                    raise
                half = len(part) // 2
                self.log("CI details timeout for %d PRs; splitting" % len(part))
                queue[:0] = [part[:half], part[half:]]
                continue
            cp["followup_queries"] += 1
            cp["points"] += (data.get("rateLimit") or {}).get("cost") or 0
            for node in data["nodes"]:
                if node and node.get("id") in by_id:
                    by_id[node["id"]]["ciHead"] = node.get("ciHead")

    # page processing ----------------------------------------------------------
    def process_page(self, conn, cp, seen, with_ci, commit_stats=True, review_link=True):
        """Filter one page by window, complete nested connections, return (records, stop)."""
        records = []
        stop = False
        fetched_at = iso(self.clock())
        for node in conn["nodes"]:
            pos = window_position(node["createdAt"], self.window_start, self.window_end)
            if pos == "after":
                cp["skipped_after_window"] += 1
                continue
            if pos == "before":
                stop = True
                break
            if node["number"] in seen:
                continue
            self.complete_nested(node, cp, commit_stats, review_link)
            node["_meta"] = {
                "repo": self.repo, "fetched_at": fetched_at, "collector": COLLECTOR_VERSION,
                "ci_fetched": bool(with_ci),
                "ci_window": parse_dt(node["createdAt"]) >= parse_dt(self.ci_since),
            }
            if not review_link:
                node["_meta"]["thread_review_link_missing"] = True
            if not commit_stats:
                node["_meta"]["commit_stats_per_commit"] = True
                node["_meta"]["commit_stats_missing"] = self.fill_commit_stats(node, cp)
            records.append(node)
        if with_ci:
            self.fetch_ci_details(records, cp)
        return records, stop

    # main loop ----------------------------------------------------------------
    def run(self):
        cp = self.load_checkpoint()
        if cp.get("phase") == "done":
            self.log("already done (%d PRs)" % cp["n_written"])
            return cp
        seen = repair_and_index_jsonl(self.jsonl)
        if len(seen) != cp["n_written"]:
            self.log("jsonl has %d PRs, checkpoint says %d; using jsonl" % (len(seen), cp["n_written"]))
            cp["n_written"] = len(seen)
        cp["status"] = "running"
        cp["message"] = ""
        self.log("start: phase=%s pages=%d written=%d window=%s..%s ci_since=%s"
                 % (cp["phase"], cp["pages"], cp["n_written"], self.window_start,
                    self.window_end, self.ci_since))
        t_run = time.monotonic()
        elapsed0 = cp["elapsed_s"]
        try:
            if cp["phase"] == "locate":
                self.locate_start(cp)
                cp["elapsed_s"] = elapsed0 + time.monotonic() - t_run
                self.save_checkpoint(cp)
            commit_stats = review_link = True
            while cp["phase"] == "fetch":
                if self.stop_requested():
                    cp["status"] = "stopped"
                    self.log("stop requested; checkpoint saved")
                    break
                t0 = time.monotonic()
                points0 = cp["points"]
                with_ci = cp["last_created"] is None or \
                    parse_dt(cp["last_created"]) >= parse_dt(self.ci_since)
                page_size = cp["page_size"]
                is_small = page_size < self.default_page_size
                smaller = self.smaller_page_size(page_size)
                try:
                    data = self.client.execute(
                        build_page_query(with_ci, commit_stats, review_link),
                        {"owner": self.owner, "name": self.name, "cursor": cp["cursor"],
                         "pageSize": page_size},
                        max_attempts=2 if smaller else 8)
                except TransientError as e:   # includes ServerTimeoutError
                    if smaller:
                        cp["page_size"] = smaller
                        cp["small_pages_ok"] = 0
                        self.log("repeated server timeout (%s): page size %d -> %d"
                                 % (e, page_size, smaller))
                        continue
                    if commit_stats and isinstance(e, ServerTimeoutError):
                        commit_stats = False
                        self.log("server timeout at page size %d (%s): retrying without "
                                 "per-commit stats" % (page_size, e))
                        continue
                    if review_link and isinstance(e, ServerTimeoutError):
                        review_link = False
                        self.log("server error at page size %d without per-commit stats (%s): "
                                 "retrying without review links of thread comments" % (page_size, e))
                        continue
                    raise
                self._check_repo(data)
                conn = data["repository"]["pullRequests"]
                page_cost = (data.get("rateLimit") or {}).get("cost") or 0
                cp["points"] += page_cost
                records, stop = self.process_page(conn, cp, seen, with_ci, commit_stats,
                                                  review_link)
                commit_stats = review_link = True
                append_jsonl(self.jsonl, records)
                for r in records:
                    seen[r["number"]] = r["createdAt"]
                    if cp["first_created"] is None or r["createdAt"] > cp["first_created"]:
                        cp["first_created"] = r["createdAt"]
                if conn["nodes"]:
                    oldest = min(n["createdAt"] for n in conn["nodes"])
                    if cp["last_created"] is None or oldest < cp["last_created"]:
                        cp["last_created"] = oldest
                cp["n_written"] = len(seen)
                cp["cursor"] = conn["pageInfo"]["endCursor"]
                cp["pages"] += 1
                if is_small:
                    cp["small_pages_ok"] += 1
                    if cp["small_pages_ok"] >= RESTORE_AFTER_PAGES:
                        cp["page_size"] = self.larger_page_size(page_size)
                        cp["small_pages_ok"] = 0
                        self.log("page size raised to %d" % cp["page_size"])
                if stop or not conn["pageInfo"]["hasNextPage"]:
                    cp["phase"] = "done"
                    cp["status"] = "done"
                cp["elapsed_s"] = elapsed0 + time.monotonic() - t_run
                self.save_checkpoint(cp)
                rate = self.client.last_rate or {}
                self.log("page %d size=%d ci=%s +%d total=%d oldest=%s cost=%s pts=%d remaining=%s %.1fs"
                         % (cp["pages"], page_size, "y" if with_ci else "n", len(records),
                            cp["n_written"], cp["last_created"], page_cost,
                            cp["points"] - points0, rate.get("remaining"), time.monotonic() - t0))
        except Exception as e:
            cp["status"] = "error"
            cp["message"] = ("%s: %s" % (type(e).__name__, e))[:300]
            cp["elapsed_s"] = elapsed0 + time.monotonic() - t_run
            self.save_checkpoint(cp)
            self.log("ERROR %s" % cp["message"])
            raise
        cp["elapsed_s"] = elapsed0 + time.monotonic() - t_run
        self.save_checkpoint(cp)
        if cp["phase"] == "done":
            self.log("done: %d PRs, %d pages, %d points, %d follow-up queries, %.0fs"
                     % (cp["n_written"], cp["pages"], cp["points"], cp["followup_queries"],
                        cp["elapsed_s"]))
        return cp


# ── Repository metadata ─────────────────────────────────────────────────────

def fetch_repo_meta(client, owner, name):
    data = client.execute(REPO_META_QUERY, {"owner": owner, "name": name,
                                            "search": search_query_closed(owner, name),
                                            "searchCi": search_query_closed(owner, name, start=CI_SINCE)})
    repo = data.get("repository")
    if repo is None:
        raise GraphQLError("repository %s/%s not found" % (owner, name))
    langs = summarize_languages(repo["languages"])
    return {
        "repo": repo["nameWithOwner"], "is_archived": repo["isArchived"],
        "is_mirror": repo["isMirror"], "is_fork": repo["isFork"],
        "owner_type": repo["owner"]["__typename"], "repo_created_at": repo["createdAt"],
        "stars": repo["stargazerCount"],
        "github_primary_language": (repo.get("primaryLanguage") or {}).get("name"),
        "languages": langs,
        "search_closed_in_window": data["search"]["issueCount"],
        "search_closed_ci_part": data["ciPart"]["issueCount"],
        "fetched_at": iso(utcnow()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--repo", required=True, help="owner/name")
    p.add_argument("--raw-dir", default=RAW_V2_DIR)
    p.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    args = p.parse_args()
    owner, name = args.repo.split("/")
    os.makedirs(args.raw_dir, exist_ok=True)
    client = GitHubClient(load_token())
    RepoFetcher(client, owner, name, raw_dir=args.raw_dir, page_size=args.page_size).run()


if __name__ == "__main__":
    main()
