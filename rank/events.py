"""Flatten raw GitHub PR dumps into reviewer events for the ranking pipeline.

All timestamps are timezone-aware UTC datetimes. A reviewer "event" is the
first human interaction (review submission or thread comment) of a reviewer
on a PR; the author of the PR and bots never produce events.
"""
import re
from datetime import datetime

# Explicit bot markers only. A human handle that merely contains "bot" as a
# substring (e.g. "abbot") must not be dropped.
_BOT_RE = re.compile(
    r"(\[bot\]$|-bot$|^bot-|dependabot|renovate|codecov|github-actions|"
    r"copilot|coderabbit|cursor|greptile|sourcery|ellipsis|sweep-ai|"
    r"devin-ai|gemini-code-assist|graphite-app|mergify|vercel|netlify|"
    r"changeset-bot|kodiak|pre-commit-ci|allcontributors)",
    re.IGNORECASE,
)


def is_bot(login):
    return bool(login) and bool(_BOT_RE.search(login))


def parse_ts(s):
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _author_login(obj):
    a = (obj or {}).get("author") or {}
    return a.get("login") or ""


def _interactions(pr):
    """Yield (login, datetime) for every human, non-author review interaction."""
    author = _author_login(pr)
    for r in (pr.get("reviews") or {}).get("nodes") or []:
        login = _author_login(r)
        t = parse_ts(r.get("submittedAt"))
        if login and t and login != author and not is_bot(login):
            yield login, t
    for th in (pr.get("reviewThreads") or {}).get("nodes") or []:
        for c in (th.get("comments") or {}).get("nodes") or []:
            login = _author_login(c)
            t = parse_ts(c.get("createdAt"))
            if login and t and login != author and not is_bot(login):
                yield login, t


def positives(pr):
    """Human reviewers who actually took part in the review of this PR."""
    return {login for login, _ in _interactions(pr)}


def review_events(prs):
    """One row per (reviewer, PR) with the reviewer's first interaction time."""
    rows = []
    for pr in prs:
        first = {}
        for login, t in _interactions(pr):
            if login not in first or t < first[login]:
                first[login] = t
        created = parse_ts(pr.get("createdAt"))
        merged = parse_ts(pr.get("mergedAt"))
        for login, t in first.items():
            rows.append({
                "reviewer": login,
                "pr": pr["number"],
                "author": _author_login(pr),
                "first_at": t,
                "pr_created": created,
                "pr_merged": merged,
            })
    return rows
