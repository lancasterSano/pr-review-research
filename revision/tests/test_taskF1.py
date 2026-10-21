"""Human-only review characteristics for the ITIA-2026 F1 analysis.

Each test names the production change that would make it fail.
Run from the project root:  .venv/bin/python -m pytest revision/tests -q
"""
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

_REV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REV)
sys.path.insert(0, os.path.join(os.path.dirname(_REV), "enrich"))

import taskF1_duration_language as f1  # noqa: E402
from review_rounds import compute_review_rounds  # noqa: E402

T0 = datetime(2025, 6, 2, 9, 0, tzinfo=timezone.utc)


def iso(minutes):
    return (T0 + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_pr(author, reviews=(), commits=(), threads=(), merged_min=24 * 60, state="MERGED"):
    """reviews: (login|None, minutes, state); commits: minutes; threads: (login, minutes)."""
    return {
        "number": 1,
        "state": state,
        "author": {"login": author},
        "createdAt": iso(0),
        "mergedAt": iso(merged_min),
        "reviews": {"nodes": [
            {"author": ({"login": a} if a is not None else None), "state": s,
             "submittedAt": iso(m), "body": ""}
            for a, m, s in reviews]},
        "commits": {"nodes": [{"commit": {"authoredDate": iso(m), "message": ""}} for m in commits],
                    "totalCount": len(commits)},
        "reviewThreads": {"nodes": [
            {"comments": {"nodes": [{"author": {"login": a}, "createdAt": iso(m)}]}}
            for a, m in threads]},
    }


# PR 1: Copilot reviews first and after every push; one human reviews twice.
PR_ASSISTANT = make_pr(
    "alice",
    reviews=[("copilot-pull-request-reviewer", 5, "COMMENTED"),
             ("bob", 180, "COMMENTED"),
             ("copilot-pull-request-reviewer", 230, "COMMENTED"),
             ("bob", 300, "APPROVED")],
    commits=[60, 240],
)
# PR 2: self-review, CI bot, unknown *[bot] app, then a human; a commit after the
# human review with no later review (boundary case of Definition 2).
PR_BOTS = make_pr(
    "carol",
    reviews=[("carol", 10, "COMMENTED"),
             ("github-actions", 20, "COMMENTED"),
             ("some-helper[bot]", 30, "COMMENTED"),
             ("dave", 120, "APPROVED")],
    commits=[180],
)
# PR 3: only AI assistants ever review.
PR_AI_ONLY = make_pr(
    "erin",
    reviews=[("coderabbitai", 3, "COMMENTED"), ("coderabbitai", 90, "COMMENTED")],
    commits=[45],
)
# PR 4: automated approver from the audit list, then a deleted ("ghost") human account.
PR_SERVICE = make_pr(
    "grafana-pr-automation",
    reviews=[("grafana-pr-approver", 1, "APPROVED"), (None, 600, "APPROVED")],
)


@pytest.mark.parametrize("login,expected", [
    ("bob", "human"),
    ("copilot-pull-request-reviewer", "ai_assistant"),
    ("coderabbitai[bot]", "ai_assistant"),
    ("github-actions", "other_bot"),
    ("some-helper[bot]", "other_bot"),      # fails if the *[bot] suffix rule is dropped
    ("grafana-pr-approver", "other_bot"),   # fails if EXTRA_AUTOMATION is dropped
    ("", "human"),                          # deleted account
])
def test_participant_class(login, expected):
    assert f1.participant_class(login) == expected


def test_first_human_reaction_skips_assistant():
    # Fails if t_fr counts the Copilot review at +5 min.
    assert f1.first_reaction_hours(PR_ASSISTANT, f1.is_human) == pytest.approx(3.0)
    assert f1.first_reaction_hours(PR_ASSISTANT, f1.include_all) == pytest.approx(5 / 60)


def test_first_human_reaction_skips_self_review_and_bots():
    # Fails if self-reviews, CI bots or unknown *[bot] apps count as a human reaction.
    assert f1.first_reaction_hours(PR_BOTS, f1.is_human) == pytest.approx(2.0)
    # The published definition starts the clock at the CI bot (+20 min), never the author.
    assert f1.first_reaction_hours(PR_BOTS, f1.include_all) == pytest.approx(20 / 60)


def test_no_human_reaction_is_none():
    assert f1.first_reaction_hours(PR_AI_ONLY, f1.is_human) is None
    assert f1.first_reaction_hours(PR_SERVICE, f1.is_human) == pytest.approx(10.0)


def test_human_iterations_ignore_assistant_reviews():
    # All events in time order: r5, c60, r180, r230, c240, r300 -> 3 iterations.
    assert f1.count_iterations(PR_ASSISTANT, f1.include_all, rule="code") == 3
    assert f1.count_iterations(PR_ASSISTANT, f1.include_all) == 3
    # Humans only: commit(+60) precedes the first human review and is ignored;
    # r180, c240, r300 -> 2 iterations. Fails if assistant reviews open iterations.
    assert f1.count_iterations(PR_ASSISTANT, f1.is_human) == 2


def test_boundary_commit_after_last_review():
    # Human review at +120, commit at +180, merged with no further review.
    # Literal HAIT Definition 2 (default, decision 14.10): no new iteration.
    assert f1.count_iterations(PR_BOTS, f1.is_human) == 1
    # The earlier code rule adds one for a merged PR (published HAIT numbers).
    assert f1.count_iterations(PR_BOTS, f1.is_human, rule="code") == 2
    open_pr = dict(PR_BOTS, state="OPEN")
    assert f1.count_iterations(open_pr, f1.is_human, rule="code") == 1


def test_ai_only_pr_has_zero_human_iterations():
    assert f1.count_iterations(PR_AI_ONLY, f1.is_human) == 0
    assert f1.count_iterations(PR_AI_ONLY, f1.include_all, rule="code") == 2


@pytest.mark.parametrize("pr", [PR_ASSISTANT, PR_BOTS, PR_AI_ONLY, PR_SERVICE])
def test_include_all_reproduces_published_algorithm(pr):
    # Fails if the parameterised copy drifts from enrich/review_rounds.py.
    assert f1.count_iterations(pr, f1.include_all, rule="code") == compute_review_rounds(pr)


def test_staleness_trace_without_assistant_events():
    # Assistant comments every ~20 days keep the PR "active"; without them the
    # longest silence is 60 days.
    pr = make_pr("alice",
                 reviews=[("bob", 60 * 24 * 60, "APPROVED")],
                 threads=[("coderabbitai", 20 * 24 * 60), ("coderabbitai", 40 * 24 * 60)],
                 merged_min=61 * 24 * 60)
    gap_all = f1.max_gap_days(f1.activity_trace(pr, f1.include_all))
    gap_h = f1.max_gap_days(f1.activity_trace(pr, lambda w: f1.participant_class(w) == "human"))
    assert gap_all == pytest.approx(20.0)
    assert gap_h == pytest.approx(60.0)


def test_assistant_summary():
    s = f1.assistant_summary(PR_ASSISTANT)
    assert s["has_ai_review"] == 1
    assert s["first_reactor_class"] == "ai_assistant"
    assert s["t_first_ai_h"] == pytest.approx(5 / 60)
    s2 = f1.assistant_summary(PR_BOTS)
    assert s2["has_ai_review"] == 0
    assert s2["first_reactor_class"] == "other_bot"
    assert s2["t_first_auto_h"] == pytest.approx(20 / 60)


# ── commit count at submission and "as known at merge" history (decision 14.10) ──

def test_commits_at_submission_counts_only_commits_authored_by_t0():
    # Commits at -30 and 0 min exist at creation; +60 and +240 come after it.
    # Fails if the final count (4) or a strict "< t0" comparison is used.
    pr = make_pr("alice", commits=[-30, 0, 60, 240])
    assert f1.commits_at_submission(pr) == 2
    assert pr["commits"]["totalCount"] == 4


def test_truncate_until_drops_post_merge_review():
    # Human review only after the merge (+25 h, merged at +24 h): known-at-merge
    # history must not see it. Fails if truncate_until keeps events after `until`.
    pr = make_pr("alice", reviews=[("bob", 25 * 60, "COMMENTED")], commits=[30],
                 threads=[("bob", 25 * 60)], merged_min=24 * 60)
    merged = f1.parse_dt(pr["mergedAt"])
    pre = f1.truncate_until(pr, merged)
    assert f1.first_reaction_hours(pr, f1.is_human) == pytest.approx(25.0)
    assert f1.first_reaction_hours(pre, f1.is_human) is None
    assert f1.count_iterations(pre, f1.is_human) == 0
    assert len(pre["commits"]["nodes"]) == 1
    assert pre["reviewThreads"]["nodes"][0]["comments"]["nodes"] == []
    # the input PR is not modified
    assert len(pr["reviews"]["nodes"]) == 1


def test_truncate_until_keeps_pre_merge_events_unchanged():
    merged = f1.parse_dt(PR_ASSISTANT["mergedAt"])
    pre = f1.truncate_until(PR_ASSISTANT, merged)
    assert f1.count_iterations(pre, f1.is_human) == f1.count_iterations(PR_ASSISTANT, f1.is_human)
    assert f1.first_reaction_hours(pre, f1.is_human) == f1.first_reaction_hours(PR_ASSISTANT, f1.is_human)


def test_commit_floor_replaces_rewritten_zero_with_one():
    # All commits are dated after creation (force-push / squash after submission):
    # the raw count at t0 is 0, the primary feature is 1. Counts >= 1 are kept.
    # Fails if the floor is dropped, applied as "+1", or caps larger counts.
    rewritten = make_pr("alice", commits=[60, 240])
    assert f1.commits_at_submission(rewritten) == 0
    assert f1.floor_commits(f1.commits_at_submission(rewritten)) == 1.0
    kept = make_pr("alice", commits=[-30, 0, 60])
    assert f1.floor_commits(f1.commits_at_submission(kept)) == 2.0
    assert list(f1.floor_commits([0, 1, 3, 0])) == [1.0, 1.0, 3.0, 1.0]


def test_configure_switches_output_suffix():
    # Fails if a sensitivity run (t0 without the floor, or the final count)
    # overwrites the primary (t0min1) outputs.
    try:
        f1.configure("final")
        assert f1.out_path("numbers.json").endswith("numbers_commits-final.json")
        assert f1.METRICS_LONG.endswith("F-metrics-long_commits-final.jsonl")
        f1.configure("t0")
        assert f1.out_path("numbers.json").endswith("numbers_commits-t0.json")
        assert f1.METRICS_LONG.endswith("F-metrics-long_commits-t0.jsonl")
        f1.configure("t0min1")
        assert f1.out_path("numbers.json").endswith(os.sep + "numbers.json")
        assert f1.METRICS_LONG.endswith(os.sep + "F-metrics-long.jsonl")
        assert f1.MODE_MATRIX["t0min1"] == "t0_floor1"
        with pytest.raises(ValueError):
            f1.configure("latest")
    finally:
        f1.configure("t0min1")


def test_default_mode_is_floored_t0():
    # Fails if the module default falls back to the unfloored or final count.
    assert f1.COMMITS_MODE == "t0min1" and f1.SUFFIX == ""
