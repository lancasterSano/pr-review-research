"""Review iterations under the literal HAIT Definition 2 (review_iterations.py).

Rule: a new iteration needs commits after a human review AND a later human review.
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

import review_iterations as ri  # noqa: E402
from review_rounds import compute_review_rounds  # noqa: E402
from taskF1_duration_language import is_human  # noqa: E402

T0 = datetime(2025, 6, 2, 9, 0, tzinfo=timezone.utc)


def iso(minutes):
    return (T0 + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


def pr(author="alice", reviews=(), commits=(), state="MERGED"):
    """reviews: (login, minutes, state); commits: authoredDate in minutes from creation."""
    return {
        "number": 7, "state": state, "author": {"login": author},
        "createdAt": iso(0), "mergedAt": iso(3000),
        "reviews": {"nodes": [{"author": {"login": a}, "state": s, "submittedAt": iso(m)}
                              for a, m, s in reviews]},
        "commits": {"nodes": [{"commit": {"authoredDate": iso(m)}} for m in commits]},
    }


def human_literal(p):
    return ri.count_iterations(p, is_human)


def test_default_rule_is_literal():
    # Fails if the default falls back to the code rule.
    assert ri.count_iterations.__defaults__[-1] == ri.RULE_LITERAL


def test_commits_after_approval_then_merge_is_not_an_iteration():
    # bob approves, alice pushes a fix, the PR is merged with no new review.
    p = pr(reviews=[("bob", 60, "APPROVED")], commits=[120])
    assert human_literal(p) == 1                                   # fails if the merge closes an iteration
    assert ri.count_iterations(p, is_human, rule=ri.RULE_CODE) == 2  # earlier code rule
    assert ri.iteration_trace(p, is_human) == (1, True)


def test_commits_after_changes_requested_and_new_review_is_an_iteration():
    p = pr(reviews=[("bob", 60, "CHANGES_REQUESTED"), ("bob", 200, "APPROVED")], commits=[120])
    assert human_literal(p) == 2
    assert ri.count_iterations(p, is_human, rule=ri.RULE_CODE) == 2


def test_comment_only_review_followed_by_rework_and_review_counts():
    # All three verdicts count, not only CHANGES_REQUESTED.
    p = pr(reviews=[("bob", 60, "COMMENTED"), ("carol", 200, "APPROVED")], commits=[120, 130])
    assert human_literal(p) == 2


@pytest.mark.parametrize("bot", ["copilot-pull-request-reviewer", "coderabbitai[bot]",
                                 "github-actions", "some-helper[bot]", "grafana-pr-approver",
                                 "woocommercebot", "angular-builds"])
def test_assistant_or_bot_review_after_rework_does_not_open_an_iteration(bot):
    # Human review, push, then only an assistant / bot / service account reviews.
    p = pr(reviews=[("bob", 60, "COMMENTED"), (bot, 130, "COMMENTED")], commits=[120])
    assert human_literal(p) == 1                                      # fails if the bot closes the loop
    assert ri.count_iterations(p, ri.include_all) == 2                # all events: it would


def test_assistant_review_before_first_human_review_is_ignored():
    p = pr(reviews=[("copilot-pull-request-reviewer", 5, "COMMENTED"), ("bob", 60, "APPROVED")],
           commits=[30])
    assert human_literal(p) == 1


def test_self_review_does_not_count():
    # The author's own review after the push neither opens nor closes an iteration.
    p = pr(reviews=[("bob", 60, "COMMENTED"), ("alice", 130, "COMMENTED")], commits=[120])
    assert human_literal(p) == 1
    only_self = pr(reviews=[("alice", 60, "APPROVED")], commits=[120])
    assert human_literal(only_self) == 0


def test_no_human_review_is_zero_iterations():
    p = pr(reviews=[("coderabbitai", 10, "COMMENTED")], commits=[30])
    assert human_literal(p) == 0
    assert human_literal(pr(commits=[30])) == 0


def test_several_reviews_without_push_are_one_iteration():
    p = pr(reviews=[("bob", 60, "COMMENTED"), ("carol", 70, "COMMENTED"), ("bob", 80, "APPROVED")],
           commits=[10, 20])  # commits before the first review: initial submission
    assert human_literal(p) == 1


def test_commit_authored_before_creation_is_ignored():
    # A rebase keeps an old authoredDate: such a commit is not rework.
    p = pr(reviews=[("bob", 60, "COMMENTED"), ("bob", 200, "APPROVED")], commits=[-10])
    assert human_literal(p) == 1


def test_review_and_commit_at_same_moment_commit_counts_after_review():
    # Tie: review first, then commit (as compute_review_rounds orders them).
    p = pr(reviews=[("bob", 60, "COMMENTED"), ("bob", 90, "APPROVED")], commits=[60])
    assert human_literal(p) == 2


def test_three_rounds_with_trailing_push():
    p = pr(reviews=[("bob", 60, "CHANGES_REQUESTED"), ("bob", 200, "CHANGES_REQUESTED"),
                    ("bob", 400, "APPROVED")], commits=[100, 300, 500])
    assert human_literal(p) == 3
    assert ri.count_iterations(p, is_human, rule=ri.RULE_CODE) == 4


def test_open_pr_rules_agree():
    # The code rule adds the trailing iteration only for merged PRs.
    p = pr(reviews=[("bob", 60, "APPROVED")], commits=[120], state="OPEN")
    assert human_literal(p) == ri.count_iterations(p, is_human, rule=ri.RULE_CODE) == 1


@pytest.mark.parametrize("p", [
    pr(reviews=[("bob", 60, "APPROVED")], commits=[120]),
    pr(reviews=[("bob", 60, "COMMENTED"), ("copilot-pull-request-reviewer", 130, "COMMENTED")],
       commits=[120, 140]),
    pr(reviews=[("alice", 5, "COMMENTED"), ("bob", 60, "CHANGES_REQUESTED"), ("bob", 200, "APPROVED")],
       commits=[-5, 30, 120, 250]),
    pr(reviews=[], commits=[30]),
])
def test_code_rule_with_all_events_reproduces_published_algorithm(p):
    # Fails if review_iterations drifts from enrich/review_rounds.compute_review_rounds.
    assert ri.count_iterations(p, ri.include_all, rule=ri.RULE_CODE) == compute_review_rounds(p)


def test_unknown_rule_raises():
    with pytest.raises(ValueError):
        ri.count_iterations(pr(), is_human, rule="strict")
