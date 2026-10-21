"""Leak-critical building blocks of the reviewer-ranking pipeline.

Every test names the production change that would make it fail.
"""
from datetime import datetime


from rank import events


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def make_pr(number, author, created, merged, reviews=(), threads=()):
    return {
        "number": number,
        "author": {"login": author},
        "createdAt": created,
        "mergedAt": merged,
        "reviews": {"nodes": [
            {"author": {"login": a}, "state": st, "submittedAt": t, "body": ""}
            for (a, st, t) in reviews
        ]},
        "reviewThreads": {"nodes": [
            {"path": p, "comments": {"nodes": [
                {"author": {"login": a}, "createdAt": t, "body": ""}]}}
            for (a, t, p) in threads
        ]},
        "reviewRequests": {"nodes": []},
    }


class TestIsBot:
    def test_bracket_bot_suffix_is_bot(self):
        assert events.is_bot("dependabot[bot]")

    def test_known_ai_tools_are_bots(self):
        assert events.is_bot("coderabbitai")
        assert events.is_bot("copilot-pull-request-reviewer")

    def test_plain_human_login_is_not_bot(self):
        assert not events.is_bot("alice")

    def test_login_containing_bot_as_a_word_only(self):
        # 'abbot' is a human handle; only explicit bot markers count
        assert not events.is_bot("abbot")


class TestPositives:
    def test_review_authors_are_positives(self):
        pr = make_pr(1, "alice", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z",
                     reviews=[("bob", "APPROVED", "2026-01-01T10:00:00Z")])
        assert events.positives(pr) == {"bob"}

    def test_thread_commenter_is_positive(self):
        pr = make_pr(1, "alice", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z",
                     threads=[("carol", "2026-01-01T11:00:00Z", "a.py")])
        assert events.positives(pr) == {"carol"}

    def test_author_is_never_positive(self):
        pr = make_pr(1, "alice", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z",
                     reviews=[("alice", "COMMENTED", "2026-01-01T10:00:00Z")],
                     threads=[("alice", "2026-01-01T11:00:00Z", "a.py")])
        assert events.positives(pr) == set()

    def test_bots_are_never_positive(self):
        pr = make_pr(1, "alice", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z",
                     reviews=[("codecov[bot]", "COMMENTED", "2026-01-01T10:00:00Z")])
        assert events.positives(pr) == set()


class TestReviewEvents:
    """events.review_events flattens one row per (reviewer, PR): first human
    interaction time of that reviewer on that PR."""

    def test_first_interaction_time_per_reviewer(self):
        pr = make_pr(7, "alice", "2026-01-01T00:00:00Z", "2026-01-05T00:00:00Z",
                     reviews=[("bob", "COMMENTED", "2026-01-02T00:00:00Z"),
                              ("bob", "APPROVED", "2026-01-03T00:00:00Z")],
                     threads=[("bob", "2026-01-01T12:00:00Z", "a.py")])
        rows = events.review_events([pr])
        assert len(rows) == 1
        r = rows[0]
        assert r["reviewer"] == "bob" and r["pr"] == 7
        assert r["first_at"] == ts("2026-01-01T12:00:00Z")
        assert r["pr_created"] == ts("2026-01-01T00:00:00Z")
        assert r["pr_merged"] == ts("2026-01-05T00:00:00Z")
        assert r["author"] == "alice"

    def test_review_without_timestamp_is_dropped(self):
        pr = make_pr(7, "alice", "2026-01-01T00:00:00Z", "2026-01-05T00:00:00Z",
                     reviews=[("bob", "COMMENTED", None)])
        assert events.review_events([pr]) == []
