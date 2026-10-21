"""ReviewerHistory answers 'what was known about a reviewer at time t0'.

Every query must ignore anything that happened at or after t0.
"""
from datetime import datetime

import math
import pytest

from rank.history import ReviewerHistory, AuthorHistory


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def ev(reviewer, pr, author, first_at, created, merged):
    return {"reviewer": reviewer, "pr": pr, "author": author,
            "first_at": ts(first_at), "pr_created": ts(created), "pr_merged": ts(merged)}


T0 = ts("2026-06-01T00:00:00Z")


@pytest.fixture
def hist():
    return ReviewerHistory([
        # bob: active 10 days before t0, on a PR still open at t0 (workload 1)
        ev("bob", 1, "alice", "2026-05-22T00:00:00Z", "2026-05-21T00:00:00Z", "2026-06-10T00:00:00Z"),
        # bob: reviewed alice's PR 100 days before t0, latency 48h, merged before t0
        ev("bob", 2, "alice", "2026-02-23T00:00:00Z", "2026-02-21T00:00:00Z", "2026-03-01T00:00:00Z"),
        # bob: reviewed a PR created before t0 but reviewed AFTER t0 (must be invisible)
        ev("bob", 3, "dave", "2026-06-02T00:00:00Z", "2026-05-30T00:00:00Z", "2026-06-05T00:00:00Z"),
        # carol: only activity 200 days before t0
        ev("carol", 4, "alice", "2025-11-13T00:00:00Z", "2025-11-12T00:00:00Z", "2025-11-20T00:00:00Z"),
        # erin: only activity after t0
        ev("erin", 5, "alice", "2026-06-03T00:00:00Z", "2026-06-02T00:00:00Z", "2026-06-04T00:00:00Z"),
    ])


class TestCandidates:
    def test_reviewer_active_within_window_is_candidate(self, hist):
        assert "bob" in hist.candidates(T0, window_days=180)

    def test_reviewer_active_only_before_window_is_not_candidate(self, hist):
        assert "carol" not in hist.candidates(T0, window_days=180)

    def test_reviewer_active_only_after_t0_is_not_candidate(self, hist):
        assert "erin" not in hist.candidates(T0, window_days=180)

    def test_event_exactly_at_t0_is_future(self, hist):
        h = ReviewerHistory([ev("x", 9, "a", "2026-06-01T00:00:00Z", "2026-05-31T00:00:00Z", "2026-06-02T00:00:00Z")])
        assert h.candidates(T0) == []


class TestActivity:
    def test_counts_events_in_window_only(self, hist):
        assert hist.activity("bob", T0, days=30) == 1
        assert hist.activity("bob", T0, days=180) == 2

    def test_total_before_t0(self, hist):
        assert hist.total_activity("bob", T0) == 2

    def test_days_since_last_event(self, hist):
        assert hist.days_since_last("bob", T0) == pytest.approx(10.0)

    def test_unknown_reviewer_has_no_activity(self, hist):
        assert hist.activity("nobody", T0, days=30) == 0
        assert math.isnan(hist.days_since_last("nobody", T0))


class TestWorkload:
    def test_engagement_started_long_ago_is_ignored(self):
        h = ReviewerHistory([ev('z', 1, 'a', '2025-01-01T00:00:00Z', '2024-12-31T00:00:00Z', '2026-12-31T00:00:00Z')])
        assert h.workload('z', T0, days=180) == 0

    def test_open_engagement_counts(self, hist):
        # PR 1: reviewed before t0, merged after t0 -> open engagement
        # PR 2: merged before t0 -> not counted; PR 3: reviewed after t0 -> not counted
        assert hist.workload("bob", T0) == 1

    def test_no_engagements(self, hist):
        assert hist.workload("carol", T0) == 0


class TestLatency:
    def test_median_latency_uses_reviews_submitted_in_window_before_t0(self, hist):
        # window 120 days by review time: PR 2 (48h, reviewed 98 d ago) and PR 1 (24h) -> median 36h; PR 3 excluded (reviewed after t0)
        assert hist.latency_median_hours("bob", T0, days=120) == pytest.approx(36.0)

    def test_median_latency_short_window(self, hist):
        assert hist.latency_median_hours("bob", T0, days=30) == pytest.approx(24.0)

    def test_no_history_gives_nan(self, hist):
        assert math.isnan(hist.latency_median_hours("carol", T0, days=30))

    def test_fast_share_within_24h(self, hist):
        # PR 1 latency 24h (counts as <= 24h), PR 2 latency 48h
        assert hist.fast_share("bob", T0, days=120, hours=24) == pytest.approx(0.5)


class TestFamiliarity:
    def test_counts_prior_prs_of_same_author(self, hist):
        assert hist.familiarity("bob", "alice", T0) == 2
        assert hist.familiarity("bob", "dave", T0) == 0  # PR 3 is after t0


class TestAuthorHistory:
    def test_prior_pr_count_and_median_time_to_first_review(self):
        prs = [
            {"number": 1, "author": {"login": "alice"}, "createdAt": "2026-01-01T00:00:00Z", "first_review_at": ts("2026-01-02T00:00:00Z")},
            {"number": 2, "author": {"login": "alice"}, "createdAt": "2026-02-01T00:00:00Z", "first_review_at": ts("2026-02-01T12:00:00Z")},
            {"number": 3, "author": {"login": "alice"}, "createdAt": "2026-05-31T00:00:00Z", "first_review_at": ts("2026-06-02T00:00:00Z")},  # reviewed after t0
            {"number": 4, "author": {"login": "bob"}, "createdAt": "2026-01-01T00:00:00Z", "first_review_at": None},
        ]
        ah = AuthorHistory(prs)
        assert ah.prior_prs("alice", T0) == 3
        assert ah.median_hours_to_first_review("alice", T0) == pytest.approx(18.0)
        assert ah.prior_prs("bob", T0) == 1
        assert math.isnan(ah.median_hours_to_first_review("bob", T0))
