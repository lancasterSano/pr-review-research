import math
import pytest

from rank.build import build_repo
from rank.tests.test_events import make_pr


def _prs():
    prs = []
    # 10 warm-up-ish PRs by alice reviewed by bob over Jan-Apr 2026 (one per 10 days)
    from datetime import datetime, timedelta, timezone
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fmt = lambda d: d.strftime("%Y-%m-%dT%H:%M:%SZ")
    for i in range(10):
        c = base + timedelta(days=10 * i)
        prs.append(make_pr(i + 1, "alice", fmt(c), fmt(c + timedelta(hours=20)),
                           reviews=[("bob", "APPROVED", fmt(c + timedelta(hours=5)))]))
    # a later PR by dave reviewed by carol; bob is active -> both candidates
    prs.append(make_pr(50, "alice", "2026-05-01T00:00:00Z", "2026-05-03T00:00:00Z",
                       reviews=[("carol", "COMMENTED", "2026-05-01T04:00:00Z")]))
    prs.append(make_pr(51, "dave", "2026-05-20T00:00:00Z", "2026-05-22T00:00:00Z",
                       reviews=[("carol", "APPROVED", "2026-05-21T00:00:00Z")]))
    return prs


class TestBuildRepo:
    def test_pairs_have_labels_features_and_split(self):
        pairs, prs = build_repo("toy", _prs(), warmup_days=30)
        assert set(["repo", "pr", "candidate", "label", "split"]).issubset(pairs.columns)
        last = pairs[pairs.pr == 51]
        assert set(last.candidate) == {"bob", "carol"}
        assert last.set_index("candidate").label.to_dict() == {"bob": 0, "carol": 1}
        assert (last.split == "test").all()

    def test_strict_features_present_and_leak_free(self):
        pairs, prs = build_repo("toy", _prs(), warmup_days=30)
        row = pairs[(pairs.pr == 51) & (pairs.candidate == "bob")].iloc[0]
        for col in ["act_90d", "act_total", "days_since_last", "workload", "lat_med_h", "fast_24h",
                    "familiarity", "author_prior_prs", "author_med_first_review_h", "hour", "weekday"]:
            assert col in pairs.columns, col
        assert row.familiarity == 0          # bob never reviewed dave before
        assert row.act_total == 10
        carol = pairs[(pairs.pr == 51) & (pairs.candidate == "carol")].iloc[0]
        assert carol.familiarity == 0 and carol.act_total == 1 and carol.lat_med_h == pytest.approx(4.0)

    def test_pr_table_reports_candidate_recall(self):
        pairs, prs = build_repo("toy", _prs(), warmup_days=30)
        r = prs.set_index("pr").loc[51]
        assert r.n_candidates == 2 and r.n_positives == 1 and r.any_positive_in_pool == 1 and r.all_positives_in_pool == 1


class TestRealizedLatency:
    def test_positive_pairs_carry_observed_first_response_hours(self):
        pairs, prs = build_repo("toy", _prs(), warmup_days=30)
        carol = pairs[(pairs.pr == 51) & (pairs.candidate == "carol")].iloc[0]
        bob = pairs[(pairs.pr == 51) & (pairs.candidate == "bob")].iloc[0]
        assert carol.realized_lat_h == pytest.approx(24.0)   # reviewed 2026-05-21 00:00, PR created 05-20 00:00
        assert math.isnan(bob.realized_lat_h)
