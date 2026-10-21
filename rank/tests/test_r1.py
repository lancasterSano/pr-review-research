"""Review round 1: most-familiar baseline and paired realized latency (synthetic data)."""
import math

import numpy as np
import pandas as pd
import pytest

from rank import evaluate as ev
from rank import run_r1 as R


def _pairs():
    # repo r1: pr 1 and 2; repo r2: pr 3. alice is top-1 of the ranker in both repos.
    rows = [
        # repo, pr, candidate, label, familiarity, act, relevance, expected lat, realized lat
        ("r1", 1, "alice", 1, 0, 9, 0.90, 20.0, 5.0),
        ("r1", 1, "bob", 1, 3, 1, 0.85, 2.0, 1.0),
        ("r1", 1, "carol", 0, 3, 2, 0.10, 1.0, np.nan),
        ("r1", 2, "alice", 1, 1, 9, 0.95, 20.0, 4.0),
        ("r1", 2, "bob", 0, 0, 1, 0.20, 2.0, np.nan),
        ("r1", 2, "carol", 0, 1, 2, 0.30, 1.0, np.nan),
        ("r2", 3, "alice", 1, 0, 5, 0.80, 3.0, 2.0),
        ("r2", 3, "dave", 0, 0, 7, 0.40, 9.0, np.nan),
    ]
    return pd.DataFrame(rows, columns=["repo", "pr", "candidate", "label", "familiarity", "s_most_active",
                                       "s_pooled_STRICT", "lat_med_h", "realized_lat_h"])


class TestMostFamiliar:
    def test_familiarity_first_then_activity(self):
        d = _pairs()
        s = pd.Series(ev.most_familiar_score(d).to_numpy(), index=d.candidate)[d.pr.to_numpy() == 1]
        # bob and carol share familiarity 3; carol is more active -> carol first, alice (0) last
        assert s.sort_values(ascending=False).index.tolist() == ["carol", "bob", "alice"]

    def test_activity_never_outweighs_familiarity(self):
        d = pd.DataFrame({"familiarity": [1, 0], "s_most_active": [0, 10**7]})
        s = ev.most_familiar_score(d).to_numpy()
        assert s[0] > s[1]

    def test_added_as_baseline_only_when_familiarity_exists(self):
        base = pd.DataFrame({"repo": ["r"] * 3, "pr": [1, 1, 1], "candidate": ["alice", "bob", "carol"],
                             "label": [1, 0, 0], "act_90d": [5, 3, 1], "days_since_last": [1, 2, 3],
                             "workload": [0, 1, 2], "lat_med_h": [1.0, 2.0, 3.0]})
        without = ev.add_baseline_scores(base, seed=0)
        with_fam = ev.add_baseline_scores(base.assign(familiarity=[0, 2, 2]), seed=0)
        assert "s_most_familiar" not in without and "s_most_familiar" in with_fam
        for col in ["s_random", "s_most_active", "s_most_recent", "s_least_loaded_active", "s_fastest"]:
            assert np.array_equal(without[col].to_numpy(), with_fam[col].to_numpy())
        assert with_fam.set_index("candidate").s_most_familiar.idxmax() == "bob"
        assert ev.BASELINES["s_most_familiar"] == "Most familiar"


class TestSummaries:
    def test_distinct_top1_counted_over_logins_as_in_table4(self):
        _, micro, per_repo = R.summary_rows(_pairs(), {"ranker": "s_pooled_STRICT"}, n_boot=50)
        row = micro.iloc[0]
        # alice is top-1 in r1 and r2: one login overall, two in the per-repository sum
        assert row.top1_unique == 1 and row.top1_unique_repo_sum == 2
        assert set(per_repo.repo) == {"r1", "r2"}
        assert "gini_top1_full" in micro.columns

    def test_no_familiarity_share(self):
        out = R.no_familiarity_share(_pairs()).set_index("repo")
        assert out.loc["r2", "share_no_familiar_candidate"] == 1.0
        assert out.loc["r1", "share_no_familiar_candidate"] == 0.0
        assert out.loc["ALL", "share_no_familiar_candidate"] == pytest.approx(1 / 3)


class TestPairedDeltas:
    def test_mean_difference_per_repo_and_overall(self):
        a = pd.DataFrame({"repo": ["r1", "r1", "r2"], "pr": [1, 2, 3], "hit3": [1.0, 1.0, 0.0], "mrr": [1.0, 0.5, 0.0]})
        b = pd.DataFrame({"repo": ["r2", "r1", "r1"], "pr": [3, 2, 1], "hit3": [0.0, 0.0, 1.0], "mrr": [0.0, 0.5, 0.5]})
        out = R.paired_deltas(a, b, n_boot=200).set_index(["repo", "metric"])
        assert out.loc[("r1", "hit3"), "delta"] == pytest.approx(0.5)
        assert out.loc[("ALL", "mrr"), "delta"] == pytest.approx(0.5 / 3)
        assert out.loc[("r2", "mrr"), "delta"] == 0.0 and out.loc[("ALL", "hit3"), "n"] == 3
        assert (out.lo <= out.delta + 1e-12).all() and (out.delta <= out.hi + 1e-12).all()


class TestPairedLatency:
    def _m(self, top1, lat):
        return pd.DataFrame({"repo": "r", "pr": range(1, len(top1) + 1), "top1": top1, "top1_realized_lat": lat})

    def test_common_subset_and_different_top1(self):
        rel = self._m(["alice", "bob", "carol", "dave"], [2.0, 10.0, np.nan, 4.0])
        lat = self._m(["alice", "carol", "carol", "bob"], [2.0, 4.0, np.nan, 6.0])
        out = R.paired_latency(rel, lat, n_boot=100).set_index("quantity")
        assert out.loc["common_rel_only_median_h", "n"] == 3
        assert out.loc["common_same_top1", "n"] == 1
        assert out.loc["same_top1_share_all_prs", "value"] == pytest.approx(0.5)
        assert out.loc["diff_top1_rel_only_median_h", "value"] == pytest.approx(7.0)
        assert out.loc["diff_top1_constr_lat_median_h", "value"] == pytest.approx(5.0)
        assert out.loc["diff_top1_constr_lat_faster_share", "value"] == pytest.approx(0.5)
        # paired differences: 4-10 = -6 and 6-4 = +2 -> median -2
        assert out.loc["diff_top1_paired_median_diff_h", "value"] == pytest.approx(-2.0)

    def test_pairs_are_aligned_by_pr_not_by_row_order(self):
        rel = self._m(["alice", "bob"], [1.0, 9.0])
        lat = self._m(["carol", "dave"], [3.0, 5.0]).iloc[::-1]
        out = R.paired_latency(rel, lat, n_boot=50).set_index("quantity")
        assert out.loc["diff_top1_paired_median_diff_h", "value"] == pytest.approx(-1.0)  # (2 + -4) / 2

    def test_bootstrap_median_ci_of_constant(self):
        lo, hi = R.bootstrap_median_ci([3.0] * 5, n_boot=50)
        assert lo == hi == 3.0


class TestEndToEnd:
    def test_r1_scores_and_outputs_on_synthetic_pairs(self):
        d = R.add_r1_scores(_pairs())
        for col in ["s_most_familiar", "s_familiarity_only", R.LAT]:
            assert d[col].notna().all()
        per_pr, micro, _ = R.summary_rows(d, {**R.FAMILIAR_CONFIGS, "constr_lat_e0.1": R.LAT}, n_boot=50)
        assert len(micro) == 5 and (micro.n == 3).all()
        # pr 1: bob is within eps of alice and much faster -> latency config picks bob
        top = per_pr["constr_lat_e0.1"].set_index("pr").top1
        assert top.loc[1] == "bob"
        out = R.paired_latency(per_pr["ranker_pooled_STRICT"], per_pr["constr_lat_e0.1"], n_boot=50).set_index("quantity")
        assert out.loc["diff_top1_paired_median_diff_h", "value"] == pytest.approx(1.0 - 5.0)
        assert not math.isnan(out.loc["common_rel_only_median_h", "value"])
