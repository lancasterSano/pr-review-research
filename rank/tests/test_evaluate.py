import numpy as np
import pandas as pd
import pytest

from rank import evaluate as ev


def _pairs():
    # two PRs; candidates a,b,c with scores; positives: pr1 -> b ; pr2 -> a,c
    return pd.DataFrame({
        "repo": ["r"] * 6, "pr": [1, 1, 1, 2, 2, 2],
        "candidate": ["a", "b", "c", "a", "b", "c"],
        "label": [0, 1, 0, 1, 0, 1],
        "score": [0.9, 0.5, 0.1, 0.2, 0.8, 0.7],
        "lat_med_h": [10, 20, 30, 10, 20, 30],
    })


class TestRankAndScore:
    def test_per_pr_metrics_from_scores(self):
        m = ev.per_pr_metrics(_pairs(), "score")
        # pr1: ranking a,b,c -> first positive b at rank 2 -> mrr .5, hit1 0, hit3 1
        # pr2: ranking b,c,a -> first positive c at rank 2 -> mrr .5, hit1 0
        assert m.set_index("pr").mrr.to_dict() == {1: 0.5, 2: 0.5}
        assert m.set_index("pr").hit1.to_dict() == {1: 0.0, 2: 0.0}
        assert m.set_index("pr").hit3.to_dict() == {1: 1.0, 2: 1.0}
        assert m.set_index("pr").top1.to_dict() == {1: "a", 2: "b"}
        assert m.set_index("pr").top1_lat.to_dict() == {1: 10, 2: 20}

    def test_summary_has_ci_gini_and_latency(self):
        m = ev.per_pr_metrics(_pairs(), "score")
        s = ev.summarize(m)
        assert s["n"] == 2 and s["mrr"] == pytest.approx(0.5)
        assert "mrr_lo" in s and "mrr_hi" in s and s["mrr_lo"] <= 0.5 <= s["mrr_hi"]
        assert s["gini_top1"] == pytest.approx(0.0)   # a once, b once -> uniform over recommended
        assert s["top1_lat_med"] == pytest.approx(15.0)


class TestBaselines:
    def test_baseline_scores_are_columns(self):
        df = _pairs().assign(act_total=[5, 3, 1, 5, 3, 1], days_since_last=[1, 2, 3, 3, 2, 1], workload=[4, 0, 0, 4, 0, 0])
        out = ev.add_baseline_scores(df, seed=0)
        for col in ["s_random", "s_most_active", "s_most_recent", "s_least_loaded_active", "s_fastest"]:
            assert col in out.columns
        # most_recent: smaller days_since_last -> larger score
        pr2 = out[out.pr == 2].set_index("candidate")
        assert pr2.s_most_recent.idxmax() == "c"
        assert pr2.s_most_active.idxmax() == "a"
        assert pr2.s_fastest.idxmax() == "a"


class TestRealizedLatency:
    def test_top1_realized_latency_only_when_top1_actually_reviewed(self):
        df = _pairs().assign(realized_lat_h=[np.nan, 7.0, np.nan, 3.0, np.nan, 9.0])
        m = ev.per_pr_metrics(df, "score")
        r = m.set_index("pr")
        # pr1 top1 = a (not a positive) -> NaN ; pr2 top1 = b (not positive) -> NaN
        assert np.isnan(r.loc[1, "top1_realized_lat"]) and np.isnan(r.loc[2, "top1_realized_lat"])
        df2 = df.copy(); df2.loc[df2.pr == 1, "score"] = [0.1, 0.9, 0.5]   # now b is top1 for pr1
        r2 = ev.per_pr_metrics(df2, "score").set_index("pr")
        assert r2.loc[1, "top1_realized_lat"] == 7.0
        s = ev.summarize(ev.per_pr_metrics(df2, "score"))
        assert s["realized_lat_med_on_hits"] == 7.0 and s["hit1"] == pytest.approx(0.5)
