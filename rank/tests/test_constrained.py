"""E4 v2: constrained re-ranking with bounded relevance loss and simulated exposure."""
import pandas as pd
import pytest

from rank import constrained as C


def _stream():
    # 3 PRs in time order; candidates a,b,c; a always most relevant, b close, c far
    rows = []
    for pr, lat in [(1, {"a": 50, "b": 5, "c": 1}), (2, {"a": 50, "b": 5, "c": 1}), (3, {"a": 50, "b": 5, "c": 1})]:
        for cand, rel in [("a", 1.0), ("b", 0.95), ("c", 0.2)]:
            rows.append({"repo": "r", "pr": pr, "candidate": cand, "s_rel": rel, "lat_med_h": lat[cand],
                         "workload": 0, "label": int(cand == "a"), "created": pd.Timestamp(2026, 1, pr, tz="UTC")})
    return pd.DataFrame(rows)


class TestConstrainedRerank:
    def test_eps_zero_keeps_relevance_order(self):
        out = C.rerank_constrained(_stream(), "s_rel", eps=0.0, w_lat=1.0, w_exp=0.0)
        assert out.groupby("pr").apply(lambda g: g.sort_values("s_constrained", ascending=False).candidate.iloc[0]).tolist() == ["a", "a", "a"]

    def test_within_tolerance_the_faster_candidate_wins(self):
        # eps 0.1 on rank-normalised relevance: a and b are near-top; b is faster -> b first; c excluded
        out = C.rerank_constrained(_stream(), "s_rel", eps=0.1, w_lat=1.0, w_exp=0.0)
        top = out.groupby("pr").apply(lambda g: g.sort_values("s_constrained", ascending=False).candidate.iloc[0]).tolist()
        assert top == ["b", "b", "b"]

    def test_exposure_counter_spreads_recommendations_over_time(self):
        # with exposure weight, after b is recommended once, a (never recommended) should win next time
        out = C.rerank_constrained(_stream(), "s_rel", eps=0.1, w_lat=0.0, w_exp=1.0)
        top = out.groupby("pr").apply(lambda g: g.sort_values("s_constrained", ascending=False).candidate.iloc[0]).tolist()
        assert top[0] != top[1]

    def test_candidates_outside_tolerance_never_reach_top(self):
        out = C.rerank_constrained(_stream(), "s_rel", eps=0.1, w_lat=1.0, w_exp=1.0)
        top = set(out.groupby("pr").apply(lambda g: g.sort_values("s_constrained", ascending=False).candidate.iloc[0]))
        assert "c" not in top


class TestInspiredBaselines:
    def test_moc2r_like_and_rebai_like_scores_exist(self):
        df = _stream().assign(act_total=[3, 2, 1] * 3, fast_24h=[0.1, 0.9, 0.5] * 3, act_30d=[1, 1, 1] * 3, familiarity=[0, 2, 0] * 3)
        out = C.add_inspired_baselines(df)
        assert {"s_moc2r_like", "s_rebai_like"} <= set(out.columns)
        pr1 = out[out.pr == 1].set_index("candidate")
        assert pr1.s_moc2r_like.idxmax() == "b"   # experience mid, response prob high, activity tie


class TestExposureSmoothing:
    def test_one_of_one_is_shrunk_below_ten_of_ten(self):
        assert C.exposure(1, 1) < C.exposure(10, 10)
        assert C.exposure(0, 0) == pytest.approx(0.5)
        assert C.exposure(1, 1) == pytest.approx(2 / 3)


class TestAblationNames:
    def test_ablations_switch_off_exactly_one_component(self):
        from rank.run_e4 import ablations_of
        assert ablations_of("constr_e0.1_L1.0_E1.0") == {"no_tolerance": "constr_e1.0_L1.0_E1.0",
                                                          "no_exposure": "constr_e0.1_L1.0_E0.0",
                                                          "no_latency": "constr_e0.1_L0.0_E1.0"}
        # a config that already has a single secondary criterion has no meaningful single-component ablation
        assert ablations_of("constr_e0.2_L1.0_E0.0") == {"no_tolerance": "constr_e1.0_L1.0_E0.0"}
        assert ablations_of("rel_only") == {}
