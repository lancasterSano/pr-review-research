"""Tests of revision/r4_calc/r4_e9_eff.py (E9): the controls with a known answer
(K1 null action, K2 shuffled forecasts, K3 row order, K4 oracle) and the
equality of the vectorised threshold with r3_report_stats.choose_threshold."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "r4_calc"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import r3_report_stats as S  # noqa: E402
import r4_e9_eff as E  # noqa: E402


def _project(n=600, seed=1, signal=2.0):
    """Synthetic project: forecasts informative about n_it >= 2 and stale."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    nit = np.where(rng.random(n) < 1 / (1 + np.exp(-(signal * z - 1.2))),
                   rng.integers(2, 6, n), rng.integers(0, 2, n)).astype(float)
    stale = (rng.random(n) < 0.05 + 0.1 * (z > 1.5)).astype(float)
    p_it = 1 / (1 + np.exp(-(signal * z - 1.2 + rng.normal(0, 0.5, n))))
    p_st = np.clip(0.05 + 0.1 * (z > 1.5) + rng.normal(0, 0.02, n), 0, 1)
    created = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(
        np.sort(rng.uniform(0, 300, n)), unit="D")
    return pd.DataFrame({"row_id": np.arange(n) * 3 + 7, "created": created,
                         "p_it": p_it, "p_st": p_st, "nit": nit, "stale": stale,
                         "tm": rng.lognormal(2, 1.5, n)})


def _arrays(df):
    df = df.sort_values(["created", "row_id"], kind="mergesort")
    P = {c: df[c].to_numpy() for c in ("row_id", "p_it", "p_st", "nit", "stale", "tm")}
    P["y_it"] = (P["nit"] >= 2).astype(float)
    P["y_st"] = P["stale"].astype(float)
    P["week"] = ((df.created - pd.Timestamp("2024-01-01", tz="UTC")).dt.days // 7).to_numpy()
    return P


@pytest.mark.parametrize("seed", range(30))
def test_threshold_equals_reference(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 400))
    p = np.round(rng.random(n), int(rng.integers(1, 4)))        # ties
    p[rng.random(n) < 0.05] = 0.0                                 # zeros never fire
    y = (rng.random(n) < p * rng.uniform(0.5, 1.5)).astype(float)
    for c in (0.1, 0.2, 0.3, 0.5):
        for mf in (1, 5, 20):
            assert E.choose_threshold(p, y, c, mf) == S.choose_threshold(p, y, c, mf)


def test_k1_null_action_gives_zero():
    P = _arrays(_project())
    n = len(P["row_id"])
    comp = E.components(P, np.arange(n // 2), np.arange(n // 2, n))
    z = np.zeros(n - n // 2, bool)
    m = E.measures(comp, z, z)
    for rule in E.RULES:
        for w in E.WTS:
            assert m[rule][w] == (0.0, 0.0)


def test_unreachable_threshold_gives_zero():
    P = _arrays(_project())
    P["p_it"] = np.full_like(P["p_it"], 0.5)      # one score: precision = base rate < 0.7
    P["p_st"] = np.full_like(P["p_st"], 0.5)
    n = len(P["row_id"])
    rows = E.run_variant(P, np.arange(n // 2), np.arange(n // 2, n))
    for r in rows:
        assert not r["reach"] and r["k"] == 0
        assert all(r[f"A_{w}"] == 0.0 for w in E.WTS)


def test_k4_oracle():
    P = _arrays(_project())
    P["p_it"] = P["y_it"].copy()
    n = len(P["row_id"])
    hist, nxt = np.arange(n // 2), np.arange(n // 2, n)
    rows = [r for r in E.run_variant(P, hist, nxt) if r["rule"] == "it"]
    comp = E.components(P, hist, nxt)
    k_true = int(P["y_it"][nxt].sum())
    for r in rows:
        assert r["k"] == k_true
        b = 1 / 3
        L = (comp["ftm"].sum() + comp["fnit"].sum() + comp["stale"].sum()) / 3
        assert r["A_w3"] == pytest.approx(b * comp["d_nit"].sum() / L, rel=1e-12)
        assert r["A_w3"] / r["R_w3"] == pytest.approx(len(nxt) / k_true, rel=1e-12)
        assert r["A_tm"] == 0.0


def test_a_rand_is_mean_over_random_choices():
    P = _arrays(_project(n=300))
    n = len(P["row_id"])
    hist, nxt = np.arange(n // 2), np.arange(n // 2, n)
    comp = E.components(P, hist, nxt)
    k = 17
    rng = np.random.default_rng(5)
    draws = []
    for _ in range(20000):
        f = np.zeros(len(nxt), bool)
        f[rng.choice(len(nxt), k, replace=False)] = True
        draws.append(E.measures(comp, f, np.zeros(len(nxt), bool))["it"]["w3"][0])
    f = np.zeros(len(nxt), bool)
    f[:k] = True
    a_rand = E.measures(comp, f, np.zeros(len(nxt), bool))["it"]["w3"][1]
    assert np.mean(draws) == pytest.approx(a_rand, rel=0.01)


def test_any_is_sum_of_rules():
    P = _arrays(_project())
    n = len(P["row_id"])
    rows = E.run_variant(P, np.arange(n // 2), np.arange(n // 2, n))
    for c in E.CR:
        r = {x["rule"]: x for x in rows if x["c"] == c}
        for w in E.WTS:
            assert r["any"][f"A_{w}"] == pytest.approx(r["it"][f"A_{w}"] + r["st"][f"A_{w}"])
            assert r["any"][f"R_{w}"] == pytest.approx(r["it"][f"R_{w}"] + r["st"][f"R_{w}"])


def test_k2_shuffled_forecasts_lift_near_one():
    lifts = []
    for seed in range(40):
        P = _arrays(_project(n=1000, seed=seed))
        n = len(P["row_id"])
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        rows = E.run_variant(P, perm[:n // 2], perm[n // 2:], perm_ctrl=rng.permutation(n))
        r = [x for x in rows if x["rule"] == "it" and x["c"] == "c03"][0]
        if r["R_w3"] > 0:
            lifts.append((r["C_w3"], r["R_w3"], r["A_w3"]))
    c, rr, a = map(np.array, zip(*lifts))
    assert c.mean() / rr.mean() == pytest.approx(1.0, abs=0.15)
    assert a.mean() / rr.mean() > 1.5                       # the real forecast carries signal


def test_k3_row_order_does_not_change_results():
    df = _project()
    P1 = _arrays(df)
    P2 = _arrays(df.sample(frac=1.0, random_state=3))
    for c in P1:
        assert np.array_equal(P1[c], P2[c])
    n = len(P1["row_id"])
    e, l = np.arange(n // 2), np.arange(n // 2, n)
    a = pd.DataFrame(E.run_project("o/r", 0, P1, (e, l), 7, 4, 3))
    b = pd.DataFrame(E.run_project("o/r", 0, P2, (e, l), 7, 4, 3))
    assert a.equals(b)


def test_eff_sum_invariant_to_order_within_sets():
    P = _arrays(_project())
    n = len(P["row_id"])
    hist, nxt = np.arange(n // 2), np.arange(n // 2, n)
    rng = np.random.default_rng(2)
    c1 = E.components(P, hist, nxt)
    c2 = E.components(P, rng.permutation(hist), rng.permutation(nxt))
    for k in ("ftm", "fnit", "stale", "d_nit"):
        assert c1[k].sum() == pytest.approx(c2[k].sum(), rel=1e-12)


def test_week_blocks_cover_rows_once():
    P = _arrays(_project())
    idx = np.arange(len(P["row_id"]))
    blocks = E.week_blocks(P["week"], idx)
    assert np.array_equal(np.sort(np.concatenate(blocks)), idx)
    assert all(len(np.unique(P["week"][b])) == 1 for b in blocks)
