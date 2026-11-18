import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "r4_calc"))
import r4_e9_baseline as E  # noqa: E402


def _synthetic(n_repos=3, n=120, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    rid = 0
    for r in range(n_repos):
        start = pd.Timestamp("2025-01-01", tz="UTC")
        created = start + pd.to_timedelta(np.sort(rng.uniform(0, 300, n)), unit="D")
        t_m = rng.lognormal(2.5, 1.5, n)
        t_m[:3] = 5.0                                     # ties in t_m
        merged = created + pd.to_timedelta(t_m, unit="h")
        for i in range(n):
            rows.append({"row_id": rid, "repo": f"o/r{r}", "pr_number": i, "created": created[i],
                         "merged": merged[i], "t_m_raw": t_m[i], "t_m": t_m[i],
                         "n_it": float(rng.integers(0, 4)), "stale": int(rng.random() < 0.05)})
            rid += 1
    return pd.DataFrame(rows)


def test_mid_cdf_self_mean_is_half():
    x = np.array([1, 1, 2, 3, 3, 3, 7.5])
    assert E.mid_cdf(x, x).mean() == pytest.approx(0.5, abs=1e-15)
    assert E.mid_cdf([1, 2, 3, 4], [0, 1, 2.5, 9]).tolist() == [0.0, 0.125, 0.5, 1.0]


def test_rolling_matches_naive_and_n_min():
    s = _synthetic(n_repos=1, n=80)
    ht = s.merged.astype("int64").to_numpy()
    qt = s.created.astype("int64").to_numpy()
    F, k = E.rolling_mid_cdf(ht, s.n_it, qt, s.n_it, n_min=10)
    for i in range(len(s)):
        hist = s.n_it[ht < qt[i]].to_numpy()
        assert k[i] == len(hist)
        if len(hist) < 10:
            assert np.isnan(F[i])
        else:
            assert F[i] == pytest.approx(E.mid_cdf(hist, [s.n_it[i]])[0])


def test_earlier_half_self_mean_is_half():
    d = E.compute(_synthetic())
    m = d[d.half == 0].groupby("repo")[["B_ftm", "B_fnit"]].mean()
    assert np.allclose(m.to_numpy(), 0.5, atol=1e-12)


def test_row_order_does_not_change_eff():
    s = _synthetic()
    d1 = E.compute(s)
    d2 = E.compute(s.sample(frac=1.0, random_state=7).reset_index(drop=True))
    for c in ("A_eff", "B_eff", "A_ftm", "A_fnit", "B_ftm", "B_fnit"):
        assert np.allclose(d1[c], d2[c], equal_nan=True), c


def test_monotone_transform_of_tm_keeps_component():
    s = _synthetic()
    d1 = E.compute(s)
    d2 = E.compute(s.assign(t_m=np.sqrt(s.t_m) * 10 + 3))
    for c in ("A_ftm", "B_ftm"):
        assert np.allclose(d1[c], d2[c], equal_nan=True), c


def test_eff_weights_and_top_share():
    assert E.eff([0.6], [0.3], [1])[0] == pytest.approx((0.6 + 0.3 + 1) / 3)
    assert E.eff([0.6], [0.3], [1], w=(1, 0, 0))[0] == pytest.approx(0.6)
    assert E.top_share([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0.1) == pytest.approx(0.1)
    assert E.top_share([0, 0, 0, 0, 0, 0, 0, 0, 0, 5], 0.1) == pytest.approx(1.0)
