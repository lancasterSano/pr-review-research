"""Tests of revision/r4_calc/r4_e9b.py (post-hoc E9 extension): K1b zero dt_r, K2b shuffled
P_tr, monotonicity in k, the re-ranking bound and the k estimators on synthetic data."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "r4_calc"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import r4_e9b as B  # noqa: E402


def _r4_project(n=800, seed=1):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    y_tr = (rng.random(n) < 1 / (1 + np.exp(-2 * z))).astype(float)
    y_tr[rng.random(n) < 0.1] = np.nan
    dtr = np.where(rng.random(n) < 0.4, rng.exponential(5, n), 0.0) * (1 + (z > 0))
    nit = rng.integers(0, 5, n).astype(float)
    return {"row_id": np.arange(n), "p_tr": 1 / (1 + np.exp(-(2 * z + rng.normal(0, .5, n)))),
            "y_tr": y_tr, "dtr": dtr, "p_it": rng.random(n), "y_it": (nit >= 2).astype(float),
            "tm": rng.lognormal(3, 1.5, n), "nit": nit, "stale": (rng.random(n) < .05).astype(float)}


def test_k1b_zero_dtr_gives_zero():
    P = _r4_project()
    P["dtr"] = np.zeros_like(P["dtr"])
    n = len(P["row_id"])
    for r in B.r4_variant(P, np.arange(n // 2), np.arange(n // 2, n), {"k1": 1.0}):
        for w in B.W4:
            assert r[f"G4_{w}"] == 0.0 and r[f"Gall_{w}"] == 0.0 and r[f"R4_{w}"] == 0.0


def test_g_monotone_in_k_and_all_bound():
    P = _r4_project()
    n = len(P["row_id"])
    rows = B.r4_variant(P, np.arange(n // 2), np.arange(n // 2, n), {"a": 0.5, "b": 1.0, "c": 2.0})
    for c in ("c01", "c02", "c03"):
        g = {r["kname"]: r for r in rows if r["c"] == c}
        assert g["a"]["G4_w3"] <= g["b"]["G4_w3"] <= g["c"]["G4_w3"]
        assert g["b"]["G4_w3"] <= g["b"]["Gall_w3"] + 1e-15
        assert g["b"]["Gc_w3"] == pytest.approx(g["b"]["G4_w3"] + g["b"]["Git_w3"])
        assert g["b"]["G4_tm"] >= g["b"]["G4_w3"]


def test_k2b_shuffled_ptr_ratio_near_one():
    c, r = [], []
    for seed in range(60):
        P = _r4_project(seed=seed)
        n = len(P["row_id"])
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        rows = B.r4_variant(P, perm[:n // 2], perm[n // 2:], {"k1": 1.0}, rng.permutation(n))
        x = [q for q in rows if q["c"] == "c03"][0]
        c.append(x["G4ctrl_w3"])
        r.append(x["R4_w3"])
    assert np.mean(c) / np.mean(r) == pytest.approx(1.0, abs=0.15)


def _k_frame(slope_after, seed=0, n_repo=6, n=400):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_repo):
        tr = rng.exponential(10, n)
        after = rng.exponential(20, n) * np.exp(0.3 * g) + slope_after * tr
        rows.append(pd.DataFrame({"repo": f"r{g}", "tr": tr, "tm": tr + after,
                                  "x_size": rng.normal(size=n)}))
    return pd.concat(rows, ignore_index=True)


def test_shift_model_recovered_when_after_time_is_independent():
    K = B.k_estimates(_k_frame(0.0), 3, 200, 20)
    assert K["shift_supported"]
    assert K["k3"] == pytest.approx(1.0, abs=0.15)
    assert K["k2"] == pytest.approx(1.0, abs=0.25)


def test_shift_model_rejected_when_after_time_grows_with_tr():
    K = B.k_estimates(_k_frame(2.0), 3, 200, 20)
    assert not K["shift_supported"] and K["c_shift"] > 0
    assert K["k3"] == pytest.approx(3.0, abs=0.4)
