"""Tests of revision/r4_calc/r4_e9c.py (post-hoc E9 extension B1/B3):
top-k selection with ties and missing values, K1c null action, K4c forecast as a heuristic,
B3 bounds, random heuristic ~ random requests, descriptive ceiling and precision."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "r4_calc"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import r4_e9c as C  # noqa: E402


def _project(n=1000, seed=1, nit_values=None):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    if nit_values is None:
        nit = 1 + rng.poisson(np.exp(1.5 * z - 1.0))
    else:
        nit = rng.choice(nit_values, n)
    nit = nit.astype(float)
    p = 1 / (1 + np.exp(-(2.5 * z + rng.normal(0, 0.5, n))))
    return {"row_id": np.arange(n), "p_it": p, "y_it": (nit >= 2).astype(float),
            "p_st": rng.random(n), "y_st": np.zeros(n), "tm": rng.lognormal(3, 1.5, n) * nit,
            "nit": nit, "stale": (rng.random(n) < 0.05).astype(float)}


def test_top_k_order_ties_and_missing():
    s = np.array([3.0, np.nan, 5.0, 3.0, 1.0])
    u = np.array([0.9, 0.0, 0.5, 0.1, 0.2])
    assert C.top_k(s, u, 1).tolist() == [False, False, True, False, False]
    # tie between rows 0 and 3 broken by u: row 3 (u = 0.1) first
    assert C.top_k(s, u, 2).tolist() == [False, False, True, True, False]
    # NaN is the lowest priority: chosen only after every known value
    assert C.top_k(s, u, 4).tolist() == [True, False, True, True, True]
    assert C.top_k(s, u, 0).sum() == 0


def test_k1c_null_action_gives_zero():
    P = _project()
    P["p_it"] = np.zeros_like(P["p_it"])          # threshold unreachable -> k = 0
    n = len(P["row_id"])
    H = {"size": np.arange(n, dtype=float)}
    r = C.variant_c(P, H, np.arange(n // 2), np.arange(n // 2, n), np.random.default_rng(0).random(n - n // 2))
    assert r["k"] == 0 and not r["reach"]
    for key in ("G_model", "R", "G_size", "G_b3", "R_b3"):
        assert r[key] == 0.0


def test_k4c_forecast_as_heuristic_equals_model():
    for seed in range(5):
        P = _project(seed=seed)
        n = len(P["row_id"])
        perm = np.random.default_rng(seed).permutation(n)
        u = np.random.default_rng(seed + 100).random(n - n // 2)
        r = C.variant_c(P, {"pit": P["p_it"]}, perm[:n // 2], perm[n // 2:], u)
        assert r["k"] > 0
        assert r["G_pit"] == r["G_model"]
        assert r["G_model"] > r["R"]


def test_b3_without_single_iteration_history_equals_model():
    P = _project(nit_values=np.array([0, 2, 3]))
    n = len(P["row_id"])
    perm = np.random.default_rng(3).permutation(n)
    r = C.variant_c(P, {}, perm[:n // 2], perm[n // 2:], np.zeros(n - n // 2))
    assert np.isinf(r["Q_r"])
    assert r["G_b3"] == pytest.approx(r["G_model"], abs=1e-15)
    assert r["R_b3"] == pytest.approx(r["R"], abs=1e-15)


def test_b3_not_below_model_and_bounded():
    P = _project()
    n = len(P["row_id"])
    perm = np.random.default_rng(4).permutation(n)
    r = C.variant_c(P, {}, perm[:n // 2], perm[n // 2:], np.zeros(n - n // 2))
    assert np.isfinite(r["Q_r"])
    assert r["G_b3"] >= r["G_model"]
    assert r["R_b3"] >= r["R"]
    assert r["G_b3"] <= 1.0


def test_random_heuristic_close_to_random_requests():
    g, rr = [], []
    for seed in range(60):
        P = _project(seed=seed)
        n = len(P["row_id"])
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        H = {"size": rng.random(n)}               # scores unrelated to the outcome
        r = C.variant_c(P, H, perm[:n // 2], perm[n // 2:], rng.random(n - n // 2))
        g.append(r["G_size"])
        rr.append(r["R"])
    assert np.mean(g) / np.mean(rr) == pytest.approx(1.0, abs=0.15)


def test_descr_ceiling_and_precision():
    P = _project()
    n = len(P["row_id"])
    perm = np.random.default_rng(5).permutation(n)
    rows = C.variant_descr(P, perm[:n // 2], perm[n // 2:])
    assert [r["c"] for r in rows] == ["c01", "c02", "c03"]
    for r in rows:
        assert 0 <= r["A"] <= r["C"] + 1e-15
        assert r["tp"] <= r["k"] and r["tp"] <= r["n_pos"]
        if r["n_pos"]:
            # the oracle acting on every n_it >= 2 request: A = C and A / R = n / n_pos
            assert r["C"] == pytest.approx(r["R"] * r["n"] / r["k"]) if r["k"] else True
    ks = [r["k"] for r in rows]
    assert ks == sorted(ks)                       # larger c -> lower threshold -> more firings
