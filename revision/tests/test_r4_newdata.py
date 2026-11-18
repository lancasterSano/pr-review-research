"""Tests of revision/r4_calc/r4_newdata.py (constructed next periods): AUC on the full next period equals the
full-sample AUC, row order of the inputs does not change the results, the variant generator is
the sequential E9 / E9b one, the R4 measures and the costs on synthetic data, the zero-tolerance
re-ranking keeps the first candidate."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "r4_calc"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import r4_newdata as N  # noqa: E402
from rank import constrained as C  # noqa: E402


# ── synthetic inputs ────────────────────────────────────────────────────────

def _table(n=600, seed=3):
    rng = np.random.default_rng(seed)
    repo = rng.choice(["a/a", "b/b", "c/c"], n)
    created = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(rng.integers(0, 10**7, n), unit="s")
    z = rng.normal(size=n)
    y_it = (rng.random(n) < 1 / (1 + np.exp(-2 * z))).astype(float)
    y_tr = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    y_tr[rng.random(n) < 0.1] = np.nan
    t = pd.DataFrame({"repo": repo, "created": created, "hait_filter": 1, "in_main": 1,
                      "y_IT__main_D": y_it, "y_TFR__main_D": y_tr, "row_id": np.arange(n)})
    p_it = 1 / (1 + np.exp(-(2 * z + rng.normal(0, 1, n))))
    p_tr = 1 / (1 + np.exp(-(z + rng.normal(0, 1, n))))
    preds = {"it": pd.DataFrame({"row_id": np.arange(n), "prob": p_it}),
             "tr": pd.DataFrame({"row_id": np.arange(n)[~np.isnan(y_tr)], "prob": p_tr[~np.isnan(y_tr)]})}
    return t, preds


def _pairs(n_pr=40, seed=5):
    rng = np.random.default_rng(seed)
    rows, prs = [], []
    for repo in ("x", "y"):
        for pr in range(n_pr):
            cands = [f"c{j}" for j in rng.choice(12, rng.integers(3, 8), replace=False)]
            pos = rng.choice(cands)
            for c in cands:
                rows.append({"repo": repo, "pr": pr, "candidate": c, "label": int(c == pos), "split": "test",
                             "s_pooled_STRICT": float(rng.random()), "lat_med_h": float(rng.exponential(5))})
            prs.append({"repo": repo, "pr": pr, "created": pd.Timestamp("2025-01-01") + pd.Timedelta(hours=int(rng.integers(0, 5000)))})
    return pd.DataFrame(rows), pd.DataFrame(prs)


# ── AUC ─────────────────────────────────────────────────────────────────────

def test_auc_full_next_period_equals_full_sample():
    t, preds = _table()
    PA = N.auc_arrays(t, preds)
    for repo, P in PA.items():
        full = N.auc_on(P, np.arange(len(P["row_id"])))
        for k in ("it", "tr"):
            y, p = P["y_" + k], P["p_" + k]
            ok = np.isfinite(y) & np.isfinite(p)
            assert abs(full[k] - N.auc(y[ok], p[ok])) < 1e-15
            # order of the next-period rows does not matter
            perm = np.random.default_rng(1).permutation(len(y))
            assert abs(N.auc_on(P, perm)[k] - full[k]) < 1e-12


def test_auc_arrays_row_order_invariant():
    t, preds = _table()
    PA = N.auc_arrays(t, preds)
    rng = np.random.default_rng(7)
    t2 = t.iloc[rng.permutation(len(t))].reset_index(drop=True)
    p2 = {k: v.iloc[rng.permutation(len(v))].reset_index(drop=True) for k, v in preds.items()}
    PA2 = N.auc_arrays(t2, p2)
    assert sorted(PA) == sorted(PA2)
    for r in PA:
        for c in PA[r]:
            assert np.array_equal(PA[r][c], PA2[r][c], equal_nan=True)


def test_auc_single_class_is_nan():
    assert np.isnan(N.auc(np.ones(5), np.arange(5.0)))
    assert np.isnan(N.auc(np.array([]), np.array([])))


def test_perms_sequential_like_e9():
    n, seed, i_r = 37, 20261115, 4
    rng = np.random.default_rng([seed, 1, i_r])
    ref = [rng.permutation(n) for _ in range(5)]
    got = N.perms(n, seed, 1, i_r, 5)
    assert all(np.array_equal(a, b) for a, b in zip(ref, got))


def test_vsum():
    s = N.vsum([1.0, 2.0, 3.0, np.nan])
    assert s["med"] == 2.0 and s["mean"] == 2.0 and s["lo"] < 1.1 and s["hi"] > 2.9
    assert N.vsum([np.nan])["med"] is None


# ── ranking ─────────────────────────────────────────────────────────────────

def test_canon_pairs_row_order_invariant():
    d, prs = _pairs()
    a = N.canon_pairs(d, prs)
    rng = np.random.default_rng(2)
    b = N.canon_pairs(d.iloc[rng.permutation(len(d))], prs.iloc[rng.permutation(len(prs))])
    assert a.equals(b)


def test_per_pr_metrics_and_variant_means_row_order_invariant():
    d, prs = _pairs()
    rng = np.random.default_rng(4)
    out = []
    for dd in (d, d.iloc[rng.permutation(len(d))]):
        c = N.canon_pairs(dd, prs)
        c["s_exp"] = C.rerank_constrained(c, "s_pooled_STRICT", eps=0.1, w_lat=0.0, w_exp=1.0)["s_constrained"].to_numpy()
        keys = N.pr_keys(c)
        m = N.per_pr_table(c, keys, "s_exp")
        idx = np.arange(0, len(keys), 2)
        out.append((m[["hit1", "hit3", "mrr"]].to_numpy(), N.exposure_on(c, keys, idx)))
    assert np.array_equal(out[0][0], out[1][0])
    assert out[0][1] == out[1][1]


def test_pr_keys_slices_and_time_order():
    d, prs = _pairs()
    c = N.canon_pairs(d, prs)
    keys = N.pr_keys(c)
    assert len(keys) == c.groupby(["repo", "pr"]).ngroups
    for _, k in keys.iterrows():
        g = c.iloc[k.start:k.stop]
        assert (g.repo == k.repo).all() and (g.pr == k.pr).all()
        assert len(g) == ((c.repo == k.repo) & (c.pr == k.pr)).sum()
    for _, g in keys.groupby("repo"):
        assert g.created.is_monotonic_increasing


def test_zero_tolerance_keeps_first_candidate():
    d, prs = _pairs()
    c = N.canon_pairs(d, prs)
    c["s0"] = C.rerank_constrained(c, "s_pooled_STRICT", eps=0.0, w_lat=1.0, w_exp=1.0)["s_constrained"].to_numpy()
    keys = N.pr_keys(c)
    a = N.per_pr_table(c, keys, "s_pooled_STRICT")
    b = N.per_pr_table(c, keys, "s0")
    assert (a.top1 == b.top1).all() and np.array_equal(a.hit1, b.hit1)


def test_exposure_on_all_prs_equals_full_stream():
    d, prs = _pairs()
    c = N.canon_pairs(d, prs)
    c["s_exp"] = C.rerank_constrained(c, "s_pooled_STRICT", eps=0.1, w_lat=0.0, w_exp=1.0)["s_constrained"].to_numpy()
    keys = N.pr_keys(c)
    full = N.per_pr_table(c, keys, "s_exp")
    got = N.exposure_on(c, keys, np.arange(len(keys)))
    for k in ("hit1", "hit3", "mrr"):
        assert abs(got[k] - full[k].mean()) < 1e-15


def test_paired_gain_zero_for_identical():
    x = pd.Series(np.random.default_rng(0).random(50))
    g = N.paired_gain(x, x)
    assert g == (0.0, 0.0, 0.0)


# ── R4 ──────────────────────────────────────────────────────────────────────

def test_r4_measures_known_values():
    rel = {"hit1": np.array([1, 0, 1, 0.]), "hit3": np.array([1, 1, 1, 0.]), "mrr": np.array([1, .5, 1, .2]),
           "top1_lat": np.array([10, 20, 30, 40.])}
    lat = {"hit1": np.array([0, 0, 1, 0.]), "hit3": np.array([1, 0, 1, 0.]), "mrr": np.array([.5, .5, 1, .2]),
           "top1_lat": np.array([5, 10, 15, 20.])}
    fire = [np.array([True, True]), np.array([False, False])]
    m = N.r4_measures(rel, lat, fire, [2, 2])
    assert m["n_fire"] == 2 and m["fire_pct"] == 50.0 and m["fire_pct_repomean"] == 50.0
    assert m["lat_before_h"] == 15.0 and m["lat_after_h"] == 7.5 and m["lat_ratio"] == 0.5
    assert m["hit3_before"] == 1.0 and m["hit3_after"] == 0.5 and m["hit3_drop"] == 0.5
    assert m["cost_fired_hit1"] == 0.5 and m["cost_fired_mrr"] == 0.25
    assert m["cost_all_hit1"] == 0.25 and m["cost_all_hit3"] == 0.25


def test_r4_measures_no_firing_and_identity():
    rel = {"hit1": np.ones(4), "hit3": np.ones(4), "mrr": np.ones(4), "top1_lat": np.arange(4.0)}
    m = N.r4_measures(rel, rel, [np.zeros(4, bool)], [4])
    assert m["n_fire"] == 0 and np.isnan(m["lat_ratio"]) and np.isnan(m["cost_fired_hit1"])
    assert m["cost_all_mrr"] == 0.0
    m = N.r4_measures(rel, rel, [np.ones(4, bool)], [4])
    assert all(m[f"cost_{w}_{k}"] == 0.0 for w in ("fired", "all") for k in ("hit1", "hit3", "mrr"))
