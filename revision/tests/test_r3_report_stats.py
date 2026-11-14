import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import r3_report_stats as S  # noqa: E402
from r3_report_models import prefix_quantile_labels, _quantile_sorted  # noqa: E402


# ── Wilson ──────────────────────────────────────────────────────────────────

def test_wilson_known_value():
    # 8 of 10, z = 1.96: (0.4902, 0.9433) – standard textbook value
    lo, hi = S.wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4)
    assert hi == pytest.approx(0.9433, abs=1e-4)


def test_wilson_bounds_and_empty():
    assert S.wilson(0, 20)[0] == 0.0
    assert S.wilson(20, 20)[1] == 1.0
    assert all(math.isnan(x) for x in S.wilson(0, 0))


def test_wilson_half_width_n20_matches_plan():
    # threshold 5 text: n = 20 gives about ±0.14–0.19 for precision 0.9–0.7
    lo, hi = S.wilson(18, 20)
    assert 0.12 < (hi - lo) / 2 < 0.16
    lo, hi = S.wilson(14, 20)
    assert 0.17 < (hi - lo) / 2 < 0.21


# ── threshold choice (threshold 5) ──────────────────────────────────────────

def test_threshold_smallest_qualifying():
    # 30 rows: top 25 by p contain 23 positives (0.92), all 30 contain 23 (0.767)
    p = np.linspace(1.0, 0.01, 30)
    y = np.array([1] * 23 + [0] * 7)
    y[[3, 10]] = 0  # two negatives among the top 25
    y[[25, 26]] = 1
    th = S.choose_threshold(p, y, 0.1)
    fire = p > th
    assert fire.sum() >= 20 and y[fire].mean() >= 0.9
    # theta is the next smaller score below the firing set (smallest theta for that set)
    assert th in set(p) and not fire[p == th].any()
    # no smaller theta qualifies (the firing set only changes at the scores)
    for t in np.r_[p[p < th], 0.0]:
        f = p > t
        assert not (f.sum() >= 20 and y[f].mean() >= 0.9)


def test_threshold_needs_20_firings():
    p = np.linspace(1, 0, 19)
    y = np.ones(19)
    assert S.choose_threshold(p, y, 0.3) is None
    p = np.linspace(1, 0.05, 20)
    assert S.choose_threshold(p, np.ones(20), 0.3) == pytest.approx(0.0)
    # a score of exactly 0 never fires under p > theta: 19 firings only
    p = np.linspace(1, 0, 20)
    assert S.choose_threshold(p, np.ones(20), 0.3) is None


def test_threshold_strict_inequality_matches_rule():
    # eq:rule fires p > theta: a row whose score equals theta does not fire
    p = np.r_[np.full(20, 0.9), np.full(5, 0.4), np.full(5, 0.2)]
    y = np.r_[np.ones(20), np.zeros(10)]
    th = S.choose_threshold(p, y, 0.1)    # only the 20 rows at 0.9 qualify
    assert th == pytest.approx(0.4)
    a = S.apply_threshold(p, y, th)
    assert a["n_fire"] == 20 and a["precision"] == 1.0


def test_threshold_unreachable():
    rng = np.random.default_rng(0)
    p = rng.random(200)
    y = np.zeros(200)
    y[:20] = 1   # 10 % positives, unrelated to p
    assert S.choose_threshold(p, y, 0.1) is None


def test_threshold_ties_count_together():
    # 25 rows with the same score: either all fire or none
    p = np.r_[np.full(25, 0.5), np.full(10, 0.1)]
    y = np.r_[np.r_[np.ones(20), np.zeros(5)], np.zeros(10)]
    th = S.choose_threshold(p, y, 0.2)   # 20/25 = 0.8 >= 0.8; theta = next score 0.1
    assert th == pytest.approx(0.1)
    assert S.apply_threshold(p, y, th)["n_fire"] == 25
    assert S.choose_threshold(p, y, 0.1) is None


def test_apply_threshold_counts():
    a = S.apply_threshold([0.9, 0.8, 0.2, 0.1], [1, 0, 1, 0], 0.5)
    assert a["n_fire"] == 2 and a["tp"] == 1
    assert a["precision"] == 0.5 and a["recall"] == 0.5 and a["fire_share"] == 0.5


# ── transfer criterion (threshold 6) ────────────────────────────────────────

def test_transfer_point_precision_not_lower_bound():
    # other projects: perfect ordering, theta reachable
    p_o = np.linspace(1, 0, 100)
    y_o = (p_o > 0.5).astype(int)
    # project: 20 firings with precision exactly 0.8 -> sufficient for c_r = 0.2
    p_p = np.r_[np.full(20, 0.95), np.full(20, 0.1)]
    y_p = np.r_[np.r_[np.ones(16), np.zeros(4)], np.zeros(20)]
    r = S.transfer_check(p_o, y_o, p_p, y_p, 0.2)
    assert r["evaluable"] and r["sufficient"]
    assert r["prec_lo"] < 0.8          # the lower bound would have failed
    r = S.transfer_check(p_o, y_o, p_p, y_p, 0.1)
    assert r["evaluable"] and not r["sufficient"]


def test_transfer_not_evaluable_with_few_firings():
    p_o = np.linspace(1, 0, 100)
    y_o = (p_o > 0.5).astype(int)
    p_p = np.r_[np.full(19, 0.95), np.full(30, 0.1)]
    y_p = np.r_[np.ones(19), np.zeros(30)]
    r = S.transfer_check(p_o, y_o, p_p, y_p, 0.2)
    assert not r["evaluable"] and not r["sufficient"]


def test_transfer_unreachable_on_others():
    r = S.transfer_check(np.full(50, 0.5), np.r_[np.ones(10), np.zeros(40)],
                         np.ones(30), np.ones(30), 0.1)
    assert r["theta"] is None and not r["evaluable"]


# ── other statistics ────────────────────────────────────────────────────────

def test_between_group_sd_zero_and_positive():
    v = np.array([0.7, 0.71, 0.69, 0.7, 0.71, 0.69])
    g = np.array(["a", "a", "a", "b", "b", "b"])
    assert S.between_group_sd(v, g) == 0.0
    v2 = np.array([0.6, 0.61, 0.59, 0.8, 0.81, 0.79])
    assert S.between_group_sd(v2, g) == pytest.approx(0.1414, abs=2e-3)


def test_mde_formula():
    assert S.mde_language(0.010) == pytest.approx(3.3 * 0.010 / math.sqrt(8))


def test_ece_perfect_and_bad():
    y = np.r_[np.zeros(50), np.ones(50)]
    assert S.ece(y, np.full(100, 0.5)) == pytest.approx(0.5)
    assert S.ece(y, y) == pytest.approx(0.0)


def test_verdict_and_diff():
    assert S.verdict_transfer(0.71) == "переноситься"
    assert S.verdict_transfer(0.65) == "переноситься слабко"
    assert S.verdict_transfer(0.59).startswith("НН3")
    assert S.verdict_rule_transfer(0, 1) == "не оцінюється"
    assert S.verdict_rule_transfer(5, 10) == "достатнє"
    assert S.verdict_rule_transfer(4, 10) == "недостатнє"
    assert S.diff_exists(0.03, 0.01, 0.05)
    assert not S.diff_exists(0.015, 0.01, 0.02)
    assert not S.diff_exists(0.05, -0.01, 0.1)


def test_cluster_boot_mean_contains_mean():
    m, lo, hi = S.cluster_boot_mean(np.linspace(0.6, 0.8, 40), B=2000)
    assert lo < m < hi and m == pytest.approx(0.7)


# ── upper-quartile label (sens.tr_q75) ──────────────────────────────────────

def test_quantile_sorted_matches_numpy():
    rng = np.random.default_rng(1)
    for n in (1, 2, 5, 31):
        s = sorted(rng.random(n))
        assert _quantile_sorted(s, 0.75) == pytest.approx(np.quantile(s, 0.75))


def test_prefix_quantile_uses_only_earlier_merges():
    # 40 PRs merged in order; PR i created at i + 0.5, merged at i + 1
    n = 40
    created = [i + 0.5 for i in range(n)]
    merged = [i + 1.0 for i in range(n)]
    tr = [float(i) for i in range(n)]
    lab, nh = prefix_quantile_labels(created, merged, tr, tr, 0.75, n_min=30)
    # PR 35 created at 35.5: merged before it are PRs 0..34 (35 values)
    assert nh[35] == 35
    q = np.quantile(np.arange(35.0), 0.75)
    assert lab[35] == int(35.0 > q)
    assert all(math.isnan(x) for x in lab[:30])
