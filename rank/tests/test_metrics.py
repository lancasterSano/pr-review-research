import math
import pytest

from rank import metrics


class TestRankMetrics:
    def test_mrr_uses_first_positive(self):
        assert metrics.mrr(["a", "b", "c"], {"b", "c"}) == pytest.approx(0.5)

    def test_mrr_zero_when_no_positive(self):
        assert metrics.mrr(["a", "b"], {"z"}) == 0.0

    def test_hit_at_k(self):
        assert metrics.hit_at_k(["a", "b", "c"], {"c"}, 3) == 1.0
        assert metrics.hit_at_k(["a", "b", "c"], {"c"}, 2) == 0.0

    def test_ndcg_at_k_binary_gains(self):
        # positives at ranks 1 and 3 of 5, k=5: DCG = 1/log2(2) + 1/log2(4) = 1.5; IDCG (2 positives) = 1 + 1/log2(3)
        got = metrics.ndcg_at_k(["p", "n", "p2", "n2", "n3"], {"p", "p2"}, 5)
        assert got == pytest.approx(1.5 / (1 + 1 / math.log2(3)))

    def test_ndcg_perfect_is_one(self):
        assert metrics.ndcg_at_k(["p", "p2", "n"], {"p", "p2"}, 5) == pytest.approx(1.0)


class TestGini:
    def test_uniform_is_zero(self):
        assert metrics.gini([3, 3, 3, 3]) == pytest.approx(0.0)

    def test_all_on_one_is_near_one(self):
        assert metrics.gini([0, 0, 0, 10]) == pytest.approx(0.75)  # (n-1)/n for n=4

    def test_empty_is_nan(self):
        assert math.isnan(metrics.gini([]))


class TestBootstrap:
    def test_ci_contains_mean_and_is_deterministic(self):
        vals = [0, 1, 1, 1, 0, 1, 1, 0, 1, 1]
        lo, hi = metrics.bootstrap_ci(vals, n_boot=500, seed=1)
        lo2, hi2 = metrics.bootstrap_ci(vals, n_boot=500, seed=1)
        assert lo <= 0.7 <= hi and (lo, hi) == (lo2, hi2)


class TestGiniWithUniverse:
    def test_zero_counts_for_never_recommended_candidates_are_included(self):
        counts = {"a": 5, "b": 5}
        assert metrics.gini_over_universe(counts, ["a", "b"]) == pytest.approx(0.0)
        # same recommendations, but two more candidates existed and were never picked
        assert metrics.gini_over_universe(counts, ["a", "b", "c", "d"]) == pytest.approx(0.5)
