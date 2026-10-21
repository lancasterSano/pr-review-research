import numpy as np
import pandas as pd

from rank import ranker, tradeoff


def _synthetic(n_pr=200, n_cand=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pr in range(n_pr):
        x = rng.normal(size=n_cand)
        noise = rng.normal(size=n_cand)
        pos = int(np.argmax(x + 0.3 * noise))
        for c in range(n_cand):
            rows.append({"repo": "r", "pr": pr, "candidate": f"c{c}", "label": int(c == pos),
                         "f_signal": x[c], "f_noise": noise[c],
                         "workload": rng.integers(0, 5), "lat_med_h": rng.uniform(1, 100),
                         "split": "train" if pr < 120 else ("val" if pr < 160 else "test")})
    return pd.DataFrame(rows)


class TestLambdarank:
    def test_learns_the_signal_feature(self):
        df = _synthetic()
        model = ranker.train(df[df.split == "train"], df[df.split == "val"], ["f_signal", "f_noise"])
        test = df[df.split == "test"].copy()
        test["s_model"] = ranker.predict(model, test, ["f_signal", "f_noise"])
        from rank.evaluate import per_pr_metrics
        m = per_pr_metrics(test, "s_model")
        assert m.hit1.mean() > 0.6   # chance would be 1/8

    def test_importance_is_grouped_by_feature(self):
        df = _synthetic()
        model = ranker.train(df[df.split == "train"], df[df.split == "val"], ["f_signal", "f_noise"])
        imp = ranker.importance(model, ["f_signal", "f_noise"])
        assert imp["f_signal"] > imp["f_noise"]


class TestTradeoff:
    def test_lambda_zero_keeps_relevance_order(self):
        df = _synthetic(n_pr=5)
        df["s_rel"] = df["f_signal"]
        out = tradeoff.rerank(df, "s_rel", lam_w=0.0, lam_t=0.0)
        for pr, g in out.groupby("pr"):
            assert list(g.sort_values("s_tradeoff", ascending=False).candidate) == list(g.sort_values("s_rel", ascending=False).candidate)

    def test_large_workload_weight_puts_least_loaded_first(self):
        df = pd.DataFrame({"repo": ["r"] * 3, "pr": [1, 1, 1], "candidate": ["a", "b", "c"],
                           "s_rel": [0.9, 0.5, 0.1], "workload": [9, 0, 5], "lat_med_h": [5, 5, 5], "label": [1, 0, 0]})
        out = tradeoff.rerank(df, "s_rel", lam_w=10.0, lam_t=0.0)
        assert out.sort_values("s_tradeoff", ascending=False).candidate.iloc[0] == "b"

    def test_pareto_front_keeps_non_dominated_points(self):
        pts = pd.DataFrame({"hit3": [0.5, 0.6, 0.4, 0.6], "gini_top1": [0.9, 0.8, 0.7, 0.7], "top1_lat_med": [10, 10, 5, 12]})
        front = tradeoff.pareto_front(pts, maximize=["hit3"], minimize=["gini_top1", "top1_lat_med"])
        assert set(front.index) == {1, 2, 3}   # point 0 is dominated by point 1
