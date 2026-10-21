"""Accuracy vs concentration vs expected latency: re-ranking and Pareto front."""
import numpy as np


def _rank_norm(s, ascending=True):
    # percentile rank within the candidate pool; NaN -> worst
    return s.rank(pct=True, ascending=ascending, na_option="bottom" if ascending else "top")


def rerank(df, rel_col, lam_w=0.0, lam_t=0.0):
    df = df.copy()
    g = df.groupby(["repo", "pr"], sort=False)
    r = g[rel_col].rank(pct=True, ascending=True)                      # high relevance -> high
    w = g["workload"].rank(pct=True, ascending=True)                   # high workload -> high (penalty)
    lat = df["lat_med_h"].fillna(df["lat_med_h"].max() if df["lat_med_h"].notna().any() else 0)
    t = lat.groupby([df["repo"], df["pr"]]).rank(pct=True, ascending=True)   # slow -> high (penalty)
    df["s_tradeoff"] = r - lam_w * w - lam_t * t
    return df


def pareto_front(points, maximize, minimize):
    vals = points[list(maximize) + list(minimize)].to_numpy(dtype=float).copy()
    vals[:, :len(maximize)] *= -1  # now everything is "minimize"
    keep = []
    for i in range(len(vals)):
        dominated = False
        for j in range(len(vals)):
            if i != j and np.all(vals[j] <= vals[i]) and np.any(vals[j] < vals[i]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return points.iloc[keep]
