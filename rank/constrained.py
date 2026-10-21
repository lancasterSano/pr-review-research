"""E4 v2: sequential re-ranking with bounded relevance loss.

Stage 1 keeps the candidates whose min-max normalised relevance score (within
the PR's candidate pool) is within `eps` of the best candidate. Stage 2 orders that near-top set by the
expected response latency (history) and by simulated exposure: how often the
candidate has already been recommended (top-1) relative to how often it was
in the candidate pool, accumulated over the evaluation stream in time order
while the history itself stays unchanged (a sequential replay of recommendation
exposure, not a simulation of adoption). Everything outside the near-top set
keeps its relevance order below the near-top set.
"""
import numpy as np
import pandas as pd


def _pct_rank(g, col, ascending=True):
    return g[col].rank(pct=True, ascending=ascending, method="average")


def _minmax(s):
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else pd.Series(1.0, index=s.index)


def exposure(rec, elig, a=1.0, b=1.0):
    """Smoothed share of past pools in which the candidate was the top-1 recommendation.

    Laplace smoothing with prior mean a/(a+b): 0 of 0 -> 0.5, 1 of 1 -> 2/3, 10 of 10 -> 11/12.
    """
    return (rec + a) / (elig + a + b)


def rerank_constrained(df, rel_col, eps=0.1, w_lat=1.0, w_exp=1.0):
    df = df.copy()
    order_col = "created" if "created" in df.columns else "pr"
    df["s_constrained"] = np.nan
    rec, elig = {}, {}
    for (repo, pr), idx in df.sort_values([order_col, "pr"], kind="stable").groupby(["repo", "pr"], sort=False).groups.items():
        g = df.loc[idx]
        r = _minmax(g[rel_col])                        # best candidate -> 1.0, worst -> 0.0
        near = r >= 1.0 - eps
        lat = g["lat_med_h"].fillna(g["lat_med_h"].max() if g["lat_med_h"].notna().any() else 0)
        lat_pen = lat.rank(pct=True, ascending=True)   # slow -> high
        expo = np.array([exposure(rec.get((repo, c), 0), elig.get((repo, c), 0)) for c in g["candidate"]])
        exp_pen = pd.Series(expo, index=g.index).rank(pct=True, ascending=True)
        second = -(w_lat * lat_pen + w_exp * exp_pen)
        # near-top set scores in [1, 2); the rest keep relevance order in [0, 1)
        score = np.where(near, 1.0 + (second - second.min()) / (second.max() - second.min() + 1e-9), r * 0.999)
        df.loc[idx, "s_constrained"] = score
        top = g["candidate"].to_numpy()[int(np.argmax(score))]
        for c in g["candidate"]:
            elig[(repo, c)] = elig.get((repo, c), 0) + 1
        rec[(repo, top)] = rec.get((repo, top), 0) + 1
    return df


def add_inspired_baselines(df):
    """Weighted sums of rank-normalised objectives inspired by prior work.

    MOC2R (Hu et al. 2021): experience, response probability within a time
    window, recent activity. Rebai et al. 2020: expertise, availability,
    collaboration.
    """
    df = df.copy()
    g = df.groupby(["repo", "pr"], sort=False)
    exp_ = g["act_total"].rank(pct=True)
    resp = g["fast_24h"].rank(pct=True, na_option="bottom")
    recent = g["act_30d"].rank(pct=True)
    df["s_moc2r_like"] = exp_ + resp + recent
    avail = g["workload"].rank(pct=True, ascending=False)
    collab = g["familiarity"].rank(pct=True)
    df["s_rebai_like"] = exp_ + avail + collab
    return df
