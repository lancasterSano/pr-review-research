"""Score columns -> per-PR ranking metrics -> summaries with bootstrap CIs."""
import math

import numpy as np
import pandas as pd

from rank import metrics as M

BASELINES = {
    "s_random": "Random",
    "s_most_active": "Most active (90 d)",
    "s_most_recent": "Most recent",
    "s_least_loaded_active": "Most active, workload re-ranked",
    "s_fastest": "Fastest expected response",
    "s_most_familiar": "Most familiar",
}


def per_pr_metrics(df, score_col, ks=(1, 3, 5)):
    rows = []
    for (repo, pr), g in df.groupby(["repo", "pr"], sort=False):
        order = np.argsort(-g[score_col].to_numpy(), kind="stable")
        ranked = g["candidate"].to_numpy()[order].tolist()
        pos = set(g.loc[g.label == 1, "candidate"])
        row = {"repo": repo, "pr": pr, "n_cand": len(ranked), "n_pos": len(pos),
               "mrr": M.mrr(ranked, pos), "ndcg5": M.ndcg_at_k(ranked, pos, 5),
               "top1": ranked[0]}
        for k in ks:
            row[f"hit{k}"] = M.hit_at_k(ranked, pos, k)
        if "lat_med_h" in g.columns:
            row["top1_lat"] = float(g["lat_med_h"].to_numpy()[order][0])
        if "workload" in g.columns:
            row["top1_workload"] = float(g["workload"].to_numpy()[order][0])
        if "realized_lat_h" in g.columns:
            # observed first-response time of the top-1 candidate on THIS PR, only if it actually reviewed
            top_label = int(g["label"].to_numpy()[order][0])
            row["top1_realized_lat"] = float(g["realized_lat_h"].to_numpy()[order][0]) if top_label == 1 else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(m, n_boot=1000, seed=42, universe=None):
    """`universe`: dict repo -> iterable of candidates seen in the split; when given,
    gini_top1_full counts zeros for candidates never recommended."""
    out = {"n": int(len(m))}
    for col in ["mrr", "hit1", "hit3", "hit5", "ndcg5"]:
        if col in m.columns:
            v = m[col].to_numpy(dtype=float)
            out[col] = float(v.mean())
            lo, hi = M.bootstrap_ci(v, n_boot=n_boot, seed=seed)
            out[f"{col}_lo"], out[f"{col}_hi"] = lo, hi
    counts = m["top1"].value_counts().to_numpy()
    out["gini_top1"] = M.gini(counts)
    if universe is not None:
        per = m.groupby(["repo", "top1"]).size().to_dict()
        vals = []
        for repo in m["repo"].unique():
            vals += [per.get((repo, c), 0) for c in universe.get(repo, [])]
        out["gini_top1_full"] = M.gini(vals)
    out["top1_unique"] = int(len(counts))
    out["top1_share_max"] = float(counts.max() / counts.sum())
    if "top1_lat" in m.columns:
        out["top1_lat_med"] = float(np.nanmedian(m["top1_lat"].to_numpy(dtype=float)))
    if "top1_workload" in m.columns:
        out["top1_workload_mean"] = float(np.nanmean(m["top1_workload"].to_numpy(dtype=float)))
    if "top1_realized_lat" in m.columns:
        v = m["top1_realized_lat"].to_numpy(dtype=float)
        out["realized_lat_med_on_hits"] = float(np.nanmedian(v)) if np.isfinite(v).any() else math.nan
        out["realized_n_hits"] = int(np.isfinite(v).sum())
    return out


def add_baseline_scores(df, seed=0):
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["s_random"] = rng.random(len(df))
    df["s_most_active"] = df["act_90d"] if "act_90d" in df.columns else df["act_total"]
    df["s_most_recent"] = -df["days_since_last"].fillna(np.inf)
    # most active first, ties and near-ties broken by lower workload: rank-normalised sum
    act = df.groupby(["repo", "pr"])["s_most_active"].rank(pct=True)
    load = df.groupby(["repo", "pr"])["workload"].rank(pct=True, ascending=False)
    df["s_least_loaded_active"] = act + load
    lat = df["lat_med_h"].fillna(df["lat_med_h"].max() if df["lat_med_h"].notna().any() else 1e9)
    df["s_fastest"] = -lat
    if "familiarity" in df.columns:
        df["s_most_familiar"] = most_familiar_score(df)
    return df


def most_familiar_score(df, tie_col="s_most_active"):
    """Most familiar: how often the candidate reviewed the author's earlier PRs,
    ties broken by `tie_col` (90-day activity). Lexicographic: the scale exceeds
    any value of the non-negative tie-breaker, so it never outweighs familiarity."""
    tie = df[tie_col].astype(float)
    scale = float(tie.max()) + 1.0 if len(tie) else 1.0
    return df["familiarity"].astype(float) * scale + tie


def evaluate_scores(df, score_cols, by_repo=True):
    """Return a tidy table: one row per (score, repo) with summary metrics."""
    rows = []
    for sc in score_cols:
        m = per_pr_metrics(df, sc)
        if by_repo:
            for repo, g in m.groupby("repo"):
                rows.append({"score": sc, "repo": repo, **summarize(g)})
        rows.append({"score": sc, "repo": "ALL (micro)", **summarize(m)})
    return pd.DataFrame(rows)
