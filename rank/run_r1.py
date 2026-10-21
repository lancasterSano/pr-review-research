"""Review round 1 of the AAIT 2026 paper: two checks on the saved test scores.

1. Most-familiar baseline: rank candidates by how often they reviewed the
   author's earlier pull requests (ties broken by 90-day activity) and compare
   it with the pooled STRICT ranker, overall, per repository and paired per PR.
2. Paired realized latency: relevance only against constrained re-ranking by
   expected latency (eps = 0.1), on the pull requests where the top-1 candidate
   of both configurations actually reviewed.

Reads rank/output/results/test_scored.parquet written by run_experiments (no
model is retrained) and writes rank/output/results/r1_*.csv.
"""
import numpy as np
import pandas as pd

from rank import constrained as C, evaluate as ev
from rank import metrics as M
from rank.run_e4 import candidate_universe
from rank.run_experiments import RES, eval_split, log

N_BOOT = 1000
SEED = 42
REL = "s_pooled_STRICT"
LAT = "s_constr_lat"
FAMILIAR_CONFIGS = {
    "most_active": "s_most_active",
    "most_familiar": "s_most_familiar",
    "most_familiar_no_tiebreak": "s_familiarity_only",
    "ranker_pooled_STRICT": REL,
}


def load_test(path=RES / "test_scored.parquet"):
    d = pd.read_parquet(path)
    return eval_split(d, "test").copy()


def add_r1_scores(d):
    d = d.copy()
    d["s_most_familiar"] = ev.most_familiar_score(d)
    d["s_familiarity_only"] = d["familiarity"].astype(float)
    d[LAT] = C.rerank_constrained(d, REL, eps=0.1, w_lat=1.0, w_exp=0.0)["s_constrained"].to_numpy()
    return d


def summary_rows(d, configs, n_boot=N_BOOT, seed=SEED):
    """Micro-average and per-repository summaries, counted as in Table 4
    (`top1_unique` over all PRs, Gini over the candidate universe)."""
    uni = candidate_universe(d)
    per_pr, micro, per_repo = {}, [], []
    for name, col in configs.items():
        m = ev.per_pr_metrics(d, col)
        per_pr[name] = m
        s = ev.summarize(m, n_boot=n_boot, seed=seed, universe=uni)
        s["top1_unique_repo_sum"] = int(m.groupby("repo").top1.nunique().sum())
        micro.append({"config": name, "score": col, "repo": "ALL (micro)", **s})
        for repo, g in m.groupby("repo"):
            per_repo.append({"config": name, "score": col, "repo": repo,
                             **ev.summarize(g, n_boot=n_boot, seed=seed, universe=uni)})
    return per_pr, pd.DataFrame(micro), pd.DataFrame(per_repo)


def paired_deltas(m_a, m_b, metrics=("hit3", "mrr"), n_boot=N_BOOT, seed=SEED):
    """Mean per-PR difference a - b with a bootstrap interval, per repository and overall."""
    a = m_a.set_index(["repo", "pr"])
    b = m_b.set_index(["repo", "pr"]).reindex(a.index)
    rows = []
    for repo in sorted(a.index.get_level_values("repo").unique()) + ["ALL"]:
        mask = np.ones(len(a), bool) if repo == "ALL" else (a.index.get_level_values("repo") == repo)
        for metric in metrics:
            diff = (a[metric] - b[metric]).to_numpy(dtype=float)[mask]
            lo, hi = M.bootstrap_ci(diff, n_boot=n_boot, seed=seed)
            rows.append({"repo": repo, "metric": metric, "n": int(diff.size), "delta": float(diff.mean()), "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def no_familiarity_share(d):
    """Share of PRs where no candidate in the pool has reviewed the author before."""
    z = d.groupby(["repo", "pr"]).familiarity.max() == 0
    rows = [{"repo": repo, "n": int(g.size), "share_no_familiar_candidate": float(g.mean())}
            for repo, g in z.groupby(level="repo")]
    rows.append({"repo": "ALL", "n": int(z.size), "share_no_familiar_candidate": float(z.mean())})
    return pd.DataFrame(rows)


def bootstrap_median_ci(values, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    meds = np.median(x[idx], axis=1)
    return float(np.quantile(meds, alpha / 2)), float(np.quantile(meds, 1 - alpha / 2))


def paired_latency(m_rel, m_lat, n_boot=N_BOOT, seed=SEED):
    """Realized latency of the top-1 candidate, compared on the same PRs.

    Rows: quantity, n, value, lo, hi. `value` is a median in hours or a share.
    """
    r = m_rel.set_index(["repo", "pr"])
    lat = m_lat.set_index(["repo", "pr"]).reindex(r.index)
    rel_h, lat_h = r["top1_realized_lat"], lat["top1_realized_lat"]
    same = (r["top1"] == lat["top1"]).to_numpy()
    both = (rel_h.notna() & lat_h.notna()).to_numpy()
    diff_set = both & ~same
    x = (lat_h - rel_h).to_numpy(dtype=float)[diff_set]
    nan = float("nan")
    rows = [
        ("hits_rel_only", int(rel_h.notna().sum()), float(rel_h.median()), nan, nan),
        ("hits_constr_lat", int(lat_h.notna().sum()), float(lat_h.median()), nan, nan),
        ("same_top1_share_all_prs", int(len(r)), float(same.mean()), nan, nan),
        ("common_rel_only_median_h", int(both.sum()), float(rel_h[both].median()), nan, nan),
        ("common_constr_lat_median_h", int(both.sum()), float(lat_h[both].median()), nan, nan),
        ("common_same_top1", int((both & same).sum()), nan, nan, nan),
        ("diff_top1_rel_only_median_h", int(diff_set.sum()), float(rel_h[diff_set].median()), nan, nan),
        ("diff_top1_constr_lat_median_h", int(diff_set.sum()), float(lat_h[diff_set].median()), nan, nan),
        ("diff_top1_constr_lat_faster_share", int(diff_set.sum()), float((x < 0).mean()) if x.size else nan, nan, nan),
    ]
    lo, hi = bootstrap_median_ci(x, n_boot=n_boot, seed=seed) if x.size else (nan, nan)
    rows.append(("diff_top1_paired_median_diff_h", int(x.size), float(np.median(x)) if x.size else nan, lo, hi))
    return pd.DataFrame(rows, columns=["quantity", "n", "value", "lo", "hi"])


def main():
    d = add_r1_scores(load_test())
    log(f"R1 loaded: {len(d):,} pairs, {d.groupby(['repo', 'pr']).ngroups:,} PRs")

    per_pr, micro, per_repo = summary_rows(d, {**FAMILIAR_CONFIGS, "constr_lat_e0.1": LAT})
    micro.to_csv(RES / "r1_micro.csv", index=False)
    per_repo.to_csv(RES / "r1_per_repo.csv", index=False)

    deltas = []
    for base in ("most_familiar", "most_active"):
        p = paired_deltas(per_pr["ranker_pooled_STRICT"], per_pr[base])
        deltas.append(p.assign(model="pooled", baseline=base))
    pd.concat(deltas, ignore_index=True)[["repo", "model", "baseline", "metric", "n", "delta", "lo", "hi"]] \
        .to_csv(RES / "r1_familiar_paired.csv", index=False)
    no_familiarity_share(d).to_csv(RES / "r1_familiar_coverage.csv", index=False)
    log("R1 most familiar done")

    paired_latency(per_pr["ranker_pooled_STRICT"], per_pr["constr_lat_e0.1"]).to_csv(RES / "r1_latency_paired.csv", index=False)
    log("R1 paired latency done")


if __name__ == "__main__":
    main()
