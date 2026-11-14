#!/usr/bin/env python3
"""Worked example of one pull request through both chapter 3 methods
(appendix В of the dissertation).

The PR is fixed in advance: a ranking test PR of the later half of its project
whose LORO prediction of a long wait exceeds the project threshold at c = 0.2,
so that rule R4 turns on the latency re-ranking. Candidates are reported as
A, B, C in relevance order, never by login.

Inputs (read only): v2/pr_table_v2.csv, v2/models/metrics.jsonl and oof/,
report/details/thresholds_by_repo.csv, rank_shared/results/test_scored.parquet.
Output: report/numbers_example.json

    R3_ROOT=revision/output/r3_b40d python revision/r3_report_example.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from rank import constrained as C  # noqa: E402

R3 = os.environ.get("R3_ROOT") or os.path.join(_HERE, "output", "r3")
REP = os.path.join(R3, "report")
REPO, SHORT, PR = "facebook/react", "react", 34406
ETA, C_RULE = 0.1, "c02"                     # tolerance of the chosen latency configuration; c = 0,2
TARGETS = {"it": "IT", "st": "ST", "tr": "TFR"}


def loro_prob(tab, recs, target, row_id):
    """LORO prediction of the PR and whether it lies in the later half of the project."""
    r = next(r for r in recs if r["variant"] == "main" and r["target"] == target and r["protocol"] == "loro"
             and r["model"] == "CatBoost" and r["test"] == REPO and r.get("features") == "t0"
             and r.get("pop") == "main")
    z = np.load(os.path.join(R3, "v2", "models", "oof", r["oof_file"]))
    order = tab.set_index("row_id").loc[z["row_id"]].sort_values("created_at").index.to_list()
    return float(z["prob"][list(z["row_id"]).index(row_id)]), order.index(row_id) >= len(order) // 2


def main():
    tab = pd.read_csv(os.path.join(R3, "v2", "pr_table_v2.csv"), low_memory=False)
    tab["row_id"] = np.arange(len(tab))
    row = tab[(tab.repo == REPO) & (tab.pr_number == PR)].iloc[0]
    recs = [json.loads(line) for line in open(os.path.join(R3, "v2", "models", "metrics.jsonl"))]
    thr = pd.read_csv(os.path.join(REP, "details", "thresholds_by_repo.csv"))
    thr = thr[thr.repo == REPO]

    v = {"example.repo": REPO, "example.pr": PR,
         "example.created": pd.Timestamp(row.created_at).strftime("%d.%m.%Y"),
         "example.lines_added": int(row.lines_added_t0), "example.lines_deleted": int(row.lines_deleted_t0),
         "example.commits": int(row.num_commits_t0),
         "example.author_prs": int(row.author_pr_count), "example.author_merge_pct": 100 * float(row.author_merge_rate),
         "example.author_tfr_h": float(row.author_avg_tfr_human), "example.open_prs": int(row.open_pr_count_t0),
         "example.merges_7d": int(row.merge_velocity_7d_t0),
         "example.n_it": int(row.n_it_human), "example.stale": "так" if row.stale30_D == 1 else "ні",
         "example.t_r_h": float(row.t_r_h_D), "example.thr_tr_h": float(row.thr_tr_h__main_D)}
    assert row.in_main == 1 and row.hait_filter == 1
    for k, tg in TARGETS.items():
        p, late = loro_prob(tab, recs, tg, int(row.row_id))
        assert late, f"{k}: the example PR must lie in the later half of the project"
        v[f"example.{k}.p"] = p
        th = thr[(thr.k == k)].set_index("c")["theta"]
        v[f"example.{k}.n_reachable"] = int(th.notna().sum())
    th_tr = float(thr[(thr.k == "tr") & (thr.c == C_RULE)].theta.iloc[0])
    v["example.tr.theta"] = th_tr
    assert v["example.tr.p"] > th_tr, "R4 must fire on the example PR"

    t = pd.read_parquet(os.path.join(R3, "rank_shared", "results", "test_scored.parquet"))
    t = t[t.split == "test"]
    t = t[t.groupby(["repo", "pr"])["label"].transform("max") == 1]
    chosen = json.load(open(os.path.join(R3, "rank_shared", "results", "e4v2_chosen.json")))["constr_lat"]
    assert chosen == f"constr_e{ETA}_L1.0_E0.0", chosen
    t["s_rel"] = t["s_pooled_STRICT"]
    t["s_lat"] = C.rerank_constrained(t, "s_rel", eps=ETA, w_lat=1.0, w_exp=0.0)["s_constrained"].to_numpy()
    g = t[(t.repo == SHORT) & (t.pr == PR)].copy()
    lo, hi = g.s_rel.min(), g.s_rel.max()
    g["s_norm"] = (g.s_rel - lo) / (hi - lo)
    g = g.sort_values("s_rel", ascending=False).reset_index(drop=True)
    g["name"] = ["A", "B", "C"] + [""] * (len(g) - 3)
    v["example.n_cand"] = int(len(g))
    v["example.n_near"] = int((g.s_norm >= 1 - ETA).sum())
    v["example.eta"] = ETA
    for i, name in enumerate("abc"):
        x = g.iloc[i]
        v[f"example.{name}.s_norm"] = float(x.s_norm)
        v[f"example.{name}.lat_h"] = float(x.lat_med_h)
        v[f"example.{name}.actual"] = "так" if x.label == 1 else "ні"
    lat_order = g.sort_values("s_lat", ascending=False).name.head(3).to_list()
    v["example.rel_top3"] = ", ".join(g.name.head(3))
    v["example.lat_top3"] = ", ".join(lat_order)
    first = g[g.name == lat_order[0]].iloc[0]
    v["example.first_lat.realized_h"] = float(first.realized_lat_h)
    json.dump(v, open(os.path.join(REP, "numbers_example.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(v, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
