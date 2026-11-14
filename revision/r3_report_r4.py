#!/usr/bin/env python3
"""Rule R4 cost on the ranking test PRs of the 12 corpus v1 repositories
(dissertation, chapter 3: cost of R4).

Inputs (read only):
  report/r4_v1_predictions.csv     P_tr of the v1 PRs (r3_report_models.py --block r4)
  report/details/thresholds_by_repo.csv   theta of threshold 5 per repo (k = tr)
  rank_shared/results/test_scored.parquet, e4v2_chosen.json
Outputs: report/numbers_r4.json, report/status_r4.json, report/tables/r3_r4.csv,
         report/details/r4_cost.csv
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)
from rank import constrained as C  # noqa: E402
from rank import evaluate as ev  # noqa: E402

R3 = os.environ.get("R3_ROOT") or os.path.join(_HERE, "output", "r3")
REP = os.path.join(R3, "report")
RANK = os.path.join(R3, "rank_shared", "results")
CR = {"c01": 0.1, "c02": 0.2, "c03": 0.3}
SHORT2FULL = {"angular": "angular/angular", "discourse": "discourse/discourse", "django": "django/django",
              "react": "facebook/react", "grafana": "grafana/grafana", "mattermost": "mattermost/mattermost",
              "vscode": "microsoft/vscode", "node": "nodejs/node", "strapi": "strapi/strapi",
              "supabase": "supabase/supabase", "next.js": "vercel/next.js", "woocommerce": "woocommerce/woocommerce"}


def parse_constr(name):
    eps = float(name.split("_e")[1].split("_L")[0])
    L = float(name.split("_L")[1].split("_E")[0])
    E = float(name.split("_E")[1])
    return eps, L, E


def per_pr(test, chosen):
    """Per-PR Hit@3 and top-1 expected latency for relevance only and for the
    chosen latency-tolerance configuration."""
    test = test.copy()
    test["s_rel"] = test["s_pooled_STRICT"]          # = rel_only of E4 (same model, checked below)
    eps, L, E = parse_constr(chosen)
    test["s_lat"] = C.rerank_constrained(test, "s_rel", eps=eps, w_lat=L, w_exp=E)["s_constrained"].to_numpy()
    a = ev.per_pr_metrics(test, "s_rel")[["repo", "pr", "hit3", "mrr", "top1_lat"]]
    b = ev.per_pr_metrics(test, "s_lat")[["repo", "pr", "hit3", "mrr", "top1_lat"]]
    return a.merge(b, on=["repo", "pr"], suffixes=("_rel", "_lat"))


def cost_table(m, fire_col):
    f = m[fire_col]
    mixed = np.where(f, m.hit3_lat, m.hit3_rel)
    sub = m[f]
    return {"n_fire": int(f.sum()), "fire_pct": 100 * float(f.mean()),
            "hit3_rel_all": float(m.hit3_rel.mean()), "hit3_with_r4_all": float(mixed.mean()),
            "cost_all": float(m.hit3_rel.mean() - mixed.mean()),
            "hit3_relevance": float(sub.hit3_rel.mean()) if len(sub) else math.nan,
            "hit3_latency": float(sub.hit3_lat.mean()) if len(sub) else math.nan,
            "exp_latency_relevance_h": float(sub.top1_lat_rel.median()) if len(sub) else math.nan,
            "exp_latency_latency_h": float(sub.top1_lat_lat.median()) if len(sub) else math.nan}


def main():
    chosen = json.load(open(os.path.join(RANK, "e4v2_chosen.json")))["constr_lat"]
    test = pd.read_parquet(os.path.join(RANK, "test_scored.parquet"))
    test = test[test.split == "test"]
    test = test[test.groupby(["repo", "pr"])["label"].transform("max") == 1]
    m = per_pr(test, chosen)
    # consistency with the saved E4 test table
    e4 = pd.read_csv(os.path.join(RANK, "e4v2_test_all.csv"))
    e4 = e4[e4.repo == "ALL (micro)"].set_index("config")["hit3"]
    check = {"hit3_rel_here": float(m.hit3_rel.mean()), "hit3_rel_e4": float(e4["rel_only"]),
             "hit3_lat_here": float(m.hit3_lat.mean()), "hit3_lat_e4": float(e4[chosen])}
    m["repo_full"] = m.repo.map(SHORT2FULL)
    pv = pd.read_csv(os.path.join(REP, "r4_v1_predictions.csv"))[["repo", "pr_number", "p_tr"]]
    m = m.merge(pv.rename(columns={"repo": "repo_full", "pr_number": "pr"}), on=["repo_full", "pr"], how="left")
    n_missing = int(m.p_tr.isna().sum())
    thr = pd.read_csv(os.path.join(REP, "details", "thresholds_by_repo.csv"))
    thr = thr[thr.k == "tr"]
    num, status, rows, det = {}, {}, [], []
    num["r4.n_prs_eval"] = int(m.p_tr.notna().sum())
    for c, cr in CR.items():
        th = thr[thr.c == c].set_index("repo")["theta"].to_dict()
        m[f"theta_{c}"] = m.repo_full.map(th)
        m[f"fire_{c}"] = m.p_tr.notna() & m[f"theta_{c}"].notna() & (m.p_tr > m[f"theta_{c}"])
        x = cost_table(m[m.p_tr.notna()], f"fire_{c}")
        for q in ("hit3_relevance", "hit3_latency", "exp_latency_relevance_h", "exp_latency_latency_h"):
            num[f"r4.{c}.{q}"] = None if math.isnan(x[q]) else x[q]
            if math.isnan(x[q]):
                status[f"r4.{c}.{q}"] = "Р4 не спрацював на жодному тестовому запиті"
        unreach = sorted(r for r in SHORT2FULL.values() if r not in th or pd.isna(th.get(r)))
        det.append({"c": c, **x, "repos_unreachable": ";".join(unreach)})
        rows.append({"c_r": str(cr).replace(".", ","), "частка запитів зі спрацюванням (%; 40 репозиторіїв)":
                     None, "частка тестових запитів ранжування зі спрацюванням (%; 12 репозиторіїв)": round(x["fire_pct"], 1),
                     "Hit@3 лише релевантність": round(x["hit3_relevance"], 3) if x["n_fire"] else "",
                     "Hit@3 з переранжуванням за часом": round(x["hit3_latency"], 3) if x["n_fire"] else "",
                     "очікуваний час відповіді до і після (год; 12 репозиторіїв)":
                         f"{x['exp_latency_relevance_h']:.2f} → {x['exp_latency_latency_h']:.2f}" if x["n_fire"] else "",
                     "ціна: зміна Hit@3 на всіх тестових запитах": round(-x["cost_all"], 4)})
    m["fire_all"] = m.p_tr.notna()
    bound = cost_table(m[m.p_tr.notna()], "fire_all")
    det.append({"c": "межа: усі запити", **bound, "repos_unreachable": ""})
    rows.append({"c_r": "межа: Р4 на всіх тестових запитах", "частка тестових запитів ранжування зі спрацюванням (%; 12 репозиторіїв)": 100.0,
                 "Hit@3 лише релевантність": round(bound["hit3_relevance"], 3),
                 "Hit@3 з переранжуванням за часом": round(bound["hit3_latency"], 3),
                 "очікуваний час відповіді до і після (год; 12 репозиторіїв)":
                     f"{bound['exp_latency_relevance_h']:.2f} → {bound['exp_latency_latency_h']:.2f}",
                 "ціна: зміна Hit@3 на всіх тестових запитах": round(-bound["cost_all"], 4)})
    os.makedirs(os.path.join(REP, "tables"), exist_ok=True)
    pd.DataFrame(det).to_csv(os.path.join(REP, "details", "r4_cost.csv"), index=False)
    m.to_csv(os.path.join(REP, "details", "r4_per_pr.csv"), index=False)
    json.dump({"rows": rows, "check": check, "n_missing_p_tr": n_missing, "chosen": chosen,
               "p_tr_v1_mean": float(m.p_tr.mean())},
              open(os.path.join(REP, "details", "r4_detail.json"), "w"), ensure_ascii=False, indent=1)
    pd.DataFrame(rows).to_csv(os.path.join(REP, "tables", "r3_r4_partial.csv"), index=False)
    json.dump(num, open(os.path.join(REP, "numbers_r4.json"), "w"), ensure_ascii=False, indent=1)
    json.dump(status, open(os.path.join(REP, "status_r4.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(check), "missing p_tr:", n_missing)
    print(pd.DataFrame(det).to_string())


if __name__ == "__main__":
    main()
