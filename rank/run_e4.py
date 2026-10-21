"""E4 v2: trade-off between reproduction accuracy, concentration and expected latency.

Configurations are chosen on the validation split; test is evaluated only for
the chosen ones (plus untuned baselines). The full grid on test is written as a
sensitivity table, not as the headline result.
"""
import itertools
import time

import pandas as pd

from rank import constrained as C, evaluate as ev, ranker, tradeoff
from rank import metrics as M
from rank.build import STRICT_FEATURES
from rank.run_experiments import RES, eval_split, load, log
from rank.selection import select_on_validation

DELTA = 0.05
WSUM_GRID = [0.1, 0.2, 0.3, 0.5, 1.0]
EPS_GRID = [0.02, 0.05, 0.1, 0.2, 0.3]
WEIGHTS = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]


def score_configs(df):
    """Return dict config_name -> score column name, adding columns to df in place."""
    cfg = {"rel_only": "s_rel", "most_active": "s_most_active", "most_recent": "s_most_recent",
           "least_loaded_active": "s_least_loaded_active", "fastest": "s_fastest",
           "moc2r_like": "s_moc2r_like", "rebai_like": "s_rebai_like"}
    for lw, lt in itertools.product([0.0] + WSUM_GRID, [0.0] + WSUM_GRID):
        if lw == 0 and lt == 0:
            continue
        name = f"wsum_w{lw}_t{lt}"
        df[name] = tradeoff.rerank(df, "s_rel", lam_w=lw, lam_t=lt)["s_tradeoff"].to_numpy()
        cfg[name] = name
    for eps in EPS_GRID + [1.0]:
        for wl, we in WEIGHTS:
            name = f"constr_e{eps}_L{wl}_E{we}"
            df[name] = C.rerank_constrained(df, "s_rel", eps=eps, w_lat=wl, w_exp=we)["s_constrained"].to_numpy()
            cfg[name] = name
    return cfg


def candidate_universe(df):
    return {repo: sorted(g.candidate.unique()) for repo, g in df.groupby("repo")}


def evaluate_all(df, cfg, split_name):
    uni = candidate_universe(df)
    rows = []
    for name, col in cfg.items():
        m = ev.per_pr_metrics(df, col)
        rows.append({"split": split_name, "config": name, "repo": "ALL (micro)", **ev.summarize(m, n_boot=500, universe=uni)})
        for repo, g in m.groupby("repo"):
            rows.append({"split": split_name, "config": name, "repo": repo, **ev.summarize(g, n_boot=300, universe=uni)})
    return pd.DataFrame(rows)


def actual_concentration(df):
    """Gini of actual review participation on the same PRs and the same candidate
    universe as the recommendations (zeros included), plus the non-zero variant."""
    uni = candidate_universe(df)
    pos = df[df.label == 1]
    per = pos.groupby(["repo", "candidate"]).size().to_dict()
    rows = []
    allvals = []
    for repo, cands in uni.items():
        vals = [per.get((repo, c), 0) for c in cands]
        allvals += vals
        rows.append({"repo": repo, "gini_actual_full": M.gini(vals), "gini_actual": M.gini([v for v in vals if v > 0]),
                     "actual_unique": int(sum(v > 0 for v in vals)), "universe_size": len(cands)})
    rows.append({"repo": "ALL (micro)", "gini_actual_full": M.gini(allvals), "gini_actual": M.gini([v for v in allvals if v > 0]),
                 "actual_unique": int(sum(v > 0 for v in allvals)), "universe_size": len(allvals)})
    return pd.DataFrame(rows)


def parse_constr(name):
    eps = float(name.split("_e")[1].split("_L")[0]); L = float(name.split("_L")[1].split("_E")[0]); E = float(name.split("_E")[1])
    return eps, L, E


def ablations_of(name):
    """Switch off exactly one component of a constrained config; skip ablations that
    would leave no secondary criterion."""
    if not name.startswith("constr_"):
        return {}
    eps, L, E = parse_constr(name)
    out = {"no_tolerance": f"constr_e1.0_L{L}_E{E}"}
    if E > 0 and L > 0:
        out["no_exposure"] = f"constr_e{eps}_L{L}_E0.0"
        out["no_latency"] = f"constr_e{eps}_L0.0_E{E}"
    return out


SELECT_OBJ = {"gini": "gini_top1_full", "lat": "top1_lat_med"}


def select_and_headline():
    """Validation-based selection and the headline test table, from the saved CSVs."""
    ev_val = pd.read_csv(RES / "e4v2_val_all.csv"); ev_test = pd.read_csv(RES / "e4v2_test_all.csv")
    conc = pd.read_csv(RES / "e4v2_actual_concentration.csv")
    micro = ev_val[ev_val.repo == "ALL (micro)"]
    fam_w = micro[micro.config.str.startswith("wsum_") | (micro.config == "rel_only")]
    fam_c = micro[(micro.config.str.startswith("constr_") & ~micro.config.str.contains("_e1.0_")) | (micro.config == "rel_only")]
    chosen = {f"{fam}_{obj}": select_on_validation(df, "rel_only", DELTA, col)
              for fam, df in (("wsum", fam_w), ("constr", fam_c)) for obj, col in SELECT_OBJ.items()}
    log(f"E4v2 chosen on validation: {chosen}")
    headline = ["rel_only", "most_active", "most_recent", "least_loaded_active", "fastest", "moc2r_like", "rebai_like"]
    headline += [c for c in chosen.values() if c not in headline]
    abl = {}
    for role in ("constr_gini", "constr_lat"):
        for k, v in ablations_of(chosen[role]).items():
            abl[f"{role}:{k}"] = v
            if v not in headline:
                headline.append(v)
    head = ev_test[ev_test.config.isin(headline)].copy()
    roles = {}
    for k, v in chosen.items():
        roles.setdefault(v, []).append(k)
    for k, v in abl.items():
        roles.setdefault(v, []).append(k)
    head["role"] = head.config.map(lambda c: ";".join(roles.get(c, [])))
    head = head.merge(conc, on="repo", how="left")
    head["config_order"] = head.config.map({c: i for i, c in enumerate(headline)})
    head = head.sort_values(["repo", "config_order"]).drop(columns="config_order")
    head.to_csv(RES / "e4v2_headline_test.csv", index=False)
    pd.Series({**chosen, **abl}).to_json(RES / "e4v2_chosen.json")
    return chosen, abl


def main():
    t = time.time()
    pairs, prs = load()
    pairs = ev.add_baseline_scores(pairs, seed=0)
    pairs = C.add_inspired_baselines(pairs)
    created = prs.set_index(["repo", "pr"])["created"]
    pairs["created"] = created.reindex(pd.MultiIndex.from_frame(pairs[["repo", "pr"]])).to_numpy()
    train, val, test = (eval_split(pairs, s) for s in ("train", "val", "test"))
    log(f"E4v2 loaded: train {len(train):,} val {len(val):,} test {len(test):,}")

    model = ranker.train(train, val, STRICT_FEATURES)
    val = val.copy(); test = test.copy()
    val["s_rel"] = ranker.predict(model, val, STRICT_FEATURES)
    test["s_rel"] = ranker.predict(model, test, STRICT_FEATURES)
    log(f"E4v2 relevance model: best_iter {model.best_iteration}")

    cfg_val = score_configs(val); log("E4v2 val configs scored")
    ev_val = evaluate_all(val, cfg_val, "val"); ev_val.to_csv(RES / "e4v2_val_all.csv", index=False); log("E4v2 val evaluated")

    cfg_test = score_configs(test); log("E4v2 test configs scored")
    ev_test = evaluate_all(test, cfg_test, "test"); ev_test.to_csv(RES / "e4v2_test_all.csv", index=False)
    conc = actual_concentration(test); conc.to_csv(RES / "e4v2_actual_concentration.csv", index=False)
    select_and_headline()
    log(f"E4v2 done in {time.time()-t:.0f}s")


if __name__ == "__main__":
    import sys
    if "--select-only" in sys.argv:
        select_and_headline()
    else:
        main()
