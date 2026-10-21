"""E0-E4 for the reviewer-ranking article. Reads output/pairs_*.parquet, writes output/results/."""
import time

import numpy as np
import pandas as pd

from rank import evaluate as ev, ranker, tradeoff
from rank.build import OUT_DIR, REPOS, STRICT_FEATURES, CONTENT_FEATURES

RES = OUT_DIR / "results"
SHORT = {"angular_angular": "angular", "discourse_discourse": "discourse", "django_django": "django",
         "facebook_react": "react", "grafana_grafana": "grafana", "mattermost_mattermost": "mattermost",
         "microsoft_vscode": "vscode", "nodejs_node": "node", "strapi_strapi": "strapi",
         "supabase_supabase": "supabase", "vercel_next.js": "next.js", "woocommerce_woocommerce": "woocommerce"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load():
    pairs = pd.concat([pd.read_parquet(OUT_DIR / f"pairs_{r}.parquet") for r in REPOS], ignore_index=True)
    prs = pd.concat([pd.read_parquet(OUT_DIR / f"prs_{r}.parquet") for r in REPOS], ignore_index=True)
    pairs["repo"] = pairs["repo"].map(SHORT); prs["repo"] = prs["repo"].map(SHORT)
    return pairs, prs


def e0(pairs, prs):
    rows = []
    for repo, g in prs.groupby("repo", sort=False):
        elig = g[g.split != "warmup"]
        withpos = elig[elig.n_positives > 0]
        top = pairs[(pairs.repo == repo) & (pairs.label == 1)].candidate.value_counts()
        rows.append({
            "repo": repo, "prs_total": len(g), "warmup": int((g.split == "warmup").sum()),
            "eligible": len(elig), "with_positive": len(withpos), "share_with_positive": len(withpos) / max(len(elig), 1),
            "n_train": int((elig.split == "train").sum()), "n_val": int((elig.split == "val").sum()), "n_test": int((elig.split == "test").sum()),
            "cand_median": float(elig.n_candidates.median()), "pos_median": float(withpos.n_positives.median()),
            "recall_any": float(withpos.any_positive_in_pool.mean()), "recall_all": float(withpos.all_positives_in_pool.mean()),
            "reviewers": int(top.shape[0]), "top10_share": float(top.head(10).sum() / top.sum()),
            "first_review_med_h": float(elig.first_review_h.median()),
            "from": str(g.created.min().date()), "to": str(g.created.max().date()),
        })
    return pd.DataFrame(rows)


def eval_split(df, split):
    d = df[df.split == split]
    # only PRs with at least one positive in pool are rankable
    keep = d.groupby(["repo", "pr"])["label"].transform("max") == 1
    return d[keep]


def main():
    RES.mkdir(exist_ok=True)
    pairs, prs = load()
    log(f"pairs {len(pairs):,} prs {len(prs):,}")
    e0_tab = e0(pairs, prs); e0_tab.to_csv(RES / "e0_dataset.csv", index=False); log("E0 done")

    pairs = ev.add_baseline_scores(pairs, seed=0)
    test = eval_split(pairs, "test"); val = eval_split(pairs, "val"); train = eval_split(pairs, "train")
    log(f"rankable pairs: train {len(train):,} val {len(val):,} test {len(test):,}")

    # ---- E1 baselines --------------------------------------------------
    e1 = ev.evaluate_scores(test, list(ev.BASELINES)); e1.to_csv(RES / "e1_baselines.csv", index=False); log("E1 done")

    # ---- E2 per-repo and pooled lambdarank, STRICT and CONTENT -----------
    feats = {"STRICT": STRICT_FEATURES, "CONTENT": STRICT_FEATURES + CONTENT_FEATURES}
    scored = test.copy()
    imps = []
    for name, F in feats.items():
        # pooled model (all repos' train, early stop on all val)
        m = ranker.train(train, val, F)
        scored[f"s_pooled_{name}"] = ranker.predict(m, scored, F)
        imps.append({"model": f"pooled_{name}", **ranker.importance(m, F)})
        log(f"E2 pooled {name}: best_iter {m.best_iteration}")
        # per-repo models
        col = f"s_perrepo_{name}"; scored[col] = np.nan
        for repo in scored.repo.unique():
            tr, va = train[train.repo == repo], val[val.repo == repo]
            mr = ranker.train(tr, va, F)
            mask = scored.repo == repo
            scored.loc[mask, col] = ranker.predict(mr, scored[mask], F)
            imps.append({"model": f"perrepo_{name}_{repo}", **ranker.importance(mr, F)})
        log(f"E2 per-repo {name} done")
    pd.DataFrame(imps).to_csv(RES / "e2_importance.csv", index=False)

    # ---- E3 LORO (STRICT and CONTENT) ------------------------------------
    for name, F in feats.items():
        col = f"s_loro_{name}"; scored[col] = np.nan
        for repo in scored.repo.unique():
            tr, va = train[train.repo != repo], val[val.repo != repo]
            ml = ranker.train(tr, va, F)
            mask = scored.repo == repo
            scored.loc[mask, col] = ranker.predict(ml, scored[mask], F)
        log(f"E3 LORO {name} done")

    model_cols = [c for c in scored.columns if c.startswith("s_pooled_") or c.startswith("s_perrepo_") or c.startswith("s_loro_")]
    e2 = ev.evaluate_scores(scored, model_cols); e2.to_csv(RES / "e2_e3_models.csv", index=False)
    scored.to_parquet(RES / "test_scored.parquet", index=False)
    log("E2/E3 evaluation done")

    # ---- E4 trade-off sweep: choose lambdas on val, report on test --------
    F = STRICT_FEATURES
    m = ranker.train(train, val, F)
    val_s = val.copy(); val_s["s_rel"] = ranker.predict(m, val_s, F)
    test_s = scored.copy(); test_s["s_rel"] = test_s["s_pooled_STRICT"]
    grid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    rows = []
    for split_name, d in [("val", val_s), ("test", test_s)]:
        for lw in grid:
            for lt in grid:
                rr = tradeoff.rerank(d, "s_rel", lam_w=lw, lam_t=lt)
                mm = ev.per_pr_metrics(rr, "s_tradeoff")
                rows.append({"split": split_name, "lam_w": lw, "lam_t": lt, "repo": "ALL (micro)", **ev.summarize(mm, n_boot=200)})
                for repo, g in mm.groupby("repo"):
                    rows.append({"split": split_name, "lam_w": lw, "lam_t": lt, "repo": repo, **ev.summarize(g, n_boot=200)})
        log(f"E4 sweep on {split_name} done")
    e4 = pd.DataFrame(rows); e4.to_csv(RES / "e4_tradeoff_sweep.csv", index=False)
    micro_test = e4[(e4.split == "test") & (e4.repo == "ALL (micro)")]
    front = tradeoff.pareto_front(micro_test.reset_index(drop=True), maximize=["hit3"], minimize=["gini_top1", "top1_lat_med"])
    front.to_csv(RES / "e4_pareto_front_test.csv", index=False)
    log("E4 done")


if __name__ == "__main__":
    main()
