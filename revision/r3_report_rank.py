"""Chapter 3 number contract: the ranking keys rank.* (except rank.sens.*).

Reads only finished results; no model is retrained:
  - keys:          revision/r3_rank_keys.json (keys and published AAIT values; R3_CONTRACT=<contract.yaml>
                   reads them from the dissertation number contract instead)
  - shared run:    revision/output/r3/rank_shared/   (RANK_ACCOUNTS=shared, r3_night.sh step 05)
  - regex run:     revision/output/r3/rank_regex/    (RANK_ACCOUNTS=regex, step 08, the AAIT filter)
  - raw dumps:     data/raw/<repo>/merged_prs.json (R3_RAW_V1)
                   (the files both runs read; only for rank.reclass.*)
  - constants:     rank/build.py, rank/run_e4.py

Every key is computed the way the AAIT 2026 (ПАІТ) number was produced
(the table script of the AAIT 2026 paper, not in this repository; rank/run_r1.py and the
AAIT text). The same functions run on rank_regex as a self-check against the
published values (pait_value in the contract).

Writes <R3_ROOT>/report/rank/ (default revision/output/r3/report/rank/):
  rank_numbers.json, rank_numbers_status.json, rank_regex_check.csv,
  r3_rank.csv, r3_tradeoff.csv, r3_rank_repo.csv, r3_rank_pool.csv

Run from the repository root:
  python revision/r3_report_rank.py
"""
import json
import sys
import os
from pathlib import Path

import pandas as pd
import yaml

CLONE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLONE))

from rank import build as B          # noqa: E402  constants only
from rank import run_e4 as E4        # noqa: E402  DELTA only

CONTRACT = Path(os.environ["R3_CONTRACT"]) if os.environ.get("R3_CONTRACT") else None
KEYS_JSON = Path(__file__).resolve().parent / "r3_rank_keys.json"
R3 = Path(os.environ["R3_ROOT"]) if os.environ.get("R3_ROOT") else CLONE / "revision/output/r3"
SHARED = R3 / "rank_shared"
REGEX = R3 / "rank_regex"
RAW = Path(os.environ.get("R3_RAW_V1") or CLONE / "data/raw")
OUT = R3 / "report/rank"

# AAIT 2026 order and languages (make_tables.py)
REPOS = ["woocommerce", "react", "vscode", "next.js", "angular", "supabase", "strapi", "grafana",
         "mattermost", "node", "discourse", "django"]
LANG = {"woocommerce": "PHP", "react": "JavaScript", "node": "JavaScript/C++", "vscode": "TypeScript",
        "next.js": "TypeScript", "angular": "TypeScript", "supabase": "TypeScript", "strapi": "TypeScript",
        "grafana": "Go", "mattermost": "Go", "discourse": "Ruby", "django": "Python"}
METHODS = {"random": "s_random", "recent": "s_most_recent", "active": "s_most_active",
           "familiar": "s_most_familiar", "ranker_repo": "s_perrepo_STRICT",
           "ranker_pooled": "s_pooled_STRICT", "ranker_loro": "s_loro_STRICT"}
# make_tables.table_importance: feature groups of the split-gain shares
GROUPS = {"familiarity": ["familiarity"],
          "activity": ["act_7d", "act_30d", "act_90d", "act_total", "days_since_last", "act_30d_share"],
          "author": ["author_prior_prs", "author_med_first_review_h"],
          "workload_latency": ["workload", "lat_med_h", "fast_24h"]}
ALL = "ALL (micro)"


def contract():
    if CONTRACT is None:
        k = json.loads(KEYS_JSON.read_text(encoding="utf-8"))
        return k["keys"], k["pait_value"], None
    c = yaml.safe_load(CONTRACT.read_text())
    flat = [row for row in c["keys_flat"] if str(row[0]).startswith("rank.") and not str(row[0]).startswith("rank.sens.")]
    pait = {}
    for fam in c["families"]:
        key, pv = fam["key"], fam.get("pait_value")
        if not key.startswith("rank.") or key.startswith("rank.sens.") or pv is None:
            continue
        if isinstance(pv, dict):
            var = next(iter(fam["expand"]))
            for v, x in pv.items():
                pait[key.replace("{" + var + "}", v)] = x
        else:
            pait[key] = pv
    return [r[0] for r in flat], pait, c


def load(run):
    R = run / "results"
    e1 = pd.read_csv(R / "e1_baselines.csv"); e2 = pd.read_csv(R / "e2_e3_models.csv")
    return {
        "e0": pd.read_csv(R / "e0_dataset.csv").set_index("repo"),
        "ev": pd.concat([e1, e2], ignore_index=True),
        "imp": pd.read_csv(R / "e2_importance.csv").set_index("model"),
        "head": pd.read_csv(R / "e4v2_headline_test.csv"),
        "val": pd.read_csv(R / "e4v2_val_all.csv"),
        "chosen": json.loads((R / "e4v2_chosen.json").read_text()),
        "paired": pd.read_csv(R / "r1_familiar_paired.csv"),
        "lat": pd.read_csv(R / "r1_latency_paired.csv").set_index("quantity"),
    }


def eps_of(config):
    return float(config.split("_e")[1].split("_L")[0])


def numbers(d):
    """All rank.* keys except rank.reclass.*, rank.shift.* and rank.sens.*."""
    v = {}
    e0, ev = d["e0"], d["ev"]
    micro = ev[ev.repo == ALL].set_index("score")
    v["rank.n_repos"] = int(len(e0))
    v["rank.n_merged"] = int(e0.prs_total.sum())
    v["rank.n_after_warmup"] = int(e0.eligible.sum())
    v["rank.n_test"] = int(e0.n_test.sum())
    v["rank.n_test_eval"] = int(micro.loc["s_pooled_STRICT", "n"])
    v["rank.pool_days"] = int(B.CAND_WINDOW_DAYS)
    v["rank.warmup_days"] = int(B.WARMUP_DAYS)
    v["rank.n_features_strict"] = len(B.STRICT_FEATURES)
    v["rank.pool_recall_any_min_pct"] = float(e0.recall_any.min() * 100)
    v["rank.pool_recall_any_max_pct"] = float(e0.recall_any.max() * 100)
    v["rank.pool_recall_all_min_pct"] = float(e0.recall_all.min() * 100)
    v["rank.pool_recall_all_max_pct"] = float(e0.recall_all.max() * 100)
    for m, col in METHODS.items():
        v[f"rank.{m}.hit3"] = float(micro.loc[col, "hit3"])
        v[f"rank.{m}.mrr"] = float(micro.loc[col, "mrr"])
    for m in ("active", "familiar", "ranker_pooled", "ranker_loro"):
        v[f"rank.{m}.hit3_lo"] = float(micro.loc[METHODS[m], "hit3_lo"])
        v[f"rank.{m}.hit3_hi"] = float(micro.loc[METHODS[m], "hit3_hi"])
    act_repo = ev[(ev.score == "s_most_active") & (ev.repo != ALL)]
    v["rank.active.hit3_repo_min"] = float(act_repo.hit3.min())
    v["rank.active.hit3_repo_max"] = float(act_repo.hit3.max())
    head = d["head"]; hm = head[head.repo == ALL].set_index("config")
    v["rank.active.n_distinct_top1"] = int(hm.loc["most_active", "top1_unique"])
    p = d["paired"]
    pa = p[(p.baseline == "most_active") & (p.metric == "hit3") & (p.repo != "ALL")]
    v["rank.n_repos_ranker_beats_active"] = int(((pa.delta > 0) & (pa.lo > 0)).sum())
    pf = p[(p.baseline == "most_familiar") & (p.repo == "ALL")].set_index("metric")
    v["rank.gain_vs_familiar.hit3"] = float(pf.loc["hit3", "delta"])
    v["rank.gain_vs_familiar.hit3_lo"] = float(pf.loc["hit3", "lo"])
    v["rank.gain_vs_familiar.hit3_hi"] = float(pf.loc["hit3", "hi"])
    v["rank.gain_vs_familiar.mrr"] = float(pf.loc["mrr", "delta"])
    # unconditional Hit@3: test PRs with no actual reviewer among the candidates count as misses for every method
    share = v["rank.n_test_eval"] / v["rank.n_test"]
    v["rank.n_test_no_pool"] = v["rank.n_test"] - v["rank.n_test_eval"]
    v["rank.no_pool_pct"] = (1 - share) * 100
    for m in ("familiar", "ranker_pooled"):
        v[f"rank.{m}.hit3_all"] = v[f"rank.{m}.hit3"] * share
    v["rank.gain_vs_familiar.hit3_all"] = v["rank.gain_vs_familiar.hit3"] * share
    v["rank.loro_drop_hit3"] = float(micro.loc["s_pooled_STRICT", "hit3"] - micro.loc["s_loro_STRICT", "hit3"])
    v["rank.content_delta_mrr"] = float(micro.loc["s_pooled_CONTENT", "mrr"] - micro.loc["s_pooled_STRICT", "mrr"])
    imp = d["imp"].loc["pooled_STRICT"]
    for g, feats in GROUPS.items():
        v[f"rank.gain_share.{g}_pct"] = float(imp[feats].sum() * 100)
    ch = d["chosen"]
    v["rank.tol.value"] = eps_of(ch["constr_lat"])
    v["rank.tol.admissible_pct"] = int(round((1 - E4.DELTA) * 100))
    val = d["val"][d["val"].repo == ALL]
    ref = float(val.loc[val.config == "rel_only", "hit3"].iloc[0])
    w = val[val.config.str.startswith("wsum_")]
    v["rank.weighted.n_admissible"] = int((w.hit3 >= (1 - E4.DELTA) * ref).sum())
    rel, cexp, clat = hm.loc["rel_only"], hm.loc[ch["constr_gini"]], hm.loc[ch["constr_lat"]]
    v["rank.relevance.distinct_top1"] = int(rel.top1_unique)
    v["rank.relevance.gini"] = float(rel.gini_top1_full)
    v["rank.relevance.exp_latency_h"] = float(rel.top1_lat_med)
    v["rank.tol_exposure.hit3_cost"] = float(rel.hit3 - cexp.hit3)
    v["rank.tol_exposure.distinct_top1"] = int(cexp.top1_unique)
    v["rank.tol_exposure.gini"] = float(cexp.gini_top1_full)
    v["rank.tol_latency.hit3_cost"] = float(rel.hit3 - clat.hit3)
    v["rank.tol_latency.exp_latency_h"] = float(clat.top1_lat_med)
    lat = d["lat"]
    v["rank.realized.n_both_hit"] = int(lat.loc["common_rel_only_median_h", "n"])
    v["rank.realized.n_pairs"] = int(lat.loc["diff_top1_paired_median_diff_h", "n"])
    v["rank.realized.faster_pct"] = float(lat.loc["diff_top1_constr_lat_faster_share", "value"] * 100)
    v["rank.realized.median_diff_h"] = float(lat.loc["diff_top1_paired_median_diff_h", "value"])
    v["rank.realized.median_diff_lo_h"] = float(lat.loc["diff_top1_paired_median_diff_h", "lo"])
    v["rank.realized.median_diff_hi_h"] = float(lat.loc["diff_top1_paired_median_diff_h", "hi"])
    v["rank.actual.gini"] = float(rel.gini_actual_full)
    v["rank.actual.n_distinct"] = int(rel.actual_unique)
    return v


def reclass():
    """Interactions (review submissions and review-thread comments by a non-author,
    as rank/events._interactions counts them) that the AAIT regex treats as human
    while the shared classification does not."""
    from rank import events as E
    n_int = n_regex_human = n_rev = 0
    by_class, prs_hit, n_prs_all = {}, 0, 0
    for repo in B.REPOS:
        prs = json.load(open(RAW / repo / "merged_prs.json"))
        for pr in prs:
            if not pr.get("createdAt"):
                continue
            n_prs_all += 1
            author = E._author_login(pr)
            items = [(E._author_login(r), r.get("submittedAt"), E._author_type(r))
                     for r in (pr.get("reviews") or {}).get("nodes") or []]
            items += [(E._author_login(c), c.get("createdAt"), E._author_type(c))
                      for th in (pr.get("reviewThreads") or {}).get("nodes") or []
                      for c in (th.get("comments") or {}).get("nodes") or []]
            touched = False
            for login, ts, typ in items:
                if not (login and E.parse_ts(ts) and login != author):
                    continue
                rx_bot = E.is_bot(login, typ, accounts="regex")
                sh_bot = E.is_bot(login, typ, accounts="shared")
                if not rx_bot:
                    n_regex_human += 1
                    if sh_bot:
                        n_int += 1; touched = True
                        cls = E._shared_class(login, typ)
                        by_class[cls] = by_class.get(cls, 0) + 1
                elif not sh_bot:
                    n_rev += 1
            prs_hit += touched
    return {"n_interactions": n_int, "n_prs": prs_hit, "n_prs_all": n_prs_all,
            "n_regex_human_interactions": n_regex_human, "by_class": by_class,
            "n_reverse_regex_bot_shared_human": n_rev}


def tables(d):
    e0, ev, head = d["e0"], d["ev"], d["head"]
    micro = ev[ev.repo == ALL].set_index("score")
    labels = {"random": "випадковий порядок", "recent": "найнедавніший", "active": "найактивніший (90 днів)",
              "familiar": "найзнайоміший", "ranker_repo": "модель окремого репозиторію",
              "ranker_pooled": "спільна модель", "ranker_loro": "модель інших репозиторіїв (LORO)"}
    rows = []
    for m, col in METHODS.items():
        r = micro.loc[col]
        rows.append({"метод": labels[m], "MRR": r.mrr, "Hit@1": r.hit1, "Hit@3": r.hit3,
                     "Hit@3_lo": r.hit3_lo, "Hit@3_hi": r.hit3_hi, "Hit@5": r.hit5, "NDCG@5": r.ndcg5})
    t_rank = pd.DataFrame(rows)

    ch = d["chosen"]; hm = head[head.repo == ALL].set_index("config")
    trade = [("лише релевантність", "rel_only"), ("допуск + експозиція", ch["constr_gini"]),
             ("допуск + час відповіді", ch["constr_lat"]),
             ("експозиція без допуску", ch["constr_gini:no_tolerance"]),
             ("час відповіді без допуску", ch["constr_lat:no_tolerance"])]
    rows = []
    for lab, cfg in trade:
        r = hm.loc[cfg]
        rows.append({"правило": lab, "конфігурація": cfg, "MRR": r.mrr, "Hit@3": r.hit3, "Hit@3_lo": r.hit3_lo,
                     "Hit@3_hi": r.hit3_hi, "коефіцієнт Джині перших рекомендацій": r.gini_top1_full,
                     "різних людей на першому місці": int(r.top1_unique),
                     "очікуваний час відповіді (год)": r.top1_lat_med,
                     "фактичний час відповіді при влученні (год)": r.realized_lat_med_on_hits})
    r = hm.loc["rel_only"]
    rows.append({"правило": "фактичні рецензенти (довідково)", "конфігурація": "actual",
                 "коефіцієнт Джині перших рекомендацій": r.gini_actual_full,
                 "різних людей на першому місці": int(r.actual_unique)})
    t_trade = pd.DataFrame(rows)

    p = d["paired"]
    pm = p[(p.baseline == "most_active") & (p.metric == "mrr")].set_index("repo")
    rows = []
    for repo in REPOS:
        g = ev[ev.repo == repo].set_index("score")
        rows.append({"репозиторій (мова)": f"{repo} ({LANG[repo]})", "запитів (тест)": int(g.loc["s_most_active", "n"]),
                     "найактивніший Hit@3": g.loc["s_most_active", "hit3"],
                     "модель окремого репозиторію Hit@3": g.loc["s_perrepo_STRICT", "hit3"],
                     "спільна модель Hit@3": g.loc["s_pooled_STRICT", "hit3"],
                     "модель LORO Hit@3": g.loc["s_loro_STRICT", "hit3"],
                     "спільна модель MRR": g.loc["s_pooled_STRICT", "mrr"],
                     "модель LORO MRR": g.loc["s_loro_STRICT", "mrr"],
                     "різниця MRR спільна − найактивніший": pm.loc[repo, "delta"],
                     "різниця MRR спільна − найактивніший_lo": pm.loc[repo, "lo"],
                     "різниця MRR спільна − найактивніший_hi": pm.loc[repo, "hi"]})
    t_repo = pd.DataFrame(rows)

    rows = []
    for repo in REPOS:
        x = e0.loc[repo]
        rows.append({"репозиторій (мова)": f"{repo} ({LANG[repo]})", "злитих": int(x.prs_total),
                     "після періоду накопичення": int(x.eligible), "тестових": int(x.n_test),
                     "медіана розміру множини кандидатів": x.cand_median, "рецензентів": int(x.reviewers),
                     "частка 10 найчастіших (%)": x.top10_share * 100,
                     "повнота множини: хоча б один (%)": x.recall_any * 100,
                     "повнота множини: усі (%)": x.recall_all * 100,
                     "медіана першої рецензії (год)": x.first_review_med_h})
    rows.append({"репозиторій (мова)": "Разом", "злитих": int(e0.prs_total.sum()),
                 "після періоду накопичення": int(e0.eligible.sum()), "тестових": int(e0.n_test.sum())})
    t_pool = pd.DataFrame(rows)
    for c in ("злитих", "після періоду накопичення", "тестових", "рецензентів"):
        t_pool[c] = t_pool[c].astype("Int64")
    return {"r3_rank.csv": t_rank, "r3_tradeoff.csv": t_trade, "r3_rank_repo.csv": t_repo, "r3_rank_pool.csv": t_pool}


def main():
    keys, pait, _ = contract()
    assert len(keys) == 79, len(keys)   # 72 + rank.realized_regex.* (02.11.2026) + unconditional Hit@3 (03.11.2026)
    sh, rx = load(SHARED), load(REGEX)
    v_sh, v_rx = numbers(sh), numbers(rx)

    rc = reclass()
    v_sh["rank.reclass.n_interactions"] = rc["n_interactions"]
    v_sh["rank.reclass.n_prs"] = rc["n_prs"]
    v_sh["rank.reclass.share_pct"] = rc["n_interactions"] / rc["n_regex_human_interactions"] * 100
    v_sh["rank.shift.hit3"] = v_sh["rank.ranker_pooled.hit3"] - pait["rank.ranker_pooled.hit3"]
    v_sh["rank.shift.mrr"] = v_sh["rank.ranker_pooled.mrr"] - pait["rank.ranker_pooled.mrr"]
    v_rx["rank.shift.hit3"] = v_rx["rank.ranker_pooled.hit3"] - pait["rank.ranker_pooled.hit3"]
    v_rx["rank.shift.mrr"] = v_rx["rank.ranker_pooled.mrr"] - pait["rank.ranker_pooled.mrr"]
    # realized latency check with the paper's filter, for the comparison in appendix В.6
    v_sh["rank.realized_regex.n_pairs"] = v_rx["rank.realized.n_pairs"]
    v_sh["rank.realized_regex.median_diff_h"] = v_rx["rank.realized.median_diff_h"]

    out = {k: v_sh.get(k) for k in keys}
    missing = [k for k in keys if k not in v_sh]
    extra = [k for k in v_sh if k not in keys]
    assert not extra, extra

    status = {k: "не обчислено" for k in missing}
    status.update({
        "rank.n_repos_ranker_beats_active":
            "означення контракту (Hit@3, парний інтервал без нуля, різниця > 0) застосовано до r1_familiar_paired.csv "
            "(pooled − most_active, hit3, бутстреп 1000, seed 42 на клітинку). У ПАІТ число 11 взято з Табл. 3, де парна "
            "різниця за MRR (e2_paired_deltas.csv); за MRR у rank_shared теж 11 (django: різниця < 0, інтервал без нуля), "
            "за Hit@3 – 11 (django: інтервал містить нуль).",
        "rank.gain_vs_familiar.hit3":
            "середня парна різниця по запитах pooled STRICT − most familiar (r1_familiar_paired.csv, рядок ALL); "
            "most familiar з розв'язанням нічиїх за act_90d, як у ПАІТ.",
        "rank.gain_share.workload_latency_pct":
            "сума груп «Workload» (workload) і «Responsiveness» (lat_med_h, fast_24h) з make_tables.table_importance "
            "(у ПАІТ 0,6 + 0,8 = 1,4). Група «Time of creation» (hour, weekday) у контракті не має ключа: "
            "rank_shared 0,2 %, ПАІТ 0,1 %.",
        "rank.gain_share.activity_pct":
            "група «Candidate activity»: act_7d, act_30d, act_90d, act_total, days_since_last, act_30d_share (make_tables).",
        "rank.tol.value":
            "eps обраної конфігурації constr_lat у e4v2_chosen.json; constr_gini обрано з тим самим eps (0,1).",
        "rank.tol.admissible_pct": "(1 − DELTA)·100 з rank/run_e4.py (DELTA = 0,05), стала, не результат.",
        "rank.weighted.n_admissible":
            "кількість конфігурацій wsum_* (ненульова вага) у e4v2_val_all.csv (ALL, валідація), де "
            "Hit@3 ≥ 0,95 · Hit@3 rel_only; обрана wsum – rel_only.",
        "rank.tol_exposure.hit3_cost":
            "Hit@3 rel_only − Hit@3 constr_e0.1_L0.0_E1.0 (e4v2_headline_test.csv, ALL), без округлення доданків.",
        "rank.tol_latency.hit3_cost":
            "Hit@3 rel_only − Hit@3 constr_e0.1_L1.0_E0.0 (e4v2_headline_test.csv, ALL), без округлення доданків.",
        "rank.relevance.distinct_top1":
            "top1_unique (різні логіни на першому місці по всіх запитах разом), як у Табл. 4 ПАІТ; "
            "не сума по репозиторіях (top1_unique_repo_sum).",
        "rank.realized.n_both_hit":
            "n рядка common_rel_only_median_h у r1_latency_paired.csv (перший кандидат обох конфігурацій рецензував).",
        "rank.realized.faster_pct": "частка пар (діф. top-1), де constr_lat швидший, ×100; не округлено (формат int).",
        "rank.ranker_repo.hit3":
            "модель окремого репозиторію = s_perrepo_STRICT (Табл. 2 ПАІТ «Ranker, per-repository, STRICT»).",
        "rank.content_delta_mrr": "MRR s_pooled_CONTENT − MRR s_pooled_STRICT (e2_e3_models.csv, ALL).",
        "rank.loro_drop_hit3": "Hit@3 s_pooled_STRICT − Hit@3 s_loro_STRICT (ALL).",
        "rank.reclass.n_interactions":
            f"рахунок за rank/events._interactions по сирих merged_prs.json 12 репозиторіїв (усі злиті запити, "
            f"включно з періодом накопичення): регулярний вираз ПАІТ – людина, спільна класифікація – не human. "
            f"За класами: {rc['by_class']}. Зворотний випадок (вираз – бот, спільна – людина): "
            f"{rc['n_reverse_regex_bot_shared_human']} взаємодій; у ключ не входить.",
        "rank.reclass.n_prs": f"запити (з {rc['n_prs_all']} у вивантаженні), де є хоча б одна така взаємодія.",
        "rank.reclass.share_pct":
            f"знаменник – взаємодії, які фільтр ПАІТ рахував людськими ({rc['n_regex_human_interactions']}); "
            "означення знаменника в контракті не задано.",
        "rank.shift.hit3": "перераховане (rank_shared, без округлення) − pait_value 0,673 з контракту.",
        "rank.shift.mrr": "перераховане (rank_shared, без округлення) − pait_value 0,569 з контракту.",
    })

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rank_numbers.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    (OUT / "rank_numbers_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1))
    chk = pd.DataFrame([{"key": k, "rank_shared": v_sh.get(k), "rank_regex": v_rx.get(k), "pait_value": pait.get(k)}
                        for k in keys])
    chk.to_csv(OUT / "rank_regex_check.csv", index=False, encoding="utf-8")
    for name, t in tables(sh).items():
        t.to_csv(OUT / name, index=False, encoding="utf-8")
    print(json.dumps(rc, ensure_ascii=False))
    print(f"keys {len(keys)}, filled {sum(v is not None for v in out.values())}, missing {missing}")


if __name__ == "__main__":
    main()
