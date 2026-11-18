#!/usr/bin/env python3
"""Chapter 4, ranking sensitivity row "admissibility bound by MRR" (fixed in the author's notes before this
code and before any of these numbers). Added after the results were seen; the main bound stays
Hit@3.

Same level (95 %), same family, objectives and tie rule as run_e4.select_and_headline, same
validation table; only the admissibility measure is MRR instead of Hit@3. Reported on the test
PRs next to the main (Hit@3) choice: tolerance, Gini of first places, distinct first-place people,
median expected latency of the first candidate, cost in Hit@1, Hit@3, MRR. Then the costs on the
200 next periods of r4_newdata (generator G_rank) and rule R4 at c = 0.2 with the MRR latency
tolerance (factual split and the E9b generator G_R4).

    PY=python   # interpreter with requirements.txt
    $PY revision/r4_calc/r4_mrr_tol.py          # main -> revision/output/r4/mrr_tol/
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r4_newdata as N  # noqa: E402  (adds revision/, r4_calc/ and the clone root to sys.path)
from rank import constrained as C  # noqa: E402
from rank.selection import select_on_validation  # noqa: E402

E = N.E
OUT = os.path.join(E.REV, "output", "r4", "mrr_tol")
ND_OUT = os.path.join(E.REV, "output", "r4", "newdata")
DELTA = 0.05                        # run_e4.DELTA: the main level, 95 %
SELECT_OBJ = {"gini": "gini_top1_full", "lat": "top1_lat_med"}   # run_e4.SELECT_OBJ
ROLE = {"exposure": "constr_gini", "latency": "constr_lat"}
MICRO = "ALL (micro)"
MET = N.MET
TOL = N.TOL


def parse_constr(name):
    """(eta, w_lat, w_exp) of a constr_* config; rel_only -> (0, 0, 0) (no re-ranking)."""
    if name == "rel_only":
        return 0.0, 0.0, 0.0
    eps = float(name.split("_e")[1].split("_L")[0])
    L = float(name.split("_L")[1].split("_E")[0])
    Ew = float(name.split("_E")[1])
    return eps, L, Ew


def constr_family(micro):
    """The tolerance family of run_e4.select_and_headline: rel_only and constr_* without eta = 1."""
    return micro[(micro.config.str.startswith("constr_") & ~micro.config.str.contains("_e1.0_"))
                 | (micro.config == "rel_only")]


def select(micro, accuracy, delta=DELTA):
    """Validation choice for both objectives with the given admissibility measure."""
    fam = constr_family(micro)
    return {f"constr_{obj}": select_on_validation(fam, "rel_only", delta, col, accuracy=accuracy)
            for obj, col in SELECT_OBJ.items()}


def score(d, cfg, col="s_pooled_STRICT"):
    if cfg == "rel_only":
        return d[col].to_numpy()
    eps, L, Ew = parse_constr(cfg)
    return C.rerank_constrained(d, col, eps=eps, w_lat=L, w_exp=Ew)["s_constrained"].to_numpy()


# ── variants (G_rank of r4_newdata) ─────────────────────────────────────────

_RK = None


def _stream_job(job):
    """Hit@1/3, MRR of a config re-run on the stream of the PRs keys[idx] (exposure from zero)."""
    cfg, idx = job
    d, keys = _RK
    rows = np.concatenate([np.arange(a, b) for a, b in zip(keys.start.to_numpy()[idx], keys.stop.to_numpy()[idx])])
    sub = d.iloc[rows]
    sub = sub.assign(s_v=score(sub, cfg))
    m = N.per_pr_table(sub, keys.iloc[idx], "s_v")
    return {k: float(m[k].mean()) for k in MET}


def variant_costs(d, keys, tabs, cfgs, vidx):
    """Per variant and config: cost = model - config in Hit@1, Hit@3, MRR. Configs with exposure
    weight are re-run on the next-period stream (as r4_newdata.exposure_on); the others are per-PR."""
    global _RK
    _RK = (d, keys)
    out = {}
    stream = [c for c in cfgs if parse_constr(c)[2] > 0]
    res = {}
    if stream:
        jobs = [(c, idx) for c in stream for idx in vidx]
        with mp.get_context("fork").Pool(N.N_PROC) as pool:
            r = pool.map(_stream_job, jobs)
        for (c, _), v in zip(jobs, r):
            res.setdefault(c, []).append(v)
    for c in cfgs:
        rows = []
        for v, idx in enumerate(vidx):
            base = {k: float(tabs["model"][k].to_numpy()[idx].mean()) for k in MET}
            if c in res:
                val = res[c][v]
            else:
                val = {k: float(tabs[c][k].to_numpy()[idx].mean()) for k in MET}
            rows.append({k: base[k] - val[k] for k in MET})
        out[c] = pd.DataFrame(rows)
    return out


# ── main ────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-variants", action="store_true", help="factual split only")
    args = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    nums, checks, bad = {}, {"repro": {}}, []

    def chk(name, got, exp, tol=TOL):
        ok = N.same(got, exp, tol) if not isinstance(exp, str) else got == exp
        checks["repro"][name] = {"got": got, "expected": exp, "ok": ok}
        if not ok:
            bad.append(name)

    ref = json.load(open(E.R4_NUMBERS, encoding="utf-8"))
    rk = ref["rank"]
    ndn = json.load(open(os.path.join(ND_OUT, "numbers.json"), encoding="utf-8"))
    chosen_ref = json.load(open(os.path.join(N.RANK_RES, "e4v2_chosen.json"), encoding="utf-8"))

    # ── validation choice ──
    val = pd.read_csv(os.path.join(N.RANK_RES, "e4v2_val_all.csv"))
    val = val[val.repo == MICRO]
    test = pd.read_csv(os.path.join(N.RANK_RES, "e4v2_test_all.csv"))
    test = test[test.repo == MICRO].set_index("config")
    ch = {"hit3": select(val, "hit3"), "mrr": select(val, "mrr")}
    for role in ("constr_gini", "constr_lat"):
        chk(f"chosen_hit3.{role}", ch["hit3"][role], chosen_ref[role])
    checks["chosen"] = ch
    vmi = val.set_index("config")
    for basis in ("hit3", "mrr"):
        ref_v = float(vmi.loc["rel_only", basis])
        nums[f"mt.val.rel_only.{basis}"] = ref_v
        nums[f"mt.val.bound.{basis}"] = (1 - DELTA) * ref_v
        nums[f"mt.val.n_admissible.{basis}"] = int((constr_family(val)[basis] >= (1 - DELTA) * ref_v).sum())

    # ── test, factual ──
    d = N.add_scores(N.load_rank())
    keys = N.pr_keys(d)
    tabs = N.rank_tables(d, keys)
    cfgs = sorted({c for b in ch.values() for c in b.values()})
    # the main configs are already scored by r4_newdata.add_scores (eta = 0.1, as e4v2_chosen.json)
    tabs.setdefault(ch["hit3"]["constr_gini"], tabs["exposure"])
    tabs.setdefault(ch["hit3"]["constr_lat"], tabs["latency"])
    tabs["rel_only"] = tabs["model"]
    for c in cfgs:
        if c not in tabs:
            d[f"s_{c}"] = score(d, c)
            tabs[c] = N.per_pr_table(d, keys, f"s_{c}")
    rel = test.loc["rel_only"]
    nums.update({"mt.rel.gini": float(rel.gini_top1_full), "mt.rel.distinct_top1": int(rel.top1_unique),
                 "mt.rel.exp_latency_h": float(rel.top1_lat_med)})
    for k in MET:
        nums[f"mt.rel.{k}"] = float(rel[k])
    for role, rname in ROLE.items():
        for basis in ("hit3", "mrr"):
            cfg = ch[basis][rname]
            row = test.loc[cfg]
            eps, L, Ew = parse_constr(cfg)
            p = f"mt.{role}.{basis}"
            nums[f"{p}.config"] = cfg
            nums[f"{p}.eta"] = eps
            nums[f"{p}.w_lat"], nums[f"{p}.w_exp"] = L, Ew
            nums[f"{p}.gini"] = float(row.gini_top1_full)
            nums[f"{p}.distinct_top1"] = int(row.top1_unique)
            nums[f"{p}.exp_latency_h"] = float(row.top1_lat_med)
            nums[f"{p}.val_{basis}_ratio"] = float(vmi.loc[cfg, basis] / vmi.loc["rel_only", basis])
            for k in MET:
                nums[f"{p}.{k}"] = float(row[k])
                nums[f"{p}.cost.{k}"] = float(rel[k] - row[k])
                chk(f"recompute.{cfg}.{k}", float(tabs[cfg][k].mean()), float(row[k]))
    # main rows = chapter 4 numbers and r4_newdata
    chk("rank.relevance.gini", nums["mt.rel.gini"], rk["relevance"]["gini"])
    chk("rank.relevance.distinct_top1", nums["mt.rel.distinct_top1"], rk["relevance"]["distinct_top1"])
    chk("rank.relevance.exp_latency_h", nums["mt.rel.exp_latency_h"], rk["relevance"]["exp_latency_h"])
    chk("rank.tol_exposure.gini", nums["mt.exposure.hit3.gini"], rk["tol_exposure"]["gini"])
    chk("rank.tol_exposure.distinct_top1", nums["mt.exposure.hit3.distinct_top1"], rk["tol_exposure"]["distinct_top1"])
    chk("rank.tol_exposure.hit3_cost", nums["mt.exposure.hit3.cost.hit3"], rk["tol_exposure"]["hit3_cost"])
    chk("rank.tol_latency.exp_latency_h", nums["mt.latency.hit3.exp_latency_h"], rk["tol_latency"]["exp_latency_h"])
    chk("rank.tol_latency.hit3_cost", nums["mt.latency.hit3.cost.hit3"], rk["tol_latency"]["hit3_cost"])
    for role in ROLE:
        for k in MET:
            chk(f"nd.cost.{role}.{k}.fact", nums[f"mt.{role}.hit3.cost.{k}"], ndn[f"nd.cost.{role}.{k}.fact"])
    print(f"factual: {time.time() - t0:.0f} s; chosen {ch}", flush=True)

    # ── R4 at c = 0.2, factual split ──
    lat_mrr = ch["mrr"]["constr_lat"]
    r4f_main, r4chk = N.r4_factual(tabs, keys)
    for name, okv in r4chk.items():
        chk(f"r4_factual_main.{name}", bool(okv), True)
    rn = ref["r4_noxt"]["c02"]
    for mine, theirs in (("fire_pct", "fire_test_pct"), ("lat_ratio", "exp_latency_ratio"), ("hit3_drop", "hit3_drop")):
        chk(f"r4_noxt.c02.{theirs}", r4f_main[mine], rn[theirs])
    tabs_m = {**tabs, "latency": tabs[lat_mrr]}
    r4f_mrr, chk_m = N.r4_factual(tabs_m, keys)
    chk("r4_factual_mrr.theta_equal", bool(chk_m["theta_equal"]), True)
    chk("r4_factual_mrr.fire_equal", bool(chk_m["fire_equal"]), True)
    for basis, r in (("hit3", r4f_main), ("mrr", r4f_mrr)):
        for q in ("fire_pct", "lat_before_h", "lat_after_h", "lat_ratio"):
            nums[f"mt.r4.{basis}.{q}.fact"] = float(r[q])
        for k in MET:
            nums[f"mt.r4.{basis}.cost_fired.{k}.fact"] = float(r[f"cost_fired_{k}"])
            nums[f"mt.r4.{basis}.cost_all.{k}.fact"] = float(r[f"cost_all_{k}"])

    VR, VR4 = None, {}
    if not args.no_variants:
        # ── 200 next periods ──
        t, preds = N.load_inputs()
        all_repos = sorted(t.loc[(t.hait_filter == 1) & (t.in_main == 1), "repo"].unique())
        vidx = N.rank_variant_idx(keys, all_repos, N.SEED_MAIN, N.N_VAR)
        vcfg = sorted(set(cfgs))
        VC = variant_costs(d, keys, tabs, vcfg, vidx)
        for role, rname in ROLE.items():
            for basis in ("hit3", "mrr"):
                cfg = ch[basis][rname]
                for k in MET:
                    for q, v in N.vsum(VC[cfg][k]).items():
                        if q != "mean":
                            nums[f"mt.{role}.{basis}.cost.{k}.{q}"] = v
            for k in MET:
                for q in ("med", "lo", "hi"):
                    chk(f"nd.cost.{role}.{k}.{q}", nums[f"mt.{role}.hit3.cost.{k}.{q}"], ndn[f"nd.cost.{role}.{k}.{q}"])
        print(f"rank variants: {time.time() - t0:.0f} s", flush=True)
        # R4 on G_R4
        nd_r4 = pd.read_csv(os.path.join(ND_OUT, "variants_r4.csv"))
        for basis, tb in (("hit3", tabs), ("mrr", tabs_m)):
            V4, _, _, _ = N.r4_part(t, preds, all_repos, tb, keys, N.SEED_MAIN, N.N_VAR)
            VR4[basis] = V4
            for q in ("fire_pct", "lat_ratio"):
                for s, v in N.vsum(V4[q]).items():
                    if s != "mean":
                        nums[f"mt.r4.{basis}.{q}.{s}"] = v
            for k in MET:
                for part in ("cost_fired", "cost_all"):
                    for s, v in N.vsum(V4[f"{part}_{k}"]).items():
                        if s != "mean":
                            nums[f"mt.r4.{basis}.{part}.{k}.{s}"] = v
        checks["r4_variants_main_equal_newdata"] = bool(np.allclose(
            VR4["hit3"].drop(columns="v").to_numpy(float), nd_r4.drop(columns="v").to_numpy(float),
            equal_nan=True, rtol=0, atol=TOL))
        if not checks["r4_variants_main_equal_newdata"]:
            bad.append("r4 variants main != newdata")
        VR = pd.concat([VC[c].add_prefix(f"{c}__cost_") for c in vcfg], axis=1)
        VR.insert(0, "v", np.arange(len(VR)))
        print(f"R4 variants: {time.time() - t0:.0f} s", flush=True)

    nums.update({"mt.level_pct": round(100 * (1 - DELTA)), "mt.n_test_prs": int(len(keys)),
                 "mt.n_variants": 0 if args.no_variants else N.N_VAR, "mt.seed": N.SEED_MAIN})
    checks["bad"] = bad
    checks["ok"] = not bad
    run = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "factual" if args.no_variants else "main",
           "git": E.git_head(), "seconds": round(time.time() - t0, 1), "ok": not bad, "bad": bad,
           "chosen_mrr": ch["mrr"], "chosen_hit3": ch["hit3"]}
    with open(os.path.join(OUT, "runs.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(run, ensure_ascii=False) + "\n")
    json.dump(checks, open(os.path.join(OUT, "checks.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False,
              default=str)
    json.dump({k: nums[k] for k in sorted(nums)}, open(os.path.join(OUT, "numbers.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    if VR is not None:
        VR.to_csv(os.path.join(OUT, "variants_cost.csv"), index=False)
        for basis, V4 in VR4.items():
            V4.to_csv(os.path.join(OUT, f"variants_r4_{basis}.csv"), index=False)
    print(json.dumps(run, ensure_ascii=False, indent=1))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
