#!/usr/bin/env python3
"""Chapter 4, recalculation on constructed next periods (variants, seeds and measures fixed in
the author's notes before this code and before any of these numbers). An addendum after the results were
seen; no confirmation criterion, a description of stability.

  AUC     saved LORO forecasts of chapter 3 (IT: n_it >= 2; TFR: long wait for the first review,
          labelled requests only), no refitting; the 200 next periods of E9 (generator S,
          seed 20261115, stream 1); per-repository AUC on the next period, mean over repositories.
  rank    fixed ranking model (s_pooled_STRICT of test_scored.parquet); 200 next periods = half of
          each repository's ranking test PRs (stream 8); Hit@1, Hit@3, MRR of the model, the most
          familiar rule and the two tolerance re-rankings (exposure re-run on the next-period stream).
  R4      rule R4 at c = 0.2 on the E9b infrastructure (9 641 PRs, stream 5): thresholds on the
          history, firing share, expected latency of the first candidate before / after and their
          ratio, Hit@1 / Hit@3 / MRR before / after on the firings and on all next-period PRs.
  E6/E7   Hit@1, Hit@3, MRR of the model and the simple rules, paired gains; costs of the three
          re-rankings in Hit@1, Hit@3, MRR on the factual split.
Reproduction of the factual numbers (pred.*.loro.auc, rank.*, r4_noxt.c02.*, E9b firings) is
checked; a mismatch stops the run.

    PY=python   # interpreter with requirements.txt
    $PY revision/r4_calc/r4_newdata.py --dev    # seed 1, 10 variants -> revision/output/r4/newdata/dev/
    $PY revision/r4_calc/r4_newdata.py          # main -> revision/output/r4/newdata/
"""
import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r4_e9_eff as E  # noqa: E402  (adds revision/ and r4_calc/ to sys.path)
import r4_e9b as E9B  # noqa: E402

CLONE = os.path.dirname(E.REV)
sys.path.insert(0, CLONE)
from rank import constrained as C  # noqa: E402
from rank import evaluate as ev  # noqa: E402
from rank import metrics as M  # noqa: E402
import r3_report_stats as S  # noqa: E402

RANK = os.path.join(E.REV, "output", "r3", "rank_shared")
RANK_RES = os.path.join(RANK, "results")
NOXT = os.path.join(E.REV, "output", "r4", "no_overlap_thr")
OUT = os.path.join(E.REV, "output", "r4", "newdata")

SEED_MAIN, SEED_DEV = 20261115, 1
N_VAR, N_VAR_DEV = 200, 10
STREAM_S, STREAM_RANK, STREAM_R4 = 1, 8, 5
C_R4 = 0.2
EPS = 0.1
N_PROC = max(1, (os.cpu_count() or 2) - 2)
TOL = 1e-9
SHORT2FULL = {"angular": "angular/angular", "discourse": "discourse/discourse", "django": "django/django",
              "react": "facebook/react", "grafana": "grafana/grafana", "mattermost": "mattermost/mattermost",
              "vscode": "microsoft/vscode", "node": "nodejs/node", "strapi": "strapi/strapi",
              "supabase": "supabase/supabase", "next.js": "vercel/next.js", "woocommerce": "woocommerce/woocommerce"}
PRS_SHORT = {f.replace("/", "_"): k for k, f in SHORT2FULL.items()}   # prs_*.parquet repo names
SCORES = {"model": "s_pooled_STRICT", "familiar": "s_fam", "random": "s_random", "recent": "s_most_recent",
          "active": "s_most_active", "latency": "s_lat", "exposure": "s_exp"}
MET = ("hit1", "hit3", "mrr")


# ── small helpers ───────────────────────────────────────────────────────────

def vsum(x):
    """Median, 2.5 and 97.5 percentiles and mean over variants (NaN ignored)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"med": None, "lo": None, "hi": None, "mean": None}
    return {"med": float(np.median(x)), "lo": float(np.percentile(x, 2.5)),
            "hi": float(np.percentile(x, 97.5)), "mean": float(x.mean())}


def put_var(nums, prefix, fact, values):
    nums[f"{prefix}.fact"] = None if fact is None else float(fact)
    for k, v in vsum(values).items():
        if k != "mean":
            nums[f"{prefix}.{k}"] = v


def perms(n, seed, stream, i_r, n_var):
    """The variant permutations of one repository, drawn sequentially as in E9 / E9b."""
    rng = np.random.default_rng([seed, stream, i_r])
    return [rng.permutation(n) for _ in range(n_var)]


def same(a, b, tol=TOL):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


# ── AUC ─────────────────────────────────────────────────────────────────────

def auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return math.nan
    return float(roc_auc_score(y, p))


def auc_arrays(t, preds):
    """Per repository (main sample, ordered by (created, row_id) as E9): row_id, forecasts and
    labels of IT and TFR (NaN where TFR is undefined or not forecast)."""
    s = t[(t.hait_filter == 1) & (t.in_main == 1)].copy()
    for k in ("it", "tr"):
        m = pd.Series(preds[k].prob.to_numpy(), index=preds[k].row_id.to_numpy())
        s["p_" + k] = m.reindex(s.row_id.to_numpy()).to_numpy()
    s["y_it"] = s.y_IT__main_D.astype(float)
    s["y_tr"] = s.y_TFR__main_D.astype(float)
    s = s.sort_values(["repo", "created", "row_id"], kind="mergesort")
    return {repo: {c: g[c].to_numpy() for c in ("row_id", "p_it", "y_it", "p_tr", "y_tr")}
            for repo, g in s.groupby("repo", sort=True)}


def auc_on(P, idx):
    out = {}
    for k in ("it", "tr"):
        p, y = P["p_" + k][idx], P["y_" + k][idx]
        ok = np.isfinite(p) & np.isfinite(y)
        out[k] = auc(y[ok], p[ok])
    return out


_AUC = None


def _auc_repo(job):
    repo, i_r, seed, n_var = job
    P = _AUC[repo]
    n = len(P["row_id"])
    h = n // 2
    return repo, [auc_on(P, perm[h:]) for perm in perms(n, seed, STREAM_S, i_r, n_var)]


def auc_part(PA, all_repos, seed, n_var, late=None):
    """Variant AUCs (n_var x repos) for IT and TFR, the factual AUC on all rows and on the E8
    later half."""
    global _AUC
    _AUC = PA
    repos = sorted(PA)
    jobs = [(r, all_repos.index(r), seed, n_var) for r in repos]
    with mp.get_context("fork").Pool(N_PROC) as pool:
        res = dict(pool.map(_auc_repo, jobs))
    A = {k: np.array([[res[r][v][k] for r in repos] for v in range(n_var)]) for k in ("it", "tr")}
    fact_all = {k: np.array([auc_on(PA[r], np.arange(len(PA[r]["row_id"])))[k] for r in repos])
                for k in ("it", "tr")}
    fact_late = None
    if late is not None:
        fact_late = {}
        for k in ("it", "tr"):
            vals = []
            for r in repos:
                pos = pd.Series(np.arange(len(PA[r]["row_id"])), index=PA[r]["row_id"])
                vals.append(auc_on(PA[r], pos.reindex(late[k][r]).to_numpy())[k])
            fact_late[k] = np.array(vals)
    return A, fact_all, fact_late, repos


# ── ranking ─────────────────────────────────────────────────────────────────

def canon_pairs(d, prs):
    """Ranking test pairs in a canonical order (repo, pr, candidate) -- the within-PR order of
    test_scored.parquet -- with `created` of the PR, rankable PRs only."""
    d = d[d.split == "test"]
    d = d[d.groupby(["repo", "pr"])["label"].transform("max") == 1]
    d = d.sort_values(["repo", "pr", "candidate"], kind="mergesort").reset_index(drop=True)
    prs = prs.assign(repo=prs.repo.map(lambda r: PRS_SHORT.get(r, r)))   # as run_experiments.load
    cr = prs.drop_duplicates(["repo", "pr"]).set_index(["repo", "pr"])["created"]
    d["created"] = cr.reindex(pd.MultiIndex.from_frame(d[["repo", "pr"]])).to_numpy()
    assert d.created.notna().all()
    return d


def load_rank():
    d = pd.read_parquet(os.path.join(RANK_RES, "test_scored.parquet"))
    prs = pd.concat([pd.read_parquet(f, columns=["repo", "pr", "created"])
                     for f in sorted(glob.glob(os.path.join(RANK, "prs_*.parquet")))], ignore_index=True)
    return canon_pairs(d, prs)


def add_scores(d):
    d = d.copy()
    d["s_fam"] = ev.most_familiar_score(d)
    d["s_lat"] = C.rerank_constrained(d, "s_pooled_STRICT", eps=EPS, w_lat=1.0, w_exp=0.0)["s_constrained"].to_numpy()
    d["s_exp"] = C.rerank_constrained(d, "s_pooled_STRICT", eps=EPS, w_lat=0.0, w_exp=1.0)["s_constrained"].to_numpy()
    return d


def pr_keys(d):
    """PRs (repo, pr) ordered by (repo, created, pr), with the row slice of each PR in d."""
    g = d.groupby(["repo", "pr"], sort=False)
    first = g.cumcount() == 0
    k = d.loc[first, ["repo", "pr", "created"]].copy()
    k["start"] = np.nonzero(first.to_numpy())[0]
    k["stop"] = np.r_[k.start.to_numpy()[1:], len(d)]
    return k.sort_values(["repo", "created", "pr"], kind="mergesort").reset_index(drop=True)


def per_pr_table(d, keys, col):
    m = ev.per_pr_metrics(d, col).set_index(["repo", "pr"])
    return m.reindex(pd.MultiIndex.from_frame(keys[["repo", "pr"]])).reset_index()


def rank_variant_idx(keys, all_repos, seed, n_var):
    """For each variant: positions (into keys) of the next-period PRs of all 12 repositories."""
    out = [[] for _ in range(n_var)]
    for short in sorted(keys.repo.unique()):
        pos = np.nonzero((keys.repo == short).to_numpy())[0]
        n, h = len(pos), len(pos) // 2
        i_r = all_repos.index(SHORT2FULL[short])
        for v, perm in enumerate(perms(n, seed, STREAM_RANK, i_r, n_var)):
            out[v].append(pos[perm[h:]])
    return [np.sort(np.concatenate(x)) for x in out]


_RK = None


def exposure_on(d, keys, idx):
    """Exposure re-ranking re-run on the stream of the PRs keys[idx] only (counters from zero)."""
    rows = np.concatenate([np.arange(a, b) for a, b in zip(keys.start.to_numpy()[idx], keys.stop.to_numpy()[idx])])
    sub = d.iloc[rows]
    sub = sub.assign(s_expv=C.rerank_constrained(sub, "s_pooled_STRICT", eps=EPS, w_lat=0.0, w_exp=1.0)["s_constrained"].to_numpy())
    m = per_pr_table(sub, keys.iloc[idx], "s_expv")
    return {k: float(m[k].mean()) for k in MET}


def _exp_job(idx):
    d, keys = _RK
    return exposure_on(d, keys, idx)


def rank_part(d, keys, tabs, all_repos, seed, n_var):
    """Per variant: micro means of every method; exposure re-run on the next-period stream."""
    global _RK
    _RK = (d, keys)
    vidx = rank_variant_idx(keys, all_repos, seed, n_var)
    with mp.get_context("fork").Pool(N_PROC) as pool:
        expo = pool.map(_exp_job, vidx)
    rows = []
    for v, idx in enumerate(vidx):
        r = {"v": v, "n_prs": int(len(idx))}
        for meth in ("model", "familiar", "latency"):
            for k in MET:
                r[f"{meth}_{k}"] = float(tabs[meth][k].to_numpy()[idx].mean())
        for k in MET:
            r[f"exposure_{k}"] = expo[v][k]
            r[f"exposure_stream_full_{k}"] = float(tabs["exposure"][k].to_numpy()[idx].mean())
        rows.append(r)
    return pd.DataFrame(rows), vidx


def paired_gain(a, b):
    diff = (a - b).to_numpy(dtype=float)
    lo, hi = M.bootstrap_ci(diff, n_boot=1000, seed=42)
    return float(diff.mean()), lo, hi


# ── R4 ──────────────────────────────────────────────────────────────────────

def r4_measures(rel, lat, fire_list, n_list):
    """Pooled over repositories. rel / lat: dicts of arrays (hit1, hit3, mrr, top1_lat) of the
    next-period PRs (concatenated); fire: bool array of the same length."""
    fire = np.concatenate(fire_list) if fire_list else np.zeros(0, bool)
    n_tot = int(sum(n_list))
    k = int(fire.sum())
    out = {"n_fire": k, "n": n_tot, "fire_pct": 100 * k / n_tot if n_tot else math.nan,
           "fire_pct_repomean": 100 * float(np.mean([f.sum() / n for f, n in zip(fire_list, n_list)]))}
    if k:
        b, a = np.nanmedian(rel["top1_lat"][fire]), np.nanmedian(lat["top1_lat"][fire])
        out.update({"lat_before_h": float(b), "lat_after_h": float(a), "lat_ratio": float(a / b) if b else math.nan,
                    "hit3_before": float(rel["hit3"][fire].mean()), "hit3_after": float(lat["hit3"][fire].mean())})
        out["hit3_drop"] = out["hit3_before"] - out["hit3_after"]
        for m in MET:
            out[f"cost_fired_{m}"] = float(rel[m][fire].mean() - lat[m][fire].mean())
    else:
        for q in ("lat_before_h", "lat_after_h", "lat_ratio", "hit3_before", "hit3_after", "hit3_drop"):
            out[q] = math.nan
        for m in MET:
            out[f"cost_fired_{m}"] = math.nan
    for m in MET:   # change on all next-period PRs: re-ranked only where the rule fired
        out[f"cost_all_{m}"] = float((rel[m][fire] - lat[m][fire]).sum() / n_tot) if n_tot else math.nan
    return out


def r4_factual(tabs, keys):
    """Factual split of r4_no_overlap_thr: thresholds on the earlier half (by created) of the
    main-sample predictions of the 12 repositories, decision on the ranking test PRs."""
    m12 = pd.read_csv(os.path.join(NOXT, "main12_predictions.csv"))
    m12["created"] = pd.to_datetime(m12.created, utc=True)
    theta = {}
    for repo, g in m12.groupby("repo"):
        g = g.sort_values(["created", "row_id"], kind="mergesort")
        early = g.iloc[:len(g) // 2]
        theta[repo] = S.choose_threshold(early.prob.to_numpy(), early.y.to_numpy(), C_R4)
    pp = pd.read_csv(os.path.join(NOXT, "per_pr.csv"))
    pp = pp[pp.p_tr.notna()]
    th = [theta.get(r) for r in pp.repo_full]
    fire = np.array([t is not None and p > t for p, t in zip(pp.p_tr, th)])
    chk = {"theta_equal": bool(all((t is None and pd.isna(x)) or (t is not None and not pd.isna(x) and abs(t - x) < 1e-12)
                                   for t, x in zip(th, pp.theta_c02))),
           "fire_equal": bool(np.array_equal(fire, pp.fire_c02.to_numpy().astype(bool)))}
    kk = pd.MultiIndex.from_frame(keys[["repo", "pr"]])
    pos = pd.Series(np.arange(len(keys)), index=kk).reindex(pd.MultiIndex.from_arrays([pp.repo, pp.pr])).to_numpy()
    rel = {m: tabs["model"][m].to_numpy()[pos] for m in MET}
    lat = {m: tabs["latency"][m].to_numpy()[pos] for m in MET}
    rel["top1_lat"] = tabs["model"]["top1_lat"].to_numpy()[pos]
    lat["top1_lat"] = tabs["latency"]["top1_lat"].to_numpy()[pos]
    # the per-PR values used here must equal the saved per_pr.csv
    chk["per_pr_equal"] = bool(np.allclose(rel["hit3"], pp.hit3_rel, atol=0) and np.allclose(lat["hit3"], pp.hit3_lat, atol=0)
                               and np.allclose(rel["mrr"], pp.mrr_rel, atol=1e-12) and np.allclose(lat["mrr"], pp.mrr_lat, atol=1e-12)
                               and np.allclose(rel["top1_lat"], pp.top1_lat_rel, atol=1e-12, equal_nan=True)
                               and np.allclose(lat["top1_lat"], pp.top1_lat_lat, atol=1e-12, equal_nan=True))
    fl = [fire[(pp.repo_full == r).to_numpy()] for r in sorted(pp.repo_full.unique())]
    order = np.argsort(pp.repo_full.to_numpy(), kind="mergesort")
    relo = {k: v[order] for k, v in rel.items()}
    lato = {k: v[order] for k, v in lat.items()}
    meas = r4_measures(relo, lato, fl, [len(x) for x in fl])
    return meas, chk


def r4_part(t, preds, all_repos, tabs, keys, seed, n_var, dev=False):
    """E9b generator (stream 5), thresholds on the history of each variant, measures on the
    next period, pooled over the 12 repositories."""
    PA4, _ = E9B.r4_arrays(t, preds, all_repos)
    prn = pd.Series(t.pr_number.to_numpy(), index=t.row_id.to_numpy())   # by row_id (row order free)
    kk = pd.Series(np.arange(len(keys)), index=pd.MultiIndex.from_frame(keys[["repo", "pr"]]))
    full2short = {v: k for k, v in SHORT2FULL.items()}
    rel_all, lat_all = {}, {}
    for repo, P in PA4.items():
        pos = kk.reindex(pd.MultiIndex.from_arrays([[full2short[repo]] * len(P["row_id"]),
                                                    prn.reindex(P["row_id"]).to_numpy()])).to_numpy()
        assert not np.isnan(pos.astype(float)).any()
        pos = pos.astype(int)
        rel_all[repo] = {m: tabs["model"][m].to_numpy()[pos] for m in MET + ("top1_lat",)}
        lat_all[repo] = {m: tabs["latency"][m].to_numpy()[pos] for m in MET + ("top1_lat",)}
    repos = sorted(PA4) if not dev else sorted(E9B.DEV_R4)
    per_var = [{"fire": [], "n": [], "rel": [], "lat": [], "k": {}} for _ in range(n_var)]
    for repo in repos:
        P = PA4[repo]
        n, h = len(P["row_id"]), len(P["row_id"]) // 2
        i_r = all_repos.index(repo)
        for v, perm in enumerate(perms(n, seed, STREAM_R4, i_r, n_var)):
            hist, nxt = perm[:h], perm[h:]
            th = E9B.choose_tr(P, hist, C_R4)
            fire = (P["p_tr"][nxt] > th) if th is not None else np.zeros(len(nxt), bool)
            pv = per_var[v]
            pv["fire"].append(fire)
            pv["n"].append(len(nxt))
            pv["rel"].append({m: x[nxt] for m, x in rel_all[repo].items()})
            pv["lat"].append({m: x[nxt] for m, x in lat_all[repo].items()})
            pv["k"][repo] = int(fire.sum())
    rows = []
    for v, pv in enumerate(per_var):
        rel = {m: np.concatenate([x[m] for x in pv["rel"]]) for m in MET + ("top1_lat",)}
        lat = {m: np.concatenate([x[m] for x in pv["lat"]]) for m in MET + ("top1_lat",)}
        rows.append({"v": v, **r4_measures(rel, lat, pv["fire"], pv["n"])})
    ks = pd.DataFrame([{"v": v, "repo": r, "k_tr": k} for v, pv in enumerate(per_var) for r, k in pv["k"].items()])
    return pd.DataFrame(rows), ks, repos, int(sum(len(PA4[r]["row_id"]) for r in repos))


# ── main ────────────────────────────────────────────────────────────────────

def load_inputs():
    cols = ["repo", "pr_number", "in_main", "hait_filter", "created_at", "t_m_h_D", "t_r_h_D",
            "n_it_human", "stale30_D", "y_IT__main_D", "y_ST__main_D", "y_TFR__main_D"]
    t = pd.read_csv(E.TABLE, usecols=cols, low_memory=False)
    t["row_id"] = np.arange(len(t))
    t["created"] = pd.to_datetime(t["created_at"], utc=True)
    preds = {k: E.load_preds(t, tg) for k, tg in (("it", "IT"), ("st", "ST"), ("tr", "TFR"))}
    return t, preds


def rank_tables(d, keys):
    return {meth: per_pr_table(d, keys, col) for meth, col in SCORES.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="seed 1, 10 variants, output in dev/")
    args = ap.parse_args(argv)
    t0 = time.time()
    seed = SEED_DEV if args.dev else SEED_MAIN
    n_var = N_VAR_DEV if args.dev else N_VAR
    out = os.path.join(OUT, "dev") if args.dev else OUT
    os.makedirs(out, exist_ok=True)
    ref = json.load(open(E.R4_NUMBERS, encoding="utf-8"))
    nums, checks, bad = {}, {}, []

    def chk(name, got, exp, tol=TOL):
        ok = same(got, exp, tol)
        checks.setdefault("repro", {})[name] = {"got": got, "expected": exp, "ok": ok}
        if not ok:
            bad.append(name)

    # ── inputs and E8 split ──
    t, preds = load_inputs()
    k5, splits = E.reproduce_e8(t, preds)
    checks["k5_e8_reproduction_ok"] = k5["ok"]
    if not k5["ok"]:
        bad.append("E8 reproduction")
    all_repos = sorted(t.loc[(t.hait_filter == 1) & (t.in_main == 1), "repo"].unique())
    assert len(all_repos) == 40
    print(f"inputs: {time.time() - t0:.0f} s", flush=True)

    # ── AUC ──
    PA = auc_arrays(t, preds)
    PE9 = E.project_arrays(t, preds, None)
    checks["S_order_equals_e9"] = all(np.array_equal(PA[r]["row_id"], PE9[r]["row_id"]) for r in all_repos)
    if not checks["S_order_equals_e9"]:
        bad.append("S order")
    A, fa, fl, repos = auc_part(PA, all_repos, seed, n_var, late={"it": {r: splits["it"][r][1] for r in all_repos},
                                                                  "tr": {r: splits["tr"][r][1] for r in all_repos}})
    for k in ("it", "tr"):
        corpus = np.nanmean(A[k], axis=1)
        nums[f"nd.auc.{k}.fact_all"] = float(np.nanmean(fa[k]))
        nums[f"nd.auc.{k}.fact_late"] = float(np.nanmean(fl[k]))
        for q, v in vsum(corpus).items():
            nums[f"nd.auc.{k}.{q}"] = v
        nums[f"nd.auc.{k}.n_undef_max"] = int(np.isnan(A[k]).sum(axis=1).max())
        nums[f"nd.auc.{k}.n_repos"] = int(np.isfinite(fa[k]).sum())
        chk(f"pred.{k}.loro.auc", nums[f"nd.auc.{k}.fact_all"], ref["pred"][k]["loro"]["auc"])
    va = pd.DataFrame({"v": np.arange(n_var), **{f"auc_{k}": np.nanmean(A[k], axis=1) for k in ("it", "tr")},
                       **{f"n_undef_{k}": np.isnan(A[k]).sum(axis=1) for k in ("it", "tr")}})
    print(f"AUC: {time.time() - t0:.0f} s", flush=True)

    # ── ranking: factual ──
    d = add_scores(load_rank())
    keys = pr_keys(d)
    tabs = rank_tables(d, keys)
    head = pd.read_csv(os.path.join(RANK_RES, "e4v2_headline_test.csv"))
    head = head[head.repo == "ALL (micro)"].set_index("config")
    r1 = pd.read_csv(os.path.join(RANK_RES, "r1_micro.csv")).set_index("config")
    e1 = pd.read_csv(os.path.join(RANK_RES, "e1_baselines.csv"))
    e1 = e1[e1.repo == "ALL (micro)"].set_index("score")
    rk = ref["rank"]
    fact = {meth: {k: float(tabs[meth][k].mean()) for k in MET} for meth in SCORES}
    for k in MET:
        chk(f"rank.ranker_pooled.{k}", fact["model"][k], rk["ranker_pooled"][k if k != "hit1" else "hit1"])
        chk(f"rank.familiar.{k}", fact["familiar"][k], rk["familiar"][k])
        chk(f"r1_micro.most_familiar.{k}", fact["familiar"][k], float(r1.loc["most_familiar", k]))
        chk(f"headline.exposure.{k}", fact["exposure"][k], float(head.loc["constr_e0.1_L0.0_E1.0", k]))
        chk(f"headline.latency.{k}", fact["latency"][k], float(head.loc["constr_e0.1_L1.0_E0.0", k]))
        for meth, col in (("random", "s_random"), ("recent", "s_most_recent"), ("active", "s_most_active")):
            chk(f"e1.{col}.{k}", fact[meth][k], float(e1.loc[col, k]))
    g_h3 = paired_gain(tabs["model"].hit3, tabs["familiar"].hit3)
    chk("rank.gain_vs_familiar.hit3", g_h3[0], rk["gain_vs_familiar"]["hit3"])
    # informational, not a stop: the bootstrap interval depends on the order of the PRs in the array;
    # here (repo, created, pr), in run_r1 the build order (ties in created ordered by the raw dump)
    checks["info_gain_vs_familiar_hit3_ci"] = {"got": [g_h3[1], g_h3[2]],
                                               "r1_familiar_paired": [rk["gain_vs_familiar"]["hit3_lo"],
                                                                      rk["gain_vs_familiar"]["hit3_hi"]]}
    chk("rank.gain_vs_familiar.mrr", paired_gain(tabs["model"].mrr, tabs["familiar"].mrr)[0], rk["gain_vs_familiar"]["mrr"])
    chk("rank.tol_exposure.hit3_cost", fact["model"]["hit3"] - fact["exposure"]["hit3"], rk["tol_exposure"]["hit3_cost"])
    chk("rank.tol_latency.hit3_cost", fact["model"]["hit3"] - fact["latency"]["hit3"], rk["tol_latency"]["hit3_cost"])
    chk("rank.n_test_eval", len(keys), rk["n_test_eval"])
    # E6/E7 table values
    for meth in ("model", "random", "recent", "active", "familiar"):
        for k in MET:
            nums[f"nd.e6.{meth}.{k}"] = fact[meth][k]
    for meth in ("random", "recent", "active", "familiar"):
        for k in MET:
            g = paired_gain(tabs["model"][k], tabs[meth][k])
            nums[f"nd.e6.gain.{meth}.{k}.delta"], nums[f"nd.e6.gain.{meth}.{k}.lo"], nums[f"nd.e6.gain.{meth}.{k}.hi"] = g
    print(f"rank factual: {time.time() - t0:.0f} s", flush=True)

    # ── R4: factual ──
    r4f, r4chk = r4_factual(tabs, keys)
    checks["r4_factual"] = r4chk
    for name, okv in r4chk.items():
        if not okv:
            bad.append(f"r4_factual.{name}")
    rn = ref["r4_noxt"]["c02"]
    for mine, theirs in (("fire_pct", "fire_test_pct"), ("lat_before_h", "exp_latency_relevance_h"),
                         ("lat_after_h", "exp_latency_latency_h"), ("lat_ratio", "exp_latency_ratio"),
                         ("hit3_before", "hit3_relevance"), ("hit3_after", "hit3_latency"), ("hit3_drop", "hit3_drop")):
        chk(f"r4_noxt.c02.{theirs}", r4f[mine], rn[theirs])
    chk("r4_noxt.c02.hit3_change_all", -r4f["cost_all_hit3"], rn["hit3_change_all"])
    chk("r4_noxt.c02.n_fire_test", r4f["n_fire"], rn["n_fire_test"])
    if bad:
        json.dump({"stopped": bad, **checks}, open(os.path.join(out, "checks.json"), "w", encoding="utf-8"),
                  indent=1, ensure_ascii=False, default=str)
        sys.exit(f"factual numbers not reproduced: {bad}")
    print("factual reproduction OK", flush=True)

    # ── ranking: variants ──
    VR, vidx = rank_part(d, keys, tabs, all_repos, seed, n_var)
    for meth in ("model", "familiar", "exposure", "latency"):
        for k in MET:
            put_var(nums, f"nd.rank.{meth}.{k}", fact[meth][k], VR[f"{meth}_{k}"])
    for k in MET:
        put_var(nums, f"nd.{k}_gain", fact["model"][k] - fact["familiar"][k], VR[f"model_{k}"] - VR[f"familiar_{k}"])
        for meth in ("exposure", "latency"):
            put_var(nums, f"nd.cost.{meth}.{k}", fact["model"][k] - fact[meth][k], VR[f"model_{k}"] - VR[f"{meth}_{k}"])
        put_var(nums, f"nd.cost.exposure_fullstream.{k}", fact["model"][k] - fact["exposure"][k],
                VR[f"model_{k}"] - VR[f"exposure_stream_full_{k}"])
    print(f"rank variants: {time.time() - t0:.0f} s", flush=True)

    # ── R4: variants ──
    VR4, ks, repos4, n_prs_r4 = r4_part(t, preds, all_repos, tabs, keys, seed, n_var, dev=args.dev)
    for q in ("fire_pct", "fire_pct_repomean", "lat_before_h", "lat_after_h", "lat_ratio", "hit3_before",
              "hit3_after", "hit3_drop"):
        put_var(nums, f"nd.r4.c02.{q}", r4f[q], VR4[q])
    nums["nd.r4.c02.n_var_no_fire"] = int((VR4.n_fire == 0).sum())
    for k in MET:
        put_var(nums, f"nd.cost.r4_c02.fired.{k}", r4f[f"cost_fired_{k}"], VR4[f"cost_fired_{k}"])
        put_var(nums, f"nd.cost.r4_c02.all.{k}", r4f[f"cost_all_{k}"], VR4[f"cost_all_{k}"])
    if not args.dev:
        e9b = pd.read_csv(os.path.join(E.OUT, "variants_e9b_r4.csv"))
        e9b = e9b[(e9b.kname == "k1") & (e9b.c == "c02")][["v", "repo", "k_tr"]]
        mg = ks.merge(e9b, on=["v", "repo"], suffixes=("", "_e9b"), how="outer")
        checks["e9b_firings_equal"] = bool(len(mg) == len(ks) and (mg.k_tr == mg.k_tr_e9b).all())
        e9n = json.load(open(os.path.join(E.OUT, "numbers_e9b.json"), encoding="utf-8"))
        checks["e9b_fire_pct"] = {"got": float(VR4.fire_pct_repomean.mean()), "expected": e9n["e9.r4.k1.c02.fire_pct"]}
        checks["e9b_fire_pct_ok"] = same(VR4.fire_pct_repomean.mean(), e9n["e9.r4.k1.c02.fire_pct"])
        if not (checks["e9b_firings_equal"] and checks["e9b_fire_pct_ok"]):
            bad.append("E9b firings")
    print(f"R4 variants: {time.time() - t0:.0f} s", flush=True)

    # ── K3: row order of the inputs ──
    rng = np.random.default_rng([seed, 9])
    t_sh = t.iloc[rng.permutation(len(t))].reset_index(drop=True)      # row_id column kept
    preds_sh = {k: v.iloc[rng.permutation(len(v))].reset_index(drop=True) for k, v in preds.items()}
    PA2 = auc_arrays(t_sh, preds_sh)
    k3 = {"auc_arrays_equal": all(np.array_equal(PA[r][c], PA2[r][c], equal_nan=True) for r in all_repos for c in PA[r])}
    A2, _, _, _ = auc_part({r: PA2[r] for r in all_repos[:3]}, all_repos, seed, 3)
    k3["auc_variants_equal"] = all(np.array_equal(A2[k], A[k][:3, :3], equal_nan=True) for k in ("it", "tr"))
    raw = pd.read_parquet(os.path.join(RANK_RES, "test_scored.parquet"))
    prs = pd.concat([pd.read_parquet(f, columns=["repo", "pr", "created"])
                     for f in sorted(glob.glob(os.path.join(RANK, "prs_*.parquet")))], ignore_index=True)
    d2 = canon_pairs(raw.iloc[rng.permutation(len(raw))], prs.iloc[rng.permutation(len(prs))])
    k3["rank_pairs_equal"] = bool(d2.drop(columns="created").equals(d.drop(columns=["created", "s_fam", "s_lat", "s_exp"]))
                                  and np.array_equal(d2.created.to_numpy(), d.created.to_numpy()))
    VR4b, _, _, _ = r4_part(t_sh, preds_sh, all_repos, tabs, keys, seed, 3, dev=args.dev)
    k3["r4_variants_equal"] = bool(np.allclose(VR4b.drop(columns="v").to_numpy(float),
                                               VR4.iloc[:3].drop(columns="v").to_numpy(float), equal_nan=True, rtol=0, atol=0))
    checks["k3_row_order"] = k3
    checks["k3_ok"] = all(k3.values())
    if not checks["k3_ok"]:
        bad.append("K3")

    # ── stop condition (fixed in advance) ──
    flag = []
    for key in sorted(nums):
        if key.startswith("nd.cost.") and not key.startswith("nd.cost.exposure_fullstream") \
                and (".hit1." in key or ".mrr." in key) and key.endswith((".fact", ".med")) \
                and nums[key] is not None and nums[key] > 0.03:
            flag.append(key)
    nums["nd.flag.cost_gt_003"] = flag
    nums.update({"nd.n_variants": n_var, "nd.seed": seed, "nd.n_repos_auc": len(repos),
                 "nd.n_repos_rank": int(keys.repo.nunique()), "nd.n_prs_rank": int(len(keys)),
                 "nd.n_repos_r4": len(repos4), "nd.n_prs_r4": n_prs_r4})
    checks["bad"] = bad
    checks["ok"] = not bad

    run = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "dev" if args.dev else "main", "seed": seed,
           "n_var": n_var, "git": E.git_head(), "seconds": round(time.time() - t0, 1), "ok": not bad, "bad": bad,
           "flag_cost_gt_003": flag,
           "summary": {k: nums.get(k) for k in ("nd.auc.it.med", "nd.auc.it.lo", "nd.auc.it.hi", "nd.auc.tr.med",
                                                "nd.hit3_gain.med", "nd.r4.c02.lat_ratio.med",
                                                "nd.r4.c02.hit3_drop.med", "nd.cost.latency.mrr.fact")}}
    with open(os.path.join(OUT, "runs.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(run, ensure_ascii=False) + "\n")
    checks["runs"] = [json.loads(x) for x in open(os.path.join(OUT, "runs.jsonl"), encoding="utf-8")]
    json.dump(checks, open(os.path.join(out, "checks.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    json.dump({k: nums[k] for k in sorted(nums)}, open(os.path.join(out, "numbers.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    va.to_csv(os.path.join(out, "variants_auc.csv"), index=False)
    VR.to_csv(os.path.join(out, "variants_rank.csv"), index=False)
    VR4.to_csv(os.path.join(out, "variants_r4.csv"), index=False)
    print(json.dumps(run, ensure_ascii=False, indent=1))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
