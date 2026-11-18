#!/usr/bin/env python3
"""Chapter 4, experiment E9 "Efficiency on forecast data" (variants, seeds, measures and
criteria fixed in the author's notes before any E9 result was seen).
Only a failed K5 (reproduction of E8) stops the run; K1-K3 are recorded in checks.json.

Question: which share of the Eff losses of a project's next period do the requests the
rules act on carry, if the action removes the problem (upper bound), and how much more than
the same number of random requests of that period?

  Eff(p) = a F^tm(t_m) + b F^nit(n_it) + g stale (chapter 2, eq:eff), F = mid-rank CDF over
  the history of the variant (fixed normalisation, variant B of r4_e9_baseline.py).
  Rules: R_it = R1/R2 on the LORO forecast P_it (same firings for the same c), R_st = R3 on
  P_st, R_any = both. Threshold = r3_report_stats.choose_threshold (n_f = 20) on the history,
  decision p > theta on the next period. Unreachable threshold -> the rule does not act (A = 0).
  Removal: R_it -> n_it := 1 if n_it >= 2; R_st -> stale := 0; t_m unchanged.
  A = sum_K delta / L, A_rand = (k / n) sum_N delta / L (expectation for k random requests),
  lift = A / A_rand. For R_any the components are additive, so A_any = A_it + A_st and
  A_rand,any = A_rand,it + A_rand,st (each rule's own k random requests).

Variant schemes per project (rows = whole requests, ordered by (created_at, row_id) first):
  shuf   200 random permutations; first n // 2 rows = history, the rest = next period.
  block  100 weekly block bootstraps inside the E8 halves (history from the earlier half,
         next period from the later half; calendar weeks of created, UTC, Monday start).
  fact   the E8 split itself (one variant).
Corpus per variant: mean over the 40 projects; lift(v) = mean A / mean A_rand. Mean over
projects with a cluster bootstrap over repositories (B = 10 000, seed 20261115, percentile 95 %).
Controls: K1 null action, K2 shuffled forecasts (threshold reachability; top-k by the
shuffled forecast with the rule's k -> lift ~ 1), K3 input row order, K5 E8 reproduction
(stop on mismatch). K4 (oracle) is a unit test.

Reads only; writes revision/output/r4/e9_eff/{numbers.json, by_repo.csv, variants.csv,
checks.json} (main run) or revision/output/r4/e9_eff/dev/ (--dev: seed 1, 3 projects);
every run is appended to revision/output/r4/e9_eff/runs.jsonl.

    PY=python   # interpreter with requirements.txt
    $PY revision/r4_calc/r4_e9_eff.py --dev     # debugging
    $PY revision/r4_calc/r4_e9_eff.py           # main run
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
sys.path.insert(0, REV)
sys.path.insert(0, HERE)
import r3_report_stats as S  # noqa: E402
from r4_e9_baseline import mid_cdf  # noqa: E402

R3_ROOT = os.environ.get("R3_ROOT") or os.path.join(REV, "output", "r3_b40d")
V2 = os.path.join(R3_ROOT, "v2")
TABLE = os.path.join(V2, "pr_table_v2.csv")
METRICS = os.path.join(V2, "models", "metrics.jsonl")
THR_FILE = os.path.join(R3_ROOT, "report", "details", "thresholds_by_repo.csv")
# chapter 4 numbers of the dissertation (nested JSON), used only for the reconciliation checks
R4_NUMBERS = os.environ.get("R4_NUMBERS_JSON") or os.path.join(REV, "r4_calc", "r4_numbers.json")
OUT = os.path.join(REV, "output", "r4", "e9_eff")

SEED_MAIN = 20261115
SEED_DEV = 1
DEV_REPOS = ["PrestaShop/PrestaShop", "Homebrew/brew", "cilium/cilium"]
N_SHUF, N_BLOCK = 200, 100
N_SHUF_DEV, N_BLOCK_DEV = 20, 10
B = 10000
CR = {"c01": 0.1, "c02": 0.2, "c03": 0.3}
WTS = {"w3": (1 / 3, 1 / 3, 1 / 3), "tm": (1.0, 0.0, 0.0), "nit": (0.0, 1.0, 0.0),
       "st": (0.0, 0.0, 1.0)}
RULES = ("it", "st", "any")
STREAM = {"shuf": 1, "block": 2, "ctrl": 3, "order": 4}
MIN_FIRES = S.MIN_FIRES


# ── thresholds ──────────────────────────────────────────────────────────────

def choose_threshold(p, y, c_r, min_fires=MIN_FIRES):
    """Vectorised r3_report_stats.choose_threshold (equality is unit-tested): the smallest
    theta such that p > theta gives >= min_fires rows with precision >= 1 - c_r; None if
    there is none. The qualifying top runs are those ending at the last index of a run of
    equal scores with a positive score; the smallest theta belongs to the longest one."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(p) < min_fires:
        return None
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order]
    tp = np.cumsum(ys)
    last = np.r_[np.nonzero(np.diff(ps))[0], len(ps) - 1]
    n = last + 1
    ok = (ps[last] > 0.0) & (n >= min_fires) & (tp[last] / n >= 1.0 - c_r - 1e-12)
    if not ok.any():
        return None
    i = last[ok][-1]
    return float(ps[i + 1]) if i + 1 < len(ps) else 0.0


# ── data ────────────────────────────────────────────────────────────────────

def load_table(path=TABLE):
    cols = ["repo", "in_main", "hait_filter", "created_at", "t_m_h_D", "n_it_human",
            "stale30_D", "y_IT__main_D", "y_ST__main_D", "y_TFR__main_D"]
    t = pd.read_csv(path, usecols=cols, low_memory=False)
    t["row_id"] = np.arange(len(t))
    t["created"] = pd.to_datetime(t["created_at"], utc=True)
    return t


def load_preds(t, target, metrics=METRICS):
    """Literal copy of r3_report.preds("main", target, "loro") on the r3_b40d run."""
    m = pd.read_json(metrics, lines=True)
    m["oof_dir"] = os.path.join(os.path.dirname(metrics), "oof")
    q = m[(m.variant == "main") & (m.target == target) & (m.protocol == "loro")
          & (m.model == "CatBoost") & (m.clock == "D") & (m["pop"] == "main")]
    q = q[q.label == target]
    parts = []
    for _, r in q.iterrows():
        z = np.load(os.path.join(r.oof_dir, r.oof_file))
        parts.append(pd.DataFrame({"row_id": z["row_id"], "prob": z["prob"].astype(float),
                                   "test": r.test, "rep": int(r.rep)}))
    d = pd.concat(parts, ignore_index=True)
    lab = t[f"y_{target}__main_D"].to_numpy()
    d["y"] = lab[d.row_id.to_numpy()]
    d["repo"] = t["repo"].to_numpy()[d.row_id.to_numpy()]
    d["created"] = t["created"].to_numpy()[d.row_id.to_numpy()]
    return d


def e8_thresholds(d, c_r):
    """Literal copy of r3_report.rule_thresholds, plus the early/late row_ids."""
    out, split = {}, {}
    for repo, g in d.groupby("test"):
        g = g.sort_values("created")
        h = len(g) // 2
        early, late = g.iloc[:h], g.iloc[h:]
        th = S.choose_threshold(early.prob.to_numpy(), early.y.to_numpy(), c_r)
        rec = {"n_early": len(early), "n_late": len(late), "theta": th}
        if th is not None and len(late):
            rec.update(S.apply_threshold(late.prob.to_numpy(), late.y.to_numpy(), th))
        out[repo] = rec
        split[repo] = (early.row_id.to_numpy(), late.row_id.to_numpy())
    return out, split


def reproduce_e8(t, preds):
    """K5: thresholds of E8 on the factual split against thresholds_by_repo.csv and the thr.*
    / r4.*.trigger_pct keys of the chapter 4 numbers. Returns (report, splits); raises on mismatch."""
    ref_file = pd.read_csv(THR_FILE)
    ref = json.load(open(R4_NUMBERS, encoding="utf-8"))
    rep, bad, splits = {}, [], {}
    for k in ("it", "st", "tr"):
        for c, cr in CR.items():
            thr, split = e8_thresholds(preds[k], cr)
            splits.setdefault(k, split)
            rf = ref_file[(ref_file.k == k) & (ref_file.c == c)].set_index("repo")
            n_diff = 0
            for repo, rec in thr.items():
                x = rf.loc[repo]
                th_ref = None if pd.isna(x.theta) else float(x.theta)
                same = (th_ref is None) == (rec["theta"] is None)
                if same and th_ref is not None:
                    same = abs(th_ref - rec["theta"]) < 1e-12 and int(x.n_fire) == rec["n_fire"] \
                        and abs(float(x.fire_share) - rec["fire_share"]) < 1e-12
                same = same and int(x.n_early) == rec["n_early"] and int(x.n_late) == rec["n_late"]
                n_diff += not same
            reach = [x for x in thr.values() if x["theta"] is not None]
            got = {"n_unreachable": sum(1 for x in thr.values() if x["theta"] is None),
                   "n_below_target": sum(1 for x in reach if x["precision"] == x["precision"]
                                         and x["precision"] < 1 - cr - 1e-12),
                   "prec_lo_median": (float(np.nanmedian([x["prec_lo"] for x in reach]))
                                      if reach else None)}
            exp = ref["thr"][k][c]
            for key, v in got.items():
                e = exp[key]
                ok = (v is None and e is None) or (v is not None and e is not None
                                                   and abs(v - e) < 1e-9)
                if not ok:
                    bad.append(f"thr.{k}.{c}.{key}: got {v}, expected {e}")
            if n_diff:
                bad.append(f"thresholds_by_repo {k} {c}: {n_diff} repositories differ")
            fire_mean = 100 * float(np.mean([x.get("fire_share", 0.0) if x["theta"] is not None
                                             else 0.0 for x in thr.values()]))
            rec_out = {**got, "n_repos_differ_from_file": n_diff, "fire_pct_mean": fire_mean,
                       "expected": exp}
            if k == "tr":
                e = ref["r4"][c]["trigger_pct"]
                rec_out["trigger_pct_expected"] = e
                if abs(fire_mean - e) > 1e-9:
                    bad.append(f"r4.{c}.trigger_pct: got {fire_mean}, expected {e}")
            rep[f"{k}.{c}"] = rec_out
    for repo in splits["it"]:
        e1, l1 = splits["it"][repo]
        e2, l2 = splits["st"][repo]
        if not (np.array_equal(e1, e2) and np.array_equal(l1, l2)):
            bad.append(f"E8 split of {repo} differs between P_it and P_st")
    rep["ok"] = not bad
    rep["mismatches"] = bad
    return rep, splits


def project_arrays(t, preds, repos=None):
    """Per project: arrays ordered by (created, row_id); one row = one whole request."""
    s = t[(t.hait_filter == 1) & (t.in_main == 1)].copy()
    for k in ("it", "st"):
        m = pd.Series(preds[k].prob.to_numpy(), index=preds[k].row_id.to_numpy())
        s["p_" + k] = m.reindex(s.row_id.to_numpy()).to_numpy()
    assert s.p_it.notna().all() and s.p_st.notna().all(), "LORO forecasts must cover the sample"
    s["tm"] = s.t_m_h_D.astype(float).clip(lower=0.0)
    s["nit"] = s.n_it_human.astype(float)
    s["stale"] = s.stale30_D.astype(float)
    s["y_it"] = (s.nit >= 2).astype(float)
    s["y_st"] = s.stale
    assert (s.y_it == s.y_IT__main_D).all() and (s.y_st == s.y_ST__main_D).all()
    s["week"] = ((s.created - pd.Timestamp("2024-01-01", tz="UTC")).dt.days // 7).astype(int)
    s = s.sort_values(["repo", "created", "row_id"], kind="mergesort")
    out = {}
    for repo, g in s.groupby("repo", sort=True):
        if repos is not None and repo not in repos:
            continue
        out[repo] = {c: g[c].to_numpy() for c in ("row_id", "week", "p_it", "p_st", "y_it",
                                                  "y_st", "tm", "nit", "stale")}
    return out


# ── one variant ─────────────────────────────────────────────────────────────

def components(P, hist, nxt):
    """Eff components of the next period with F fixed on the history, and the reductions
    delta of the removal per component (before weighting)."""
    ftm = mid_cdf(P["tm"][hist], P["tm"][nxt])
    fnit = mid_cdf(P["nit"][hist], P["nit"][nxt])
    f1 = float(mid_cdf(P["nit"][hist], np.array([1.0]))[0])
    nit = P["nit"][nxt]
    st = P["stale"][nxt]
    d_nit = np.where(nit >= 2, fnit - f1, 0.0)
    return {"ftm": ftm, "fnit": fnit, "stale": st, "d_nit": d_nit, "d_st": st.copy()}


def measures(comp, fire_it, fire_st):
    """A, A_rand and their parts for R_it, R_st, R_any under every weight vector."""
    n = len(comp["ftm"])
    k_it, k_st = int(fire_it.sum()), int(fire_st.sum())
    out = {"it": {"k": k_it}, "st": {"k": k_st}, "any": {"k": int((fire_it | fire_st).sum())}}
    s_ftm, s_fnit, s_st = comp["ftm"].sum(), comp["fnit"].sum(), comp["stale"].sum()
    g_nit, a_nit = comp["d_nit"][fire_it].sum(), comp["d_nit"].sum()
    g_st, a_st = comp["d_st"][fire_st].sum(), comp["d_st"].sum()
    for wn, (a, b, g) in WTS.items():
        L = a * s_ftm + b * s_fnit + g * s_st
        Ait = b * g_nit / L if L > 0 else 0.0
        Rit = (k_it / n) * b * a_nit / L if L > 0 and n else 0.0
        Ast = g * g_st / L if L > 0 else 0.0
        Rst = (k_st / n) * g * a_st / L if L > 0 and n else 0.0
        out["it"][wn] = (Ait, Rit)
        out["st"][wn] = (Ast, Rst)
        out["any"][wn] = (Ait + Ast, Rit + Rst)
    out["n"] = n
    return out


def run_variant(P, hist, nxt, perm_ctrl=None):
    """Thresholds on the history, decisions and measures on the next period. With perm_ctrl
    (a permutation of the project's rows) also control K2 on shuffled forecasts."""
    comp = components(P, hist, nxt)
    rows = []
    for c, cr in CR.items():
        fire, theta = {}, {}
        for k in ("it", "st"):
            th = choose_threshold(P["p_" + k][hist], P["y_" + k][hist], cr)
            theta[k] = th
            fire[k] = (P["p_" + k][nxt] > th) if th is not None else np.zeros(len(nxt), bool)
        m = measures(comp, fire["it"], fire["st"])
        ctrl = None
        if perm_ctrl is not None:
            fire_c, reach_c = {}, {}
            for k in ("it", "st"):
                ps = P["p_" + k][perm_ctrl]
                th = choose_threshold(ps[hist], P["y_" + k][hist], cr)
                reach_c[k] = th is not None
                kk = int(fire[k].sum())
                top = np.argsort(-ps[nxt], kind="stable")[:kk]
                f = np.zeros(len(nxt), bool)
                f[top] = True
                fire_c[k] = f
            ctrl = (measures(comp, fire_c["it"], fire_c["st"]), reach_c)
        for rule in RULES:
            reach = (theta["it"] is not None or theta["st"] is not None) if rule == "any" \
                else theta[rule] is not None
            r = {"c": c, "rule": rule, "k": m[rule]["k"], "n": m["n"], "reach": bool(reach),
                 "theta": (theta[rule] if rule != "any" else None)}
            for wn in WTS:
                r[f"A_{wn}"], r[f"R_{wn}"] = m[rule][wn]
            if ctrl is not None:
                cm, rc = ctrl
                for wn in WTS:
                    r[f"C_{wn}"] = cm[rule][wn][0]
                r["reach_ctrl"] = (rc["it"] or rc["st"]) if rule == "any" else rc[rule]
            rows.append(r)
    return rows


# ── schemes ─────────────────────────────────────────────────────────────────

def week_blocks(weeks, idx):
    """Row indices (into the project arrays) grouped by calendar week."""
    w = weeks[idx]
    order = np.argsort(w, kind="mergesort")
    w_sorted, i_sorted = w[order], idx[order]
    cuts = np.nonzero(np.diff(w_sorted))[0] + 1
    return np.split(i_sorted, cuts)


def run_project(repo, i_r, P, split_e8, seed, n_shuf, n_block):
    """All variants of one project. split_e8: (early positions, late positions) in P order."""
    rows = []
    n = len(P["row_id"])
    h = n // 2
    rng = np.random.default_rng([seed, STREAM["shuf"], i_r])
    rng_c = np.random.default_rng([seed, STREAM["ctrl"], i_r])
    for v in range(n_shuf):
        perm = rng.permutation(n)
        pc = rng_c.permutation(n)
        for r in run_variant(P, perm[:h], perm[h:], perm_ctrl=pc):
            rows.append({"scheme": "shuf", "v": v, "repo": repo, **r})
    early, late = split_e8
    for r in run_variant(P, early, late):
        rows.append({"scheme": "fact", "v": 0, "repo": repo, **r})
    rng_b = np.random.default_rng([seed, STREAM["block"], i_r])
    be, bl = week_blocks(P["week"], early), week_blocks(P["week"], late)
    for v in range(n_block):
        he = np.concatenate([be[j] for j in rng_b.integers(0, len(be), len(be))])
        nl = np.concatenate([bl[j] for j in rng_b.integers(0, len(bl), len(bl))])
        for r in run_variant(P, he, nl):
            rows.append({"scheme": "block", "v": v, "repo": repo, **r})
    return rows


# ── summaries ───────────────────────────────────────────────────────────────

def pct(x, q):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(np.percentile(x, q)) if len(x) else None


def fnum(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def summarise(V, repos, seed):
    """Flat e9.* numbers and the by-repository table."""
    nums, by_repo = {}, []
    R = len(repos)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, R, size=(B, R))          # one resampling for all measures (paired)
    for scheme in ("shuf", "block", "fact"):
        Vs = V[V.scheme == scheme]
        if Vs.empty:
            continue
        nv = int(Vs.v.max()) + 1
        for rule in RULES:
            for c in CR:
                sub = Vs[(Vs.rule == rule) & (Vs.c == c)].sort_values(["v", "repo"])
                assert len(sub) == nv * R
                reach = sub.reach.to_numpy().reshape(nv, R)
                fire = (sub.k / sub.n).to_numpy().reshape(nv, R)
                pre = f"e9.{scheme}.{rule}.{c}"
                n_unreach = (~reach).sum(axis=1)
                nums[f"{pre}.n_unreach_med"] = float(np.median(n_unreach))
                nums[f"{pre}.n_unreach_all"] = int((~reach).all(axis=0).sum())
                nums[f"{pre}.n_unreach_any"] = int((~reach).any(axis=0).sum())
                nums[f"{pre}.fire_pct"] = 100 * float(fire.mean())
                for wn in WTS:
                    A = sub[f"A_{wn}"].to_numpy().reshape(nv, R)
                    Rr = sub[f"R_{wn}"].to_numpy().reshape(nv, R)
                    p = f"{pre}.{wn}"
                    a_r, r_r = A.mean(axis=0), Rr.mean(axis=0)
                    mA, mR = a_r[idx].mean(axis=1), r_r[idx].mean(axis=1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        lb = np.where(mR > 0, mA / mR, np.nan)
                    nums[f"{p}.A_pct"] = 100 * float(a_r.mean())
                    nums[f"{p}.A_lo_pct"] = 100 * pct(mA, 2.5)
                    nums[f"{p}.A_hi_pct"] = 100 * pct(mA, 97.5)
                    nums[f"{p}.Arand_pct"] = 100 * float(r_r.mean())
                    nums[f"{p}.Arand_lo_pct"] = 100 * pct(mR, 2.5)
                    nums[f"{p}.Arand_hi_pct"] = 100 * pct(mR, 97.5)
                    d = mA - mR
                    nums[f"{p}.diff_pct"] = 100 * float(a_r.mean() - r_r.mean())
                    nums[f"{p}.diff_lo_pct"] = 100 * pct(d, 2.5)
                    nums[f"{p}.diff_hi_pct"] = 100 * pct(d, 97.5)
                    nums[f"{p}.lift"] = fnum(a_r.mean() / r_r.mean()) if r_r.mean() > 0 else None
                    nums[f"{p}.lift_lo"] = pct(lb, 2.5)
                    nums[f"{p}.lift_hi"] = pct(lb, 97.5)
                    nums[f"{p}.lift_boot_undefined"] = int(np.isnan(lb).sum())
                    if nv > 1:
                        Av, Rv = A.mean(axis=1), Rr.mean(axis=1)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            Lv = np.where(Rv > 0, Av / Rv, np.nan)
                        nums[f"{p}.var.A_med_pct"] = 100 * float(np.median(Av))
                        nums[f"{p}.var.A_p025_pct"] = 100 * pct(Av, 2.5)
                        nums[f"{p}.var.A_p975_pct"] = 100 * pct(Av, 97.5)
                        nums[f"{p}.var.Arand_med_pct"] = 100 * float(np.median(Rv))
                        nums[f"{p}.var.lift_med"] = pct(Lv, 50)
                        nums[f"{p}.var.lift_p025"] = pct(Lv, 2.5)
                        nums[f"{p}.var.lift_p975"] = pct(Lv, 97.5)
                        nums[f"{p}.var.lift_undefined"] = int(np.isnan(Lv).sum())
                    for j, repo in enumerate(repos):
                        by_repo.append({"scheme": scheme, "repo": repo, "rule": rule, "c": c,
                                        "w": wn, "A_mean": a_r[j], "Arand_mean": r_r[j],
                                        "lift": (a_r[j] / r_r[j]) if r_r[j] > 0 else np.nan,
                                        "fire_share_mean": fire[:, j].mean(),
                                        "n_variants": nv,
                                        "n_variants_unreachable": int((~reach[:, j]).sum())})
                    if scheme == "shuf" and "C_w3" in sub:
                        C = sub[f"C_{wn}"].to_numpy().reshape(nv, R)
                        Cv, Rv = C.mean(axis=1), Rr.mean(axis=1)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            Lc = np.where(Rv > 0, Cv / Rv, np.nan)
                        q = f"e9.ctrl.{scheme}.{rule}.{c}.{wn}"
                        nums[f"{q}.lift_var_med"] = pct(Lc, 50)
                        nums[f"{q}.lift_var_p025"] = pct(Lc, 2.5)
                        nums[f"{q}.lift_var_p975"] = pct(Lc, 97.5)
                        nums[f"{q}.lift_var_defined"] = int((~np.isnan(Lc)).sum())
                        c_r = C.mean(axis=0)
                        nums[f"{q}.lift"] = fnum(c_r.mean() / r_r.mean()) if r_r.mean() > 0 else None
                if scheme == "shuf" and "reach_ctrl" in sub:
                    rc = sub.reach_ctrl.to_numpy().astype(bool).reshape(nv, R)
                    nums[f"e9.ctrl.{scheme}.{rule}.{c}.n_reach_med"] = float(np.median(rc.sum(axis=1)))
                    nums[f"e9.ctrl.{scheme}.{rule}.{c}.n_reach_max"] = int(rc.sum(axis=1).max())
    by_repo = pd.DataFrame(by_repo)
    # descriptive per-project spread (no criterion): where the rule acted at least once
    for (scheme, rule, c, wn), g in by_repo.groupby(["scheme", "rule", "c", "w"]):
        if wn not in ("w3", "nit"):
            continue
        acting = g[g.A_mean > 0]
        p = f"e9.{scheme}.{rule}.{c}.{wn}.repo"
        nums[f"{p}.n_acting"] = int(len(acting))
        if len(acting):
            nums[f"{p}.n_lift_gt1"] = int((acting.lift > 1).sum())
            nums[f"{p}.lift_min"] = float(acting.lift.min())
            nums[f"{p}.lift_med"] = float(acting.lift.median())
            nums[f"{p}.lift_max"] = float(acting.lift.max())
            top = acting.loc[acting.A_mean.idxmax()]
            nums[f"{p}.A_max_pct"] = 100 * float(top.A_mean)
            nums[f"{p}.A_max_name"] = top.repo.split("/", 1)[1]
            nums[f"{p}.A_max_fire_pct"] = 100 * float(top.fire_share_mean)
    return nums, by_repo


def k2_verdict(nums):
    """K2(b): the variant interval of the control lift contains 1 wherever it is defined."""
    out = {}
    for key, v in nums.items():
        if key.startswith("e9.ctrl.shuf.") and key.endswith(".lift_var_p025") and v is not None:
            hi = nums[key.replace("p025", "p975")]
            base = key[:-len(".lift_var_p025")]
            out[base] = {"p025": v, "p975": hi, "contains_1": v <= 1 <= hi,
                         "n_variants_defined": nums[base + ".lift_var_defined"]}
    return out


# ── main ────────────────────────────────────────────────────────────────────

def git_head():
    try:
        return subprocess.check_output(["git", "-C", REV, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="seed 1, 3 projects, 20/10 variants")
    args = ap.parse_args(argv)
    t0 = time.time()
    seed = SEED_DEV if args.dev else SEED_MAIN
    n_shuf, n_block = (N_SHUF_DEV, N_BLOCK_DEV) if args.dev else (N_SHUF, N_BLOCK)
    out = os.path.join(OUT, "dev") if args.dev else OUT
    os.makedirs(out, exist_ok=True)

    t = load_table()
    preds = {k: load_preds(t, tg) for k, tg in (("it", "IT"), ("st", "ST"), ("tr", "TFR"))}
    k5, splits = reproduce_e8(t, preds)
    print("K5 E8 reproduction:", "OK" if k5["ok"] else k5["mismatches"])
    if not k5["ok"]:
        json.dump({"k5_e8_reproduction": k5}, open(os.path.join(out, "checks.json"), "w"),
                  indent=1, ensure_ascii=False)
        sys.exit("E8 numbers not reproduced: stop (see checks.json)")

    all_repos = sorted(t.loc[(t.hait_filter == 1) & (t.in_main == 1), "repo"].unique())
    assert len(all_repos) == 40
    repos = sorted(DEV_REPOS) if args.dev else all_repos
    PA = project_arrays(t, preds, set(repos))

    rows = []
    for repo in repos:
        i_r = all_repos.index(repo)
        P = PA[repo]
        pos = pd.Series(np.arange(len(P["row_id"])), index=P["row_id"])
        e, l = splits["it"][repo]
        split = (pos.reindex(e).to_numpy(), pos.reindex(l).to_numpy())
        rows.extend(run_project(repo, i_r, P, split, seed, n_shuf, n_block))
        print(f"  {repo}: {time.time() - t0:.0f} s", flush=True)
    V = pd.DataFrame(rows)

    # K1: null action gives A = 0 exactly
    P0 = PA[repos[0]]
    n0 = len(P0["row_id"])
    comp = components(P0, np.arange(n0 // 2), np.arange(n0 // 2, n0))
    z = np.zeros(n0 - n0 // 2, bool)
    m0 = measures(comp, z, z)
    k1 = {"A_all_zero": all(m0[r][w][0] == 0.0 and m0[r][w][1] == 0.0 for r in RULES for w in WTS),
          "A_zero_where_unreachable": bool((V.loc[~V.reach, [f"A_{w}" for w in WTS]] == 0).all().all())}
    k1["ok"] = all(k1.values())
    # K3: input row order (whole table shuffled) gives identical variants for the first projects
    t_sh = t.sample(frac=1.0, random_state=np.random.default_rng([seed, STREAM["order"]])
                    .integers(0, 2**31 - 1)).reset_index(drop=True)   # row_id column kept
    PA2 = project_arrays(t_sh, preds, set(repos[:3]))
    k3 = {}
    for repo in repos[:3]:
        same_arrays = all(np.array_equal(PA[repo][c], PA2[repo][c]) for c in PA[repo])
        i_r = all_repos.index(repo)
        pos = pd.Series(np.arange(len(PA2[repo]["row_id"])), index=PA2[repo]["row_id"])
        e, l = splits["it"][repo]
        r2 = pd.DataFrame(run_project(repo, i_r, PA2[repo], (pos.reindex(e).to_numpy(),
                                                             pos.reindex(l).to_numpy()),
                                      seed, 3, 2))
        r1 = V[(V.repo == repo) & (((V.scheme == "shuf") & (V.v < 3)) | (V.scheme == "fact")
                                   | ((V.scheme == "block") & (V.v < 2)))]
        key = ["scheme", "v", "c", "rule"]
        a = r1.sort_values(key).reset_index(drop=True)[[f"A_{w}" for w in WTS] + ["k", "n"]]
        b = r2.sort_values(key).reset_index(drop=True)[[f"A_{w}" for w in WTS] + ["k", "n"]]
        k3[repo] = {"arrays_equal": bool(same_arrays), "variants_equal": bool(a.equals(b))}
    k3_ok = all(x["arrays_equal"] and x["variants_equal"] for x in k3.values())

    nums, by_repo = summarise(V, repos, seed)
    nums["e9.n_variants"] = n_shuf
    nums["e9.n_variants_block"] = n_block
    nums["e9.n_repos"] = len(repos)
    nums["e9.seed"] = seed
    nums["e9.n_boot"] = B
    k2 = k2_verdict(nums)
    k2_ok = all(x["contains_1"] for x in k2.values())

    run = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "dev" if args.dev else "main",
           "seed": seed, "repos": repos if args.dev else len(repos), "n_shuf": n_shuf,
           "n_block": n_block, "git": git_head(), "seconds": round(time.time() - t0, 1),
           "k1": k1["ok"], "k2": k2_ok, "k3": k3_ok, "k5": k5["ok"],
           "summary": {k: nums.get(k) for k in (
               "e9.shuf.it.c02.w3.A_pct", "e9.shuf.it.c02.w3.Arand_pct",
               "e9.shuf.it.c02.w3.lift", "e9.shuf.it.c02.w3.lift_lo",
               "e9.shuf.it.c02.w3.lift_hi", "e9.shuf.st.c02.w3.A_pct")}}
    with open(os.path.join(OUT, "runs.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(run, ensure_ascii=False) + "\n")
    runs = [json.loads(x) for x in open(os.path.join(OUT, "runs.jsonl"), encoding="utf-8")]

    checks = {"k5_e8_reproduction": k5, "k1_null_action": k1, "k2_shuffled_forecasts": k2,
              "k2_ok": k2_ok, "k3_row_order": k3, "k3_ok": k3_ok, "runs": runs}
    json.dump(checks, open(os.path.join(out, "checks.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    json.dump({k: nums[k] for k in sorted(nums)}, open(os.path.join(out, "numbers.json"), "w",
                                                      encoding="utf-8"), indent=1, ensure_ascii=False)
    by_repo.to_csv(os.path.join(out, "by_repo.csv"), index=False)
    V.to_csv(os.path.join(out, "variants.csv"), index=False)
    print(json.dumps(run, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
