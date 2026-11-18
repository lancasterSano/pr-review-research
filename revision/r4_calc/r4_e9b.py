#!/usr/bin/env python3
"""Chapter 4, E9 extension added AFTER the E9 results were seen; its variants, seeds and
measures were fixed in the author's notes before this calculation.

  E9b  rule R4 "availability priority" on the 12 ranking repositories (ranking test PRs that are
       in the r3_b40d main sample): removal t_m' = max(t_m - k * dt_r, 0), dt_r = expected response
       time of the first candidate by relevance minus after re-ranking (per_pr.csv), k = 1 (shift
       model, main upper bound) and the empirical k; R1+R2+R4 together; re-ranking on all requests.
  R3grid  rule R3 on the c grid 0.2 ... 0.95 (same generator as E9): where it starts to act, coverage
       of staleness (recall), precision of escalations, G_R3. No criterion.
  k    empirical k from the earlier half of the main sample (OLS on logs with repository fixed
       effects and t_0 controls; median regression in hours; shift-model check on t_m - t_r).

    PY=python   # interpreter with requirements.txt
    $PY revision/r4_calc/r4_e9b.py --dev    # seed 1, 3 ranking repositories, few variants
    $PY revision/r4_calc/r4_e9b.py          # main -> revision/output/r4/e9_eff/*_e9b.*
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r4_e9_eff as E  # noqa: E402
from r4_e9_baseline import mid_cdf  # noqa: E402

PER_PR = os.path.join(E.REV, "output", "r4", "no_overlap_thr", "per_pr.csv")
MAIN12 = os.path.join(E.REV, "output", "r4", "no_overlap_thr", "main12_predictions.csv")
DEV_R4 = ["angular/angular", "django/django", "vercel/next.js"]
CR3 = {"c020": 0.2, "c030": 0.3, "c050": 0.5, "c070": 0.7, "c080": 0.8, "c090": 0.9, "c095": 0.95}
W4 = {"w3": (1 / 3, 1 / 3, 1 / 3), "tm": (1.0, 0.0, 0.0)}
STREAM_R4, STREAM_R4_CTRL = 5, 6
CTRL_COLS = ["lines_added_t0", "lines_deleted_t0", "commit_files_t0", "num_commits_t0",
             "author_pr_count", "author_tenure_days"]
B_OLS, B_QR = 2000, 200


# ── shared summaries ────────────────────────────────────────────────────────

def pct(x, q):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(np.percentile(x, q)) if len(x) else None


def cluster_summary(A, R, idx, prefix, nums, var=True):
    """A, R: (n_variants, n_repos). Mean over repos; cluster bootstrap over repos (idx) of the
    per-repo means over variants; ratio of means; variant percentiles of the corpus values."""
    a_r, r_r = A.mean(axis=0), R.mean(axis=0)
    mA, mR = a_r[idx].mean(axis=1), r_r[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lb = np.where(mR > 0, mA / mR, np.nan)
    nums[f"{prefix}.G_pct"] = 100 * float(a_r.mean())
    nums[f"{prefix}.G_lo_pct"] = 100 * pct(mA, 2.5)
    nums[f"{prefix}.G_hi_pct"] = 100 * pct(mA, 97.5)
    nums[f"{prefix}.Grand_pct"] = 100 * float(r_r.mean())
    nums[f"{prefix}.Grand_lo_pct"] = 100 * pct(mR, 2.5)
    nums[f"{prefix}.Grand_hi_pct"] = 100 * pct(mR, 97.5)
    nums[f"{prefix}.lift"] = float(a_r.mean() / r_r.mean()) if r_r.mean() > 0 else None
    nums[f"{prefix}.lift_lo"] = pct(lb, 2.5)
    nums[f"{prefix}.lift_hi"] = pct(lb, 97.5)
    if var and A.shape[0] > 1:
        Av, Rv = A.mean(axis=1), R.mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            Lv = np.where(Rv > 0, Av / Rv, np.nan)
        nums[f"{prefix}.var.G_p025_pct"] = 100 * pct(Av, 2.5)
        nums[f"{prefix}.var.G_p975_pct"] = 100 * pct(Av, 97.5)
        nums[f"{prefix}.var.lift_med"] = pct(Lv, 50)
        nums[f"{prefix}.var.lift_p025"] = pct(Lv, 2.5)
        nums[f"{prefix}.var.lift_p975"] = pct(Lv, 97.5)
        nums[f"{prefix}.var.lift_min"] = float(np.nanmin(Lv)) if np.isfinite(Lv).any() else None
    return a_r, r_r


# ── R3grid: R3 on the c grid ───────────────────────────────────────────────────

def r3_grid(PA, all_repos, repos, seed, n_var):
    rows = []
    for repo in repos:
        i_r = all_repos.index(repo)
        P = PA[repo]
        n = len(P["row_id"])
        h = n // 2
        rng = np.random.default_rng([seed, E.STREAM["shuf"], i_r])   # same permutations as E9
        for v in range(n_var):
            perm = rng.permutation(n)
            hist, nxt = perm[:h], perm[h:]
            comp = E.components(P, hist, nxt)
            st = P["stale"][nxt]
            z = np.zeros(len(nxt), bool)
            for c, cr in CR3.items():
                th = E.choose_threshold(P["p_st"][hist], P["y_st"][hist], cr)
                fire = (P["p_st"][nxt] > th) if th is not None else z
                m = E.measures(comp, z, fire)["st"]
                rows.append({"v": v, "repo": repo, "c": c, "reach": th is not None,
                             "k": int(fire.sum()), "n": len(nxt), "n_stale": int(st.sum()),
                             "tp": int((fire & (st == 1)).sum()),
                             "A_w3": m["w3"][0], "R_w3": m["w3"][1],
                             "A_st": m["st"][0], "R_st": m["st"][1]})
    return pd.DataFrame(rows)


def r3_summary(V, repos, idx, nums):
    R = len(repos)
    nv = int(V.v.max()) + 1
    min_c = None
    for c, cr in CR3.items():
        s = V[V.c == c].sort_values(["v", "repo"])
        reach = s.reach.to_numpy().reshape(nv, R)
        p = f"e9.r3grid.{c}"
        nums[f"{p}.c"] = cr
        nums[f"{p}.n_reach_med"] = float(np.median(reach.sum(axis=1)))
        nums[f"{p}.fire_pct"] = 100 * float((s.k / s.n).to_numpy().reshape(nv, R).mean())
        g = s.groupby("v")[["tp", "n_stale", "k"]].sum()
        rec = g.tp / g.n_stale
        prec = (g.tp / g.k).where(g.k > 0)
        for name, x in (("recall", rec), ("prec", prec)):
            nums[f"{p}.{name}_med_pct"] = 100 * pct(x, 50) if x.notna().any() else None
            nums[f"{p}.{name}_p025_pct"] = 100 * pct(x, 2.5) if x.notna().any() else None
            nums[f"{p}.{name}_p975_pct"] = 100 * pct(x, 97.5) if x.notna().any() else None
        nums[f"{p}.n_variants_prec_defined"] = int(prec.notna().sum())
        pm = nums[f"{p}.prec_med_pct"]
        nums[f"{p}.unneeded_med_pct"] = None if pm is None else 100 - pm
        nums[f"{p}.unneeded_per10"] = None if pm is None else 10 * (1 - pm / 100)
        for w in ("w3", "st"):
            A = s[f"A_{w}"].to_numpy().reshape(nv, R)
            Rr = s[f"R_{w}"].to_numpy().reshape(nv, R)
            cluster_summary(A, Rr, idx, f"{p}.{w}", nums)
        if min_c is None and nums[f"{p}.n_reach_med"] >= R / 2:
            min_c = cr
    nums["e9.r3grid.min_c_half"] = min_c
    return nums


# ── E9b: R4 data ────────────────────────────────────────────────────────────

def r4_arrays(t, preds, all_repos):
    """Ranking test PRs that are in the main sample, per repository, ordered by (created, row_id)."""
    pp = pd.read_csv(PER_PR)
    s = t[(t.hait_filter == 1) & (t.in_main == 1)]
    j = pp.merge(s, left_on=["repo_full", "pr"], right_on=["repo", "pr_number"], how="inner",
                 suffixes=("_pp", ""))
    m12 = pd.read_csv(MAIN12).set_index("row_id")["prob"]
    chk = {"n_per_pr": int(len(pp)), "n_in_main": int(len(j)),
           "p_tr_equals_main12": bool(np.allclose(j.p_tr.to_numpy(),
                                                  m12.reindex(j.row_id).to_numpy(), atol=1e-12))}
    pit = pd.Series(preds["it"].prob.to_numpy(), index=preds["it"].row_id.to_numpy())
    j["p_it"] = pit.reindex(j.row_id).to_numpy()
    dtr = (j.top1_lat_rel - j.top1_lat_lat).fillna(0.0).clip(lower=0.0)
    j["dtr"] = dtr
    j["tm"] = j.t_m_h_D.astype(float).clip(lower=0.0)
    j["nit"] = j.n_it_human.astype(float)
    j["stale"] = j.stale30_D.astype(float)
    j["y_it"] = (j.nit >= 2).astype(float)
    j["y_tr"] = j.y_TFR__main_D.astype(float)
    j = j.sort_values(["repo", "created", "row_id"], kind="mergesort")
    out = {}
    for repo, g in j.groupby("repo", sort=True):
        out[repo] = {c: g[c].to_numpy() for c in ("row_id", "p_tr", "y_tr", "dtr", "p_it", "y_it",
                                                  "tm", "nit", "stale")}
    chk["n_by_repo"] = {r: int(len(v["row_id"])) for r, v in out.items()}
    # K5b: known numbers of 4.6.2 / E7 on the ranking test PRs (file values, all 11 341)
    f = pp[pp.fire_c02]
    chk["repro"] = {"fire_c02_pct": 100 * float(pp.fire_c02.mean()),
                    "fired_med_rel_h": float(f.top1_lat_rel.median()),
                    "fired_med_lat_h": float(f.top1_lat_lat.median()),
                    "all_med_rel_h": float(pp.top1_lat_rel.median()),
                    "all_med_lat_h": float(pp.top1_lat_lat.median())}
    return out, chk


def choose_tr(P, hist, cr):
    ok = ~np.isnan(P["y_tr"][hist])
    return E.choose_threshold(P["p_tr"][hist][ok], P["y_tr"][hist][ok], cr)


def r4_variant(P, hist, nxt, kvals, perm_ctrl=None):
    """Rows per (c, k): parts of G for R4, R1+R2 and the re-ranking on all requests."""
    comp = E.components(P, hist, nxt)
    ftm, fnit, st, d_nit = comp["ftm"], comp["fnit"], comp["stale"], comp["d_nit"]
    n = len(nxt)
    d_tm = {}
    for kn, kv in kvals.items():
        tm2 = np.maximum(P["tm"][nxt] - max(kv, 0.0) * P["dtr"][nxt], 0.0)
        d_tm[kn] = ftm - mid_cdf(P["tm"][hist], tm2)
    rows = []
    for c, cr in E.CR.items():
        th_tr, th_it = choose_tr(P, hist, cr), E.choose_threshold(P["p_it"][hist], P["y_it"][hist], cr)
        f_tr = (P["p_tr"][nxt] > th_tr) if th_tr is not None else np.zeros(n, bool)
        f_it = (P["p_it"][nxt] > th_it) if th_it is not None else np.zeros(n, bool)
        k_tr, k_it = int(f_tr.sum()), int(f_it.sum())
        f_c = None
        if perm_ctrl is not None:
            ps = P["p_tr"][perm_ctrl][nxt]
            f_c = np.zeros(n, bool)
            f_c[np.argsort(-ps, kind="stable")[:k_tr]] = True
        for kn in kvals:
            r = {"c": c, "kname": kn, "k_tr": k_tr, "k_it": k_it, "n": n,
                 "reach_tr": th_tr is not None, "reach_it": th_it is not None,
                 "k_any": int((f_tr | f_it).sum())}
            for wn, (a, b, g) in W4.items():
                L = a * ftm.sum() + b * fnit.sum() + g * st.sum()
                G4 = a * d_tm[kn][f_tr].sum() / L
                Gall = a * d_tm[kn].sum() / L
                R4 = (k_tr / n) * Gall
                Git = b * d_nit[f_it].sum() / L
                Rit = (k_it / n) * b * d_nit.sum() / L
                r.update({f"G4_{wn}": G4, "R4_" + wn: R4, "Gall_" + wn: Gall,
                          "Git_" + wn: Git, "Rit_" + wn: Rit,
                          "Gc_" + wn: G4 + Git, "Rc_" + wn: R4 + Rit})
                if f_c is not None:
                    r["G4ctrl_" + wn] = a * d_tm[kn][f_c].sum() / L
            rows.append(r)
    return rows


def r4_run(PA4, all_repos, seed, n_var, kvals, ctrl=True):
    rows = []
    for repo, P in PA4.items():
        i_r = all_repos.index(repo)
        n = len(P["row_id"])
        h = n // 2
        rng = np.random.default_rng([seed, STREAM_R4, i_r])
        rng_c = np.random.default_rng([seed, STREAM_R4_CTRL, i_r])
        for v in range(n_var):
            perm = rng.permutation(n)
            pc = rng_c.permutation(n)
            for r in r4_variant(P, perm[:h], perm[h:], kvals, pc if ctrl else None):
                rows.append({"v": v, "repo": repo, **r})
    return pd.DataFrame(rows)


def r4_summary(V, repos, idx, nums):
    R = len(repos)
    nv = int(V.v.max()) + 1
    for kn in V.kname.unique():
        for c in E.CR:
            s = V[(V.kname == kn) & (V.c == c)].sort_values(["v", "repo"])
            p = f"e9.r4.{kn}.{c}"
            reach = s.reach_tr.to_numpy().reshape(nv, R)
            nums[f"{p}.n_unreach_med"] = float(np.median((~reach).sum(axis=1)))
            nums[f"{p}.fire_pct"] = 100 * float((s.k_tr / s.n).to_numpy().reshape(nv, R).mean())
            nums[f"{p}.fire_it_pct"] = 100 * float((s.k_it / s.n).to_numpy().reshape(nv, R).mean())
            nums[f"{p}.fire_any_pct"] = 100 * float((s.k_any / s.n).to_numpy().reshape(nv, R).mean())
            reach_it = s.reach_it.to_numpy().reshape(nv, R)
            nums[f"{p}.n_unreach_it_med"] = float(np.median((~reach_it).sum(axis=1)))
            nums[f"{p}.n_unreach_any_med"] = float(np.median((~(reach | reach_it)).sum(axis=1)))
            for wn in W4:
                M = lambda col: s[col].to_numpy().reshape(nv, R)  # noqa: E731
                cluster_summary(M("G4_" + wn), M("R4_" + wn), idx, f"{p}.{wn}", nums)
                cluster_summary(M("Gc_" + wn), M("Rc_" + wn), idx, f"{p}.{wn}.comb", nums)
                cluster_summary(M("Git_" + wn), M("Rit_" + wn), idx, f"{p}.{wn}.it", nums)
                ga = M("Gall_" + wn).mean(axis=0)
                mg = ga[idx].mean(axis=1)
                nums[f"{p}.{wn}.all.G_pct"] = 100 * float(ga.mean())
                nums[f"{p}.{wn}.all.G_lo_pct"] = 100 * pct(mg, 2.5)
                nums[f"{p}.{wn}.all.G_hi_pct"] = 100 * pct(mg, 97.5)
                if "G4ctrl_" + wn in s:
                    C, Rr = M("G4ctrl_" + wn).mean(axis=1), M("R4_" + wn).mean(axis=1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        Lc = np.where(Rr > 0, C / Rr, np.nan)
                    q = f"e9.r4.ctrl.{kn}.{c}.{wn}"
                    nums[f"{q}.lift_var_p025"] = pct(Lc, 2.5)
                    nums[f"{q}.lift_var_p975"] = pct(Lc, 97.5)
                    nums[f"{q}.lift_var_defined"] = int(np.isfinite(Lc).sum())
    return nums


# ── empirical k ─────────────────────────────────────────────────────────────

def k_data(t, splits, repos):
    s = t[(t.hait_filter == 1) & (t.in_main == 1) & t.repo.isin(repos)].copy()
    early = np.concatenate([splits[r][0] for r in repos])
    s = s[s.row_id.isin(early) & s.t_r_h_D.notna()]
    s["tm"] = s.t_m_h_D.astype(float).clip(lower=0.0)
    s["tr"] = s.t_r_h_D.astype(float).clip(lower=0.0)
    s["size"] = s.lines_added_t0.fillna(np.nan) + s.lines_deleted_t0
    X = pd.DataFrame({"size": s["size"], "files": s.commit_files_t0, "commits": s.num_commits_t0,
                      "prs": s.author_pr_count, "tenure": s.author_tenure_days}).astype(float)
    X = X.fillna(X.median())
    s[["x_" + c for c in X.columns]] = np.log1p(X.clip(lower=0.0)).to_numpy()
    return s.reset_index(drop=True)


def design(s, groups, var, logs=True):
    xs = [c for c in s.columns if c.startswith("x_")]
    g = pd.Categorical(groups)
    D = np.eye(len(g.categories))[g.codes]
    v = np.log1p(s[var].to_numpy()) if logs else s[var].to_numpy()
    return np.column_stack([D, v, s[xs].to_numpy()]), D.shape[1], g.codes


def ols_fit(s, groups, yvar, xvar):
    X, nd, codes = design(s, groups, xvar, logs=True)
    y = np.log1p(s[yvar].to_numpy())
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    r2 = 1 - res.var() / y.var()
    yw = y - pd.Series(y).groupby(codes).transform("mean").to_numpy()
    r2w = 1 - res.var() / yw.var()
    return float(beta[nd]), float(r2), float(r2w)


def qr_fit(s, groups, yvar, xvar):
    from sklearn.linear_model import QuantileRegressor
    X, nd, _ = design(s, groups, xvar, logs=False)
    q = QuantileRegressor(quantile=0.5, alpha=0.0, fit_intercept=False, solver="highs")
    q.fit(X, s[yvar].to_numpy())
    return float(q.coef_[nd])


_BOOT = None
N_PROC = max(1, (os.cpu_count() or 2) - 2)


def _boot_frame(pick):
    s, repos = _BOOT
    return pd.concat([s[s.repo == repos[j]].assign(cl=f"{k}") for k, j in enumerate(pick)],
                     ignore_index=True)


def _qr_boot(pick):
    d = _boot_frame(pick)
    return qr_fit(d, d.cl, "tm", "tr"), 1 + qr_fit(d, d.cl, "s_after", "tr")


def k_estimates(s, seed, b_ols, b_qr, quantile=True):
    out = {}
    s = s.copy()
    s["s_after"] = (s.tm - s.tr).clip(lower=0.0)
    med_tm, med_tr = float(s.tm.median()), float(s.tr.median())
    conv = (1 + med_tm) / (1 + med_tr)
    b, r2, r2w = ols_fit(s, s.repo, "tm", "tr")
    c, _, _ = ols_fit(s, s.repo, "s_after", "tr")
    out.update({"n_rows": int(len(s)), "n_repos": int(s.repo.nunique()), "med_tm_h": med_tm,
                "med_tr_h": med_tr, "b": b, "r2": r2, "r2_within": r2w, "k1": b * conv,
                "c_shift": c})
    if quantile:
        out["k2"] = qr_fit(s, s.repo, "tm", "tr")
        out["k3"] = 1 + qr_fit(s, s.repo, "s_after", "tr")
    repos = sorted(s.repo.unique())
    rng = np.random.default_rng(seed)
    picks = [rng.integers(0, len(repos), len(repos))
             for _ in range(max(b_ols, b_qr if quantile else 0))]   # same draw sequence for both
    global _BOOT
    _BOOT = (s, repos)
    bb, cc = [], []
    for pick in picks[:b_ols]:
        d = _boot_frame(pick)
        bb.append(ols_fit(d, d.cl, "tm", "tr")[0])
        cc.append(ols_fit(d, d.cl, "s_after", "tr")[0])
    kq2, kq3 = [], []
    if quantile and b_qr:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(N_PROC) as pool:   # median regressions in parallel
            res = pool.map(_qr_boot, picks[:b_qr])
        kq2 = [r[0] for r in res]
        kq3 = [r[1] for r in res]
    out["b_lo"], out["b_hi"] = pct(bb, 2.5), pct(bb, 97.5)
    out["k1_lo"], out["k1_hi"] = out["b_lo"] * conv, out["b_hi"] * conv
    out["c_lo"], out["c_hi"] = pct(cc, 2.5), pct(cc, 97.5)
    out["shift_supported"] = bool(out["c_lo"] <= 0 <= out["c_hi"])
    if quantile:
        out["k2_lo"], out["k2_hi"] = pct(kq2, 2.5), pct(kq2, 97.5)
        out["k3_lo"], out["k3_hi"] = pct(kq3, 2.5), pct(kq3, 97.5)
        src = "k2" if r2w < 0.10 else "k1"
        out["kemp_source"] = src
        out["kemp"], out["kemp_lo"], out["kemp_hi"] = out[src], out[src + "_lo"], out[src + "_hi"]
    return out


# ── summaries again from the saved variants ────────────────────────────────

def resummarise(dev=False):
    """Recompute r4_summary and r3_summary from variants_e9b_*.csv with the same bootstrap
    indices; every existing key must keep its value; new keys are added. No model is fitted."""
    t0 = time.time()
    seed = E.SEED_DEV if dev else E.SEED_MAIN
    out = os.path.join(E.OUT, "dev") if dev else E.OUT
    old = json.load(open(os.path.join(out, "numbers_e9b.json"), encoding="utf-8"))
    V4 = pd.read_csv(os.path.join(out, "variants_e9b_r4.csv"))
    V3 = pd.read_csv(os.path.join(out, "variants_e9b_r3.csv"))
    repos4, repos3 = sorted(V4.repo.unique()), sorted(V3.repo.unique())
    nums = {}
    idx4 = np.random.default_rng(seed).integers(0, len(repos4), size=(E.B, len(repos4)))
    r4_summary(V4, repos4, idx4, nums)
    idx3 = np.random.default_rng(seed).integers(0, len(repos3), size=(E.B, len(repos3)))
    r3_summary(V3, repos3, idx3, nums)
    same = lambda a, b: (a is None and b is None) or (a is not None and b is not None and (  # noqa: E731
        a == b or (isinstance(a, float) and abs(a - b) <= 1e-12 * max(1.0, abs(b)))))
    diff = [k for k in nums if k in old and not same(nums[k], old[k])]
    assert not diff, diff[:10]
    new = sorted(k for k in nums if k not in old)
    merged = {**old, **nums}
    json.dump({k: merged[k] for k in sorted(merged)},
              open(os.path.join(out, "numbers_e9b.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    run = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "resummarise_e9b" + ("_dev" if dev else ""),
           "seed": seed, "git": E.git_head(), "seconds": round(time.time() - t0, 1),
           "old_keys_unchanged": True, "n_new_keys": len(new), "new_key_examples": new[:6]}
    with open(os.path.join(E.OUT, "runs.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(run, ensure_ascii=False) + "\n")
    ch = json.load(open(os.path.join(out, "checks_e9b.json"), encoding="utf-8"))
    ch["runs"] = [json.loads(x) for x in open(os.path.join(E.OUT, "runs.jsonl"), encoding="utf-8")]
    json.dump(ch, open(os.path.join(out, "checks_e9b.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print(json.dumps(run, ensure_ascii=False))
    return 0


# ── main ────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--resummarise", action="store_true",
                    help="recompute the summaries from the saved variants (no fitting) and add keys")
    args = ap.parse_args(argv)
    if args.resummarise:
        return resummarise(args.dev)
    t0 = time.time()
    seed = E.SEED_DEV if args.dev else E.SEED_MAIN
    n_var = 10 if args.dev else E.N_SHUF
    b_ols, b_qr = (50, 4) if args.dev else (B_OLS, B_QR)
    out = os.path.join(E.OUT, "dev") if args.dev else E.OUT
    os.makedirs(out, exist_ok=True)

    cols = ["repo", "pr_number", "in_main", "hait_filter", "created_at", "t_m_h_D", "t_r_h_D",
            "n_it_human", "stale30_D", "y_IT__main_D", "y_ST__main_D", "y_TFR__main_D"] + CTRL_COLS
    t = pd.read_csv(E.TABLE, usecols=cols, low_memory=False)
    t["row_id"] = np.arange(len(t))
    t["created"] = pd.to_datetime(t["created_at"], utc=True)
    preds = {k: E.load_preds(t, tg) for k, tg in (("it", "IT"), ("st", "ST"), ("tr", "TFR"))}
    k5, splits = E.reproduce_e8(t, preds)
    assert k5["ok"], k5["mismatches"]
    all_repos = sorted(t.loc[(t.hait_filter == 1) & (t.in_main == 1), "repo"].unique())
    checks, nums = {"k5_e8_reproduction_ok": True}, {}

    # empirical k
    PA4, chk4 = r4_arrays(t, preds, all_repos)
    rank_repos = sorted(PA4)
    assert len(rank_repos) == 12
    e8_split = {r: splits["it"][r] for r in all_repos}
    kd = k_data(t, e8_split, rank_repos)
    K = k_estimates(kd, seed, b_ols, b_qr)
    K40 = k_estimates(k_data(t, e8_split, all_repos), seed, b_ols, 0, quantile=False)
    for k, v in K.items():
        nums[f"e9.r4.k.{k}"] = v
    for k, v in K40.items():
        nums[f"e9.r4.k40.{k}"] = v
    print("k:", json.dumps(K), flush=True)

    # E9b
    if args.dev:
        PA4 = {r: PA4[r] for r in DEV_R4}
    repos4 = sorted(PA4)
    kvals = {"k1": 1.0, "kemp": K["kemp"], "kemp_lo": K["kemp_lo"], "kemp_hi": K["kemp_hi"]}
    V4 = r4_run(PA4, all_repos, seed, n_var, kvals)
    idx4 = np.random.default_rng(seed).integers(0, len(repos4), size=(E.B, len(repos4)))
    r4_summary(V4, repos4, idx4, nums)
    nums["e9.r4.n_prs"] = int(sum(len(PA4[r]["row_id"]) for r in repos4))
    nums["e9.r4.n_prs_ranking_test"] = chk4["n_per_pr"]
    nums["e9.r4.n_repos"] = len(repos4)
    for k, v in chk4["repro"].items():
        nums[f"e9.r4.repro.{k}"] = v
    # K1b: zero dt_r gives G = 0 exactly
    P0 = dict(PA4[repos4[0]])
    P0["dtr"] = np.zeros_like(P0["dtr"])
    n0 = len(P0["row_id"])
    z = r4_variant(P0, np.arange(n0 // 2), np.arange(n0 // 2, n0), {"k1": 1.0})
    checks["k1b_zero_dtr"] = all(r[f"G4_{w}"] == 0.0 and r[f"Gall_{w}"] == 0.0 for r in z for w in W4)
    ctrl = {k: v for k, v in nums.items() if k.startswith("e9.r4.ctrl.k1.") and ("lift_var_p025" in k or "lift_var_p975" in k)}
    k2b = {}
    for key, v in ctrl.items():
        if key.endswith("p025") and v is not None:
            hi = ctrl[key.replace("p025", "p975")]
            k2b[key[:-len(".lift_var_p025")]] = {"p025": v, "p975": hi, "contains_1": v <= 1 <= hi}
    checks["k2b_shuffled_ptr"] = k2b
    checks["k2b_ok"] = all(x["contains_1"] for x in k2b.values())
    checks["k5b_r4_known"] = {**chk4["repro"], "expected": {
        "fire_c02_pct": 12.318137730358874, "fired_med_rel_h": 11.574722222222222,
        "fired_med_lat_h": 5.9325, "all_med_rel_h": 4.280277777777778,
        "all_med_lat_h": 2.6283333333333334}}
    checks["k5b_ok"] = all(abs(chk4["repro"][k] - v) < 1e-9
                           for k, v in checks["k5b_r4_known"]["expected"].items())
    checks["data_r4"] = {k: v for k, v in chk4.items() if k != "repro"}

    # R3grid: R3 grid on the E9 generator
    PA = E.project_arrays(t, preds, None)
    repos3 = sorted(DEV_R4) if args.dev else all_repos
    V3 = r3_grid(PA, all_repos, repos3, seed, n_var)
    idx3 = np.random.default_rng(seed).integers(0, len(repos3), size=(E.B, len(repos3)))
    r3_summary(V3, repos3, idx3, nums)
    if not args.dev:
        e9 = json.load(open(os.path.join(E.OUT, "numbers.json"), encoding="utf-8"))
        checks["r3grid_matches_e9"] = all(
            nums[f"e9.r3grid.{g}.n_reach_med"] == 40 - e9[f"e9.shuf.st.{c}.n_unreach_med"]
            and abs(nums[f"e9.r3grid.{g}.w3.G_pct"] - e9[f"e9.shuf.st.{c}.w3.A_pct"]) < 1e-12
            for g, c in (("c020", "c02"), ("c030", "c03")))

    run = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": ("dev" if args.dev else "main") + "_e9b",
           "seed": seed, "n_var": n_var, "repos_r4": repos4 if args.dev else len(repos4),
           "repos_r3": repos3 if args.dev else len(repos3), "git": E.git_head(),
           "seconds": round(time.time() - t0, 1),
           "checks": {k: v for k, v in checks.items() if k.endswith("ok") or k.startswith("k1b")
                      or k == "r3grid_matches_e9"},
           "summary": {k: nums.get(k) for k in (
               "e9.r4.k1.c02.w3.G_pct", "e9.r4.k1.c02.w3.Grand_pct", "e9.r4.k1.c02.w3.lift",
               "e9.r4.k1.c02.w3.lift_lo", "e9.r4.k1.c02.w3.lift_hi", "e9.r4.k.kemp",
               "e9.r4.k.c_shift", "e9.r3grid.min_c_half")}}
    with open(os.path.join(E.OUT, "runs.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(run, ensure_ascii=False) + "\n")
    checks["runs"] = [json.loads(x) for x in open(os.path.join(E.OUT, "runs.jsonl"), encoding="utf-8")]
    json.dump(checks, open(os.path.join(out, "checks_e9b.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    json.dump({k: nums[k] for k in sorted(nums)},
              open(os.path.join(out, "numbers_e9b.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    V4.to_csv(os.path.join(out, "variants_e9b_r4.csv"), index=False)
    V3.to_csv(os.path.join(out, "variants_e9b_r3.csv"), index=False)
    print(json.dumps(run, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
