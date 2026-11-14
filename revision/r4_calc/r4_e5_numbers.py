"""E5 for chapter 4: calibration on the project history and the rule threshold on a short
history. Collects numbers from the finished outputs of r3_calib_stability.py and
r3_calib_adversarial/ (13.11.2026); no new fits.

Inputs (read only, produced earlier):
  revision/output/r3/calib_stability/   (revision/r3_calib_stability.py: bootstrap of a history
                                          of n = 30/50/100/200 merged PRs, 40 repositories)
  revision/output/r3/calib_adversarial/ (revision/r3_calib_adversarial/adv_*.py: measures on the
                                          later half, rolling origin rules, short-history threshold)

Flat keys (prefix e5.), k in {it, tr, tm, st}:
  minority_med_200.<k>          median over repos of 200 * min(rate, 1 - rate)
  m100_hist_{med,p10,p90,min,max}.<k>  history length 100 / min(rate, 1 - rate) for >= 100 PRs of
                                       each class; rate = (rate_early + rate_late) / 2 of whole_half.csv
                                       (as in adv_calib.py)
  minority_med_200_hist.<k>     median over repos of the actual minority count in the history of
                                200 PRs used by r3_calib_stability (per_repo_by_n.csv)
  stab.<k>.n<N>.*               summary_by_n.csv
  ece.<k>.{raw,platt,corr,const}  ECE (10 equal-count bins) on the later half, mean over repos
  rules.<rule>.<k>.{brier,logloss,ece}  rules_rolling2.csv; ece is null (no per-PR predictions saved)
  diff_ece_corr_minus_platt{,_lo,_hi}.<k>  reliab_diff.csv (paired cluster bootstrap, 10 000)
  thr30.<k>.c<cc>.*             threshold_short.csv (MIN_FIRES = 20), history n = 30
  thr30_nominfire.<k>.c<cc>.*   threshold_nominfire.csv (no firing floor), history n = 30
  n_b1_nonpos.<k>               reference_summary.csv (b1 <= 0 on the whole earlier half)
Added on 14.11:
  m100_hist_direct_{med,p10,p90}.<k>, m100_n_repos_reached.<k>  direct count on the LORO table:
                                all merged PRs of the repo in merge order, first length with >= 100
                                PRs of each class (median and percentiles over repos that reach it)
  ece.<k>.const_early, diff_ece_corr_minus_const_early{,_lo,_hi}.<k>, n_repos_corr_better_const_early.<k>
                                whole_half.csv: constant early-half rate of the project; difference
                                is ECE(correction) - ECE(constant), negative = correction better;
                                paired cluster bootstrap over repos, 10 000, seed 20261114
  oe_out_08_125.<k>             whole_half.csv: repos with O/E of the corrected forecast outside [0.8; 1.25]
  stab.<k>.stable_first_n       summary_by_n.csv: smallest n with K1-K4 met at it and every larger n
  thr30_nominfire.<k>.c<cc>.hist_fires_med  median firings on the history of 30 (reachable thresholds)
  thr_early.<k>.c<cc>.*         threshold_short.csv n = -1: threshold on the whole earlier half
  rules.none.<k>.*              = rules.prior: class-weight correction only, no calibration on the history
  ece_pct.<k>.{raw,platt,corr,const,const_early}  the same ECE x 100 (percentage points)
  diff_ece_pct_const_early_minus_corr{,_lo,_hi}.<k>  -(diff_ece_corr_minus_const_early) x 100 with the
                                same bootstrap draws (positive = correction better)
Percent keys (*_pct*) are in percent units.

Output: revision/output/r4/e5_calibration/{numbers.json, sources.json}.
"""
import json
import os

import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CLONE = os.path.dirname(os.path.dirname(HERE))
OUT_R3 = os.path.join(CLONE, "revision", "output", "r3")
STAB = os.path.join(OUT_R3, "calib_stability")
ADV = os.path.join(OUT_R3, "calib_adversarial")
OUT = os.path.join(CLONE, "revision", "output", "r4", "e5_calibration")

sys.path.insert(0, os.path.join(CLONE, "revision", "r3_calib_adversarial"))
import adv_calib  # noqa: E402  (load/preds of the LORO forecasts, read only)

LABELS = ("it", "tr", "tm", "st")
SEED = 20261114
B = 10_000
NS = (30, 50, 100, 200)
RULES = {"r200": "r200", "m100": "v100", "epv20": "epv20"}  # key name -> rule in adv_rules2.py
CS = (0.1, 0.2, 0.3)

num, src = {}, {}


def put(key, value, source):
    if value is not None and not isinstance(value, str):
        value = None if pd.isna(value) else (int(value) if isinstance(value, (int, np.integer)) else float(value))
    num["e5." + key] = value
    src["e5." + key] = source


def rel(path):
    return os.path.relpath(path, CLONE)


def main():
    # ── class balance and history needed (inline computation on whole_half.csv)
    wh_path = os.path.join(ADV, "whole_half.csv")
    W = pd.read_csv(wh_path)
    W["rate"] = (W.rate_early + W.rate_late) / 2
    W["minr"] = np.minimum(W.rate, 1 - W.rate)
    s_rate = f"{rel(wh_path)}: rate = (rate_early + rate_late) / 2, minr = min(rate, 1 - rate)"
    for k in LABELS:
        g = W[W.label == k]
        put(f"minority_med_200.{k}", (200 * g.minr).median(), s_rate + "; median over repos of 200 * minr")
        h = 100 / g.minr
        put(f"m100_hist_med.{k}", h.median(), s_rate + "; median over repos of 100 / minr")
        put(f"m100_hist_p10.{k}", h.quantile(.1), s_rate + "; 10th percentile over repos of 100 / minr")
        put(f"m100_hist_p90.{k}", h.quantile(.9), s_rate + "; 90th percentile over repos of 100 / minr")
        put(f"m100_hist_min.{k}", h.min(), s_rate + "; min over repos of 100 / minr")
        put(f"m100_hist_max.{k}", h.max(), s_rate + "; max over repos of 100 / minr")

    per_path = os.path.join(STAB, "per_repo_by_n.csv")
    P = pd.read_csv(per_path)
    P200 = P[P.n == 200].copy()
    P200["minority"] = np.minimum(P200.pos_hist, 200 - P200.pos_hist)
    for k in LABELS:
        put(f"minority_med_200_hist.{k}", P200[P200.label == k].minority.median(),
            f"{rel(per_path)}: n = 200, median over repos of min(pos_hist, 200 - pos_hist)")

    # ── stability of Platt on a history of n (summary_by_n.csv)
    sb_path = os.path.join(STAB, "summary_by_n.csv")
    S = pd.read_csv(sb_path)
    for k in LABELS:
        for n in NS:
            r = S[(S.label == k) & (S.n == n)].iloc[0]
            pre, s = f"stab.{k}.n{n}.", f"{rel(sb_path)} label={k} n={n}: "
            put(pre + "n_repos", r.n_repos, s + "n_repos")
            put(pre + "pos_med", r.pos_hist_median, s + "pos_hist_median (labels 1 in the history)")
            put(pre + "b1_med", r.b1_med_median, s + "b1_med_median (median over repos of bootstrap medians)")
            put(pre + "b1_lo", r.b1_q05_median, s + "b1_q05_median (median over repos of the 5th bootstrap percentile)")
            put(pre + "b1_hi", r.b1_q95_median, s + "b1_q95_median (median over repos of the 95th bootstrap percentile)")
            put(pre + "b1_w90", r.b1_w90_median, s + "b1_w90_median (median width of the 90 % interval, K3)")
            put(pre + "n_b1_nonpos", r.pt_b1_le0, s + "pt_b1_le0 (repos with b1 <= 0 for the point fit on the history)")
            put(pre + "n_fit", r.n_repos - r.pt_oneclass, s + "n_repos - pt_oneclass (repos where the point fit exists)")
            put(pre + "n_b1_le0_gt5pct", r.repos_b1_le0_gt5pct, s + "repos_b1_le0_gt5pct (repos with > 5 % bootstrap b1 <= 0, K2)")
            put(pre + "ece", r.pt_ece_full_mean, s + "pt_ece_full_mean (ECE after the point fit, mean over repos)")
            put(pre + "ece_boot", r.ece_full_mean, s + "ece_full_mean (mean over repos of bootstrap median ECE)")
            put(pre + "stable", "yes" if bool(r.stable) else "no", s + "stable (K1-K4)")
        g = S[S.label == k].sort_values("n")
        first = None
        for n in g.n[::-1]:
            if bool(g[g.n == n].stable.iloc[0]):
                first = int(n)
            else:
                break
        put(f"stab.{k}.stable_first_n", first,
            f"{rel(sb_path)}: smallest n with stable at it and at every larger n (null: none)")

    # ── ECE on the later half: raw, Platt on the whole earlier half, prior correction, constant pi
    rs_path = os.path.join(STAB, "reference_summary.csv")
    R = pd.read_csv(rs_path).set_index("label")
    for k in LABELS:
        put(f"ece.{k}.raw", R.loc[k, "ece_raw"], f"{rel(rs_path)}: ece_raw")
        put(f"ece.{k}.platt", R.loc[k, "ece_cal"], f"{rel(rs_path)}: ece_cal (Platt on the whole earlier half)")
        put(f"ece.{k}.corr", R.loc[k, "ece_prior"], f"{rel(rs_path)}: ece_prior (class-weight correction)")
        put(f"ece.{k}.const", W[W.label == k].constpi_ece10.mean(), f"{rel(wh_path)}: mean constpi_ece10 (constant pi)")
        put(f"n_b1_nonpos.{k}", R.loc[k, "b1_le0"], f"{rel(rs_path)}: b1_le0 (whole earlier half)")
        # cross-checks against the independent sklearn re-fit
        g = W[W.label == k]
        assert abs(g.raw_ece10.mean() - R.loc[k, "ece_raw"]) < 1e-3
        assert abs(g.plattE_ece10.mean() - R.loc[k, "ece_cal"]) < 1e-3
        assert abs(g.prior_ece10.mean() - R.loc[k, "ece_prior"]) < 1e-3
        assert int((g.b1_early <= 0).sum()) == int(R.loc[k, "b1_le0"])

    # ── rules for switching calibration on, rolling origin over the whole history
    rr_path = os.path.join(ADV, "rules_rolling2.csv")
    O = pd.read_csv(rr_path)
    a = O.groupby(["label", "repo", "rule"]).agg(n=("n", "sum"), sse=("sse", "sum"), ll=("ll", "sum")).reset_index()
    a["brier"] = a.sse / a.n
    a["logloss"] = a.ll / a.n
    m = a.groupby(["label", "rule"])[["brier", "logloss"]].mean()
    for key, rule in RULES.items():
        for k in LABELS:
            s = f"{rel(rr_path)} rule={rule}: per repo sum / n over all origins, mean over repos"
            put(f"rules.{key}.{k}.brier", m.loc[(k, rule), "brier"], s)
            put(f"rules.{key}.{k}.logloss", m.loc[(k, rule), "logloss"], s)
            put(f"rules.{key}.{k}.ece", None, f"{rel(rr_path)}: per-block sums only, no per-PR predictions; ECE not computed")
    for k in LABELS:
        put(f"rules.prior.{k}.brier", m.loc[(k, "prior"), "brier"], f"{rel(rr_path)} rule=prior (correction only)")
        put(f"rules.prior.{k}.logloss", m.loc[(k, "prior"), "logloss"], f"{rel(rr_path)} rule=prior (correction only)")
        s_none = f"{rel(rr_path)} rule=prior: no calibration on the history (class-weight correction only), same as rules.prior"
        put(f"rules.none.{k}.brier", m.loc[(k, "prior"), "brier"], s_none)
        put(f"rules.none.{k}.logloss", m.loc[(k, "prior"), "logloss"], s_none)

    # ── paired cluster bootstrap: ECE correction - Platt on the whole earlier half
    rd_path = os.path.join(ADV, "reliab_diff.csv")
    D = pd.read_csv(rd_path).set_index("label")
    for k in LABELS:
        s = f"{rel(rd_path)}: ece_prior_minus_plattE [ece_lo; ece_hi], 10 000 draws over repos, seed 1"
        put(f"diff_ece_corr_minus_platt.{k}", D.loc[k, "ece_prior_minus_plattE"], s)
        put(f"diff_ece_corr_minus_platt_lo.{k}", D.loc[k, "ece_lo"], s)
        put(f"diff_ece_corr_minus_platt_hi.{k}", D.loc[k, "ece_hi"], s)

    # ── threshold on a history of 30 merged PRs (start of the later half), evaluated on the later half
    def thr(path, prefix, share_col):
        T = pd.read_csv(path)
        T = T[T.n == 30].copy()
        T["below"] = T.prec < 1 - T.c - 1e-12
        for k in ("it", "tr", "tm"):
            for c in CS:
                g = T[(T.label == k) & (np.isclose(T.c, c))]
                rch = g[g.reach.astype(bool)]
                pre = f"{prefix}.{k}.c{int(round(c * 10)):02d}."
                s = f"{rel(path)} label={k} c={c} n=30"
                put(pre + "n_projects", len(g), s + ": repos with a history of 30")
                put(pre + "n_reachable", len(rch), s + ": reach")
                put(pre + "fire_pct_min", rch[share_col].min() * 100 if len(rch) else None, s + f": min {share_col} x 100")
                put(pre + "fire_pct_max", rch[share_col].max() * 100 if len(rch) else None, s + f": max {share_col} x 100")
                put(pre + "fire_pct_med", rch[share_col].median() * 100 if len(rch) else None, s + f": median {share_col} x 100")
                put(pre + "n_below_target", int(rch.below.sum()), s + ": later-half precision < 1 - c")
                put(pre + "prec_med", rch.prec.median() if len(rch) else None, s + ": median later-half precision")

    thr(os.path.join(ADV, "threshold_short.csv"), "thr30", "fire_share")
    thr(os.path.join(ADV, "threshold_nominfire.csv"), "thr30_nominfire", "fire")

    tn_path = os.path.join(ADV, "threshold_nominfire.csv")
    TN = pd.read_csv(tn_path)
    TN = TN[(TN.n == 30) & TN.reach.astype(bool)]
    for k in ("it", "tr", "tm"):
        for c in CS:
            g = TN[(TN.label == k) & np.isclose(TN.c, c)]
            put(f"thr30_nominfire.{k}.c{int(round(c * 10)):02d}.hist_fires_med", g.fires_hist.median() if len(g) else None,
                f"{rel(tn_path)} label={k} c={c} n=30: median fires_hist over reachable thresholds")

    ts_path = os.path.join(ADV, "threshold_short.csv")
    TE = pd.read_csv(ts_path)
    TE = TE[TE.n == -1].copy()
    TE["below"] = TE.prec < 1 - TE.c - 1e-12
    for k in ("it", "tr", "tm"):
        for c in CS:
            g = TE[(TE.label == k) & np.isclose(TE.c, c)]
            rch = g[g.reach.astype(bool)]
            pre = f"thr_early.{k}.c{int(round(c * 10)):02d}."
            s = f"{rel(ts_path)} label={k} c={c} n=-1 (whole earlier half, MIN_FIRES = 20)"
            put(pre + "n_projects", len(g), s + ": repos")
            put(pre + "n_reachable", len(rch), s + ": reach")
            put(pre + "n_below_target", int(rch.below.sum()), s + ": later-half precision < 1 - c")
            put(pre + "fire_pct_med", rch.fire_share.median() * 100 if len(rch) else None, s + ": median fire_share x 100")
            put(pre + "prec_med", rch.prec.median() if len(rch) else None, s + ": median later-half precision")

    # ── constant early-half rate, O/E of the corrected forecast (whole_half.csv)
    rng = np.random.default_rng(SEED)
    for k in LABELS:
        g = W[W.label == k]
        put(f"ece.{k}.const_early", g.constE_ece10.mean(), f"{rel(wh_path)}: mean constE_ece10 (constant early-half rate)")
        d = (g.prior_ece10 - g.constE_ece10).to_numpy()
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
        s = (f"{rel(wh_path)}: mean over repos of prior_ece10 - constE_ece10 (negative = correction better); "
             f"paired cluster bootstrap over repos, B = {B}, seed {SEED}, percentile 95 %")
        put(f"diff_ece_corr_minus_const_early.{k}", d.mean(), s)
        put(f"diff_ece_corr_minus_const_early_lo.{k}", np.quantile(bs, .025), s)
        put(f"diff_ece_corr_minus_const_early_hi.{k}", np.quantile(bs, .975), s)
        put(f"n_repos_corr_better_const_early.{k}", int((d < 0).sum()), f"{rel(wh_path)}: repos with prior_ece10 < constE_ece10")
        put(f"oe_out_08_125.{k}", int(((g.prior_oe < .8) | (g.prior_oe > 1.25)).sum()),
            f"{rel(wh_path)}: repos with prior_oe outside [0.8; 1.25]")

    # ── direct count: history length (merged PRs in merge order) with >= 100 PRs of each class
    t, mm = adv_calib.load()
    for k, tg in adv_calib.TGT.items():
        d = adv_calib.preds(t, mm, tg)
        lens = []
        for repo, g in d.groupby("test"):
            y = g[g.merged.notna()].sort_values("merged", kind="stable").y.to_numpy(float)
            ok = np.nonzero((np.cumsum(y) >= 100) & (np.cumsum(1 - y) >= 100))[0]
            lens.append(int(ok[0]) + 1 if len(ok) else np.nan)
        L = pd.Series(lens, dtype=float)
        s = ("LORO forecasts (revision/output/r3_b40d/v2, adv_calib.load/preds): merged PRs of the repo in merge "
             "order, first length with >= 100 of each class; over repos that reach it")
        put(f"m100_n_repos_reached.{k}", int(L.notna().sum()), s + f" (of {len(L)})")
        put(f"m100_n_repos.{k}", len(L), "LORO forecasts: repos with forecasts")
        put(f"m100_hist_direct_med.{k}", L.median(), s)
        put(f"m100_hist_direct_p10.{k}", L.quantile(.1), s)
        put(f"m100_hist_direct_p90.{k}", L.quantile(.9), s)

    # ── mirror keys in percentage points for the text
    for k in LABELS:
        for c in ("raw", "platt", "corr", "const", "const_early"):
            put(f"ece_pct.{k}.{c}", num[f"e5.ece.{k}.{c}"] * 100, f"e5.ece.{k}.{c} x 100")
        b = f"diff_ece_corr_minus_const_early"
        s = f"-(e5.{b}.{k}) x 100, bounds -hi and -lo of the same bootstrap (const early minus correction)"
        put(f"diff_ece_pct_const_early_minus_corr.{k}", -num[f"e5.{b}.{k}"] * 100, s)
        put(f"diff_ece_pct_const_early_minus_corr_lo.{k}", -num[f"e5.{b}_hi.{k}"] * 100, s)
        put(f"diff_ece_pct_const_early_minus_corr_hi.{k}", -num[f"e5.{b}_lo.{k}"] * 100, s)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "numbers.json"), "w") as f:
        json.dump(num, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "sources.json"), "w") as f:
        json.dump(src, f, ensure_ascii=False, indent=1)
    print(f"{len(num)} keys ->", rel(os.path.join(OUT, "numbers.json")))


if __name__ == "__main__":
    main()
