#!/usr/bin/env python3
"""Stability of the per-project Platt calibration on a short history
(item 3.2.5 of chapter 3).

Reuses the stored LORO predictions of the main run (no retraining). For each
label and test repository, the split is the one of r3_report.calibration():
PRs sorted by creation, earlier half / later half. The calibration moment t_b
is the creation of the first PR of the later half; the history of size n is
the n PRs of the earlier half merged last strictly before t_b (as H_k(p) in the
method). For n in {30, 50, 100, 200} a bootstrap of the history gives the
distribution of b0, b1 (full Platt) and of b0 with b1 = 1 (shift only), and
Brier / ECE (10 equal-count bins) of each fit on the later half.

Reads revision/output/r3_b40d/v2 (rerun of 02.11.2026, the source of the
chapter 3 numbers; override with R3_OUT). Writes ONLY into
revision/output/r3/calib_stability/.

    python -I revision/r3_calib_stability.py
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Predictions and labels of the rerun of 02.11.2026 (r3_rerun_0211.sh): the
# source of the chapter 3 numbers (r3_numbers.json, table r3_calib).
V2 = os.environ.get("R3_OUT") or os.path.join(HERE, "output", "r3_b40d", "v2")
OUT = os.path.join(HERE, "output", "r3", "calib_stability")
os.makedirs(OUT, exist_ok=True)

SEED = 20261113
B = 1000
NS = [30, 50, 100, 200]
POS_TARGETS = [5, 10]          # variable-length history for rare labels (option a)
N_START = 30                   # the variable history is never shorter than n_min
TGT = {"it": "IT", "st": "ST", "tr": "TFR", "tm": "TM"}
C_REG = 1e6                    # as in r3_report_stats.platt_fit (LogisticRegression(C=1e6))
LAM = 1.0 / C_REG
SEP = 20.0                     # |b1| above this = (quasi-)separated history
BINS = 10
EPS = 1e-6


# ── data (same selection as r3_report.preds) ────────────────────────────────

def load():
    t = pd.read_csv(os.path.join(V2, "pr_table_v2.csv"), low_memory=False,
                    usecols=["repo", "created_at", "merged_at"] + [f"y_{v}__main_D" for v in TGT.values()])
    t["created"] = pd.to_datetime(t["created_at"], utc=True)
    t["merged"] = pd.to_datetime(t["merged_at"], utc=True)
    m = pd.read_json(os.path.join(V2, "models", "metrics.jsonl"), lines=True)
    m = m[(m.variant == "main") & (m.protocol == "loro") & (m.model == "CatBoost") & (m.clock == "D")
          & (m["pop"] == "main") & (m.label == m.target)]
    return t, m


def preds(t, m, tg):
    q = m[m.target == tg]
    parts = []
    for _, r in q.iterrows():
        z = np.load(os.path.join(V2, "models", "oof", r.oof_file))
        parts.append(pd.DataFrame({"row_id": z["row_id"], "prob": z["prob"].astype(float), "test": r.test,
                                   "pos_train": float(r.pos_train)}))
    d = pd.concat(parts, ignore_index=True)
    rid = d.row_id.to_numpy()
    d["y"] = t[f"y_{tg}__main_D"].to_numpy()[rid]
    d["created"] = t["created"].to_numpy()[rid]
    d["merged"] = t["merged"].to_numpy()[rid]
    return d


# ── fits (batched over bootstrap weights) ───────────────────────────────────

def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def fit_full(x, y, W, iters=200):
    """Weighted logistic regression y ~ a + b*x for every row of W (B x n).
    Objective as sklearn LogisticRegression(C=1e6): sum w*loss + b^2/(2C)."""
    Bn = W.shape[0]
    a = np.zeros(Bn)
    b = np.zeros(Bn)
    for _ in range(iters):
        p = sigmoid(a[:, None] + b[:, None] * x[None, :])
        r = W * (p - y[None, :])
        s = W * p * (1 - p)
        ga = r.sum(1)
        gb = (r * x[None, :]).sum(1) + LAM * b
        haa = s.sum(1) + 1e-12
        hab = (s * x[None, :]).sum(1)
        hbb = (s * x[None, :] ** 2).sum(1) + LAM
        det = haa * hbb - hab ** 2
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        step = np.maximum(1.0, np.sqrt(da ** 2 + db ** 2) / 5.0)  # cap the Newton step
        a -= da / step
        b -= db / step
        if np.max(np.abs(da) + np.abs(db)) < 1e-9:
            break
    return a, b


def fit_shift(x, y, W, iters=200):
    """Shift only: y ~ a + 1*x (offset), unpenalised."""
    a = np.zeros(W.shape[0])
    for _ in range(iters):
        p = sigmoid(a[:, None] + x[None, :])
        g = (W * (p - y[None, :])).sum(1)
        h = (W * p * (1 - p)).sum(1) + 1e-12
        d = g / h
        a -= d / np.maximum(1.0, np.abs(d) / 5.0)
        if np.max(np.abs(d)) < 1e-9:
            break
    return a


def ece_rows(y, P):
    """ECE with equal-count bins (r3_report_stats.ece) for every row of P."""
    m = P.shape[1]
    order = np.argsort(P, axis=1, kind="stable")
    ps = np.take_along_axis(P, order, axis=1)
    ys = y[order]
    sizes = [len(c) for c in np.array_split(np.arange(m), min(BINS, m))]
    edges = np.concatenate([[0], np.cumsum(sizes)])
    cp = np.concatenate([np.zeros((P.shape[0], 1)), np.cumsum(ps, 1)], 1)
    cy = np.concatenate([np.zeros((P.shape[0], 1)), np.cumsum(ys, 1)], 1)
    tot = np.zeros(P.shape[0])
    for lo, hi in zip(edges[:-1], edges[1:]):
        tot += np.abs((cy[:, hi] - cy[:, lo]) - (cp[:, hi] - cp[:, lo]))
    return tot / m


def ece1(y, p):
    return float(ece_rows(y, p[None, :])[0])


def q(v, a):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    return float(np.quantile(v, a)) if len(v) else np.nan


# ── one history ─────────────────────────────────────────────────────────────

def boot_history(xh, yh, xl, yl, rng):
    """Bootstrap of one history (xh, yh) evaluated on the later half (xl, yl)."""
    n = len(xh)
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(float)
    npos = W @ yh
    one = (npos == 0) | (npos == n)
    ok = ~one
    raw = sigmoid(xl)                       # = clipped raw prediction
    b0 = np.full(B, np.nan); b1 = np.full(B, np.nan); s0 = np.full(B, np.nan)
    if ok.any():
        a, b = fit_full(xh, yh, W[ok])
        b0[ok], b1[ok] = a, b
        s0[ok] = fit_shift(xh, yh, W[ok])
    Pf = np.tile(raw, (B, 1)); Ps = np.tile(raw, (B, 1))
    if ok.any():
        Pf[ok] = sigmoid(b0[ok, None] + b1[ok, None] * xl[None, :])
        Ps[ok] = sigmoid(s0[ok, None] + xl[None, :])
    ef, es = ece_rows(yl, Pf), ece_rows(yl, Ps)
    bf, bs = ((Pf - yl) ** 2).mean(1), ((Ps - yl) ** 2).mean(1)
    fitted = b1[ok]
    return {
        "share_oneclass": float(one.mean()),
        "b0_med": q(b0, .5), "b0_q05": q(b0, .05), "b0_q95": q(b0, .95),
        "b1_med": q(b1, .5), "b1_q05": q(b1, .05), "b1_q95": q(b1, .95),
        "b1_w90": q(b1, .95) - q(b1, .05), "b0_w90": q(b0, .95) - q(b0, .05),
        "share_b1_le0": float(np.mean(fitted <= 0)) if len(fitted) else np.nan,
        "share_sep": float(np.mean(np.abs(fitted) > SEP)) if len(fitted) else np.nan,
        "s0_med": q(s0, .5), "s0_w90": q(s0, .95) - q(s0, .05),
        "ece_full_med": float(np.median(ef)), "ece_full_q95": float(np.quantile(ef, .95)),
        "ece_shift_med": float(np.median(es)), "ece_shift_q95": float(np.quantile(es, .95)),
        "brier_full_med": float(np.median(bf)), "brier_shift_med": float(np.median(bs)),
        "share_ece_worse_full": float(np.mean(ef > ece1(yl, raw))),
    }


def point_fit(xh, yh, xl, yl):
    """Fit on the history itself (no bootstrap)."""
    if yh.min() == yh.max():
        return {"pt_oneclass": 1, "pt_b0": np.nan, "pt_b1": np.nan, "pt_ece_full": np.nan, "pt_ece_shift": np.nan}
    W = np.ones((1, len(xh)))
    a, b = fit_full(xh, yh, W)
    s = fit_shift(xh, yh, W)
    return {"pt_oneclass": 0, "pt_b0": float(a[0]), "pt_b1": float(b[0]),
            "pt_ece_full": ece1(yl, sigmoid(a[0] + b[0] * xl)), "pt_ece_shift": ece1(yl, sigmoid(s[0] + xl))}


# ── driver ──────────────────────────────────────────────────────────────────

def split_repo(g):
    g = g.sort_values("created", kind="stable")
    h = len(g) // 2
    early, late = g.iloc[:h], g.iloc[h:]
    return h, early, late


def main():
    t0 = time.time()
    t, m = load()
    ref_rows, rows, var_rows = [], [], []
    for li, (k, tg) in enumerate(TGT.items()):
        d = preds(t, m, tg)
        d = d[~d.y.isna()]
        for ri, (repo, g) in enumerate(sorted(d.groupby("test"), key=lambda z: z[0])):
            h, early, late = split_repo(g)
            if h < 20 or len(late) == 0:
                continue
            xl, yl = logit(late.prob.to_numpy()), late.y.to_numpy(dtype=float)
            ece_raw = ece1(yl, late.prob.to_numpy())
            brier_raw = float(np.mean((late.prob.to_numpy() - yl) ** 2))
            # reference: the whole earlier half, as in r3_report.calibration()
            xe, ye = logit(early.prob.to_numpy()), early.y.to_numpy(dtype=float)
            ref = {"label": k, "repo": repo, "n_early": h, "n_late": len(late), "pos_early": int(ye.sum()),
                   "pos_rate_late": float(yl.mean()), "ece_raw": ece_raw, "brier_raw": brier_raw}
            if ye.min() < ye.max():
                a, b = fit_full(xe, ye, np.ones((1, len(xe))))
                s = fit_shift(xe, ye, np.ones((1, len(xe))))
                pc = sigmoid(a[0] + b[0] * xl)
                ref.update(ref_b0=float(a[0]), ref_b1=float(b[0]), ref_ece_cal=ece1(yl, pc),
                           ref_brier_cal=float(np.mean((pc - yl) ** 2)),
                           ref_ece_shift=ece1(yl, sigmoid(s[0] + xl)))
            # zero-parameter prior correction of the balanced class weights (needs no history):
            # logit p = logit p_w + ln(pi / (1 - pi)), pi = share of label 1 in the training projects
            pi = float(g.pos_train.iloc[0])
            pp = sigmoid(xl + np.log(pi / (1 - pi)))
            ref.update(pos_train=pi, ece_prior=ece1(yl, pp), brier_prior=float(np.mean((pp - yl) ** 2)))
            ref_rows.append(ref)
            if "ref_b1" not in ref:
                continue
            # history as in the method: merged strictly before the calibration moment
            tb = late.created.iloc[0]
            pool = early[early.merged < tb].sort_values("merged", kind="stable")
            ref["n_pool"] = len(pool)
            xp, yp = logit(pool.prob.to_numpy()), pool.y.to_numpy(dtype=float)
            for n in NS:
                if len(pool) < n:
                    continue
                xh, yh = xp[-n:], yp[-n:]
                rng = np.random.default_rng(np.random.SeedSequence([SEED, li, ri, n]))
                rec = {"label": k, "repo": repo, "n": n, "pos_hist": int(yh.sum()), "ece_raw": ece_raw,
                       "brier_raw": brier_raw, "ref_ece_cal": ref["ref_ece_cal"], "ref_b1": ref["ref_b1"]}
                rec.update(point_fit(xh, yh, xl, yl))
                rec.update(boot_history(xh, yh, xl, yl, rng))
                rows.append(rec)
            # option (a): variable history until m positives (never shorter than N_START)
            cpos = np.cumsum(yp[::-1])
            for mp in POS_TARGETS:
                idx = np.nonzero(cpos >= mp)[0]
                if not len(idx):
                    var_rows.append({"label": k, "repo": repo, "m_pos": mp, "reached": 0, "n_pool": len(pool),
                                     "pos_pool": int(yp.sum())})
                    continue
                n = max(N_START, int(idx[0]) + 1)
                if len(pool) < n:
                    var_rows.append({"label": k, "repo": repo, "m_pos": mp, "reached": 0, "n_pool": len(pool),
                                     "pos_pool": int(yp.sum())})
                    continue
                xh, yh = xp[-n:], yp[-n:]
                rng = np.random.default_rng(np.random.SeedSequence([SEED, li, ri, 1000 + mp]))
                rec = {"label": k, "repo": repo, "m_pos": mp, "reached": 1, "n": n, "n_pool": len(pool),
                       "pos_pool": int(yp.sum()), "pos_hist": int(yh.sum()), "ece_raw": ece_raw,
                       "brier_raw": brier_raw, "ref_ece_cal": ref["ref_ece_cal"], "ref_b1": ref["ref_b1"]}
                rec.update(point_fit(xh, yh, xl, yl))
                rec.update(boot_history(xh, yh, xl, yl, rng))
                var_rows.append(rec)
        print(f"{k}: done {time.time() - t0:.0f}s", flush=True)

    ref = pd.DataFrame(ref_rows)
    per = pd.DataFrame(rows)
    var = pd.DataFrame(var_rows)
    ref.to_csv(os.path.join(OUT, "reference_early_half.csv"), index=False)
    per.to_csv(os.path.join(OUT, "per_repo_by_n.csv"), index=False)
    var.to_csv(os.path.join(OUT, "per_repo_pos_target.csv"), index=False)

    # reference check against table r3_calib (mean over repos)
    chk = ref.dropna(subset=["ref_b1"]).groupby("label").agg(
        n_repos=("repo", "size"), brier_raw=("brier_raw", "mean"), brier_cal=("ref_brier_cal", "mean"),
        ece_raw=("ece_raw", "mean"), ece_cal=("ref_ece_cal", "mean"), ece_shift=("ref_ece_shift", "mean"),
        ece_prior=("ece_prior", "mean"), brier_prior=("brier_prior", "mean"),
        b1_le0=("ref_b1", lambda v: int((v <= 0).sum())), b1_median=("ref_b1", "median"),
        b0_median=("ref_b0", "median"), pos_early_median=("pos_early", "median"))
    chk.to_csv(os.path.join(OUT, "reference_summary.csv"))

    def summarise(df, key):
        out = []
        for (k, v), g in df.groupby(["label", key]):
            r = {"label": k, key: v, "n_repos": len(g),
                 "n_hist_median": float(g.n.median()) if "n" in g else np.nan,
                 "pos_hist_median": float(g.pos_hist.median()), "pos_hist_min": int(g.pos_hist.min()),
                 "repos_pos0": int((g.pos_hist == 0).sum()), "repos_pos_lt5": int((g.pos_hist < 5).sum()),
                 "repos_pos_lt10": int((g.pos_hist < 10).sum()),
                 "oneclass_median": float(g.share_oneclass.median()), "oneclass_max": float(g.share_oneclass.max()),
                 "b0_med_median": float(g.b0_med.median()), "b0_w90_median": float(g.b0_w90.median()),
                 "b1_med_median": float(g.b1_med.median()), "b1_q05_median": float(g.b1_q05.median()),
                 "b1_q95_median": float(g.b1_q95.median()), "b1_w90_median": float(g.b1_w90.median()),
                 "b1_le0_median": float(g.share_b1_le0.median()), "b1_le0_max": float(g.share_b1_le0.max()),
                 "repos_b1_le0_gt5pct": int((g.share_b1_le0 > 0.05).sum()),
                 "share_repos_b1_le0_le5pct": float((g.share_b1_le0 <= 0.05).mean()),
                 "sep_median": float(g.share_sep.median()), "sep_max": float(g.share_sep.max()),
                 "pt_oneclass": int(g.pt_oneclass.sum()), "pt_b1_le0": int((g.pt_b1 <= 0).sum()),
                 "s0_w90_median": float(g.s0_w90.median()),
                 "ece_raw_mean": float(g.ece_raw.mean()), "ece_full_mean": float(g.ece_full_med.mean()),
                 "ece_shift_mean": float(g.ece_shift_med.mean()), "ece_ref_mean": float(g.ref_ece_cal.mean()),
                 "ece_full_q95_mean": float(g.ece_full_q95.mean()), "ece_shift_q95_mean": float(g.ece_shift_q95.mean()),
                 "brier_raw_mean": float(g.brier_raw.mean()), "brier_full_mean": float(g.brier_full_med.mean()),
                 "brier_shift_mean": float(g.brier_shift_med.mean()),
                 "worse_full_median": float(g.share_ece_worse_full.median()),
                 "pt_ece_full_mean": float(g.pt_ece_full.fillna(g.ece_raw).mean()),
                 "pt_ece_shift_mean": float(g.pt_ece_shift.fillna(g.ece_raw).mean())}
            r["K1"] = r["oneclass_median"] <= 0.05
            r["K2"] = r["share_repos_b1_le0_le5pct"] >= 0.90
            r["K3"] = r["b1_w90_median"] <= 1.0
            r["K4"] = (r["ece_full_mean"] < r["ece_raw_mean"]) and (r["ece_full_mean"] - r["ece_ref_mean"] <= 0.02)
            r["stable"] = bool(r["K1"] and r["K2"] and r["K3"] and r["K4"])
            out.append(r)
        return pd.DataFrame(out)

    sm = summarise(per, "n")
    order = {k: i for i, k in enumerate(TGT)}
    sm = sm.sort_values(["label", "n"], key=lambda s: s.map(order) if s.name == "label" else s)
    sm.to_csv(os.path.join(OUT, "summary_by_n.csv"), index=False)
    min_stable = {}
    for k, g in sm.groupby("label"):
        g = g.sort_values("n")
        ok = g.stable.to_numpy()
        cand = [int(n) for i, n in enumerate(g.n) if ok[i:].all()]
        min_stable[k] = cand[0] if cand else None
    sv = summarise(var[var.reached == 1], "m_pos") if len(var) and (var.reached == 1).any() else pd.DataFrame()
    if len(sv):
        reach = var.groupby(["label", "m_pos"]).agg(n_repos_total=("reached", "size"),
                                                    n_repos_reached=("reached", "sum")).reset_index()
        sv = sv.merge(reach, on=["label", "m_pos"], how="left")
        sv["n_hist_q90"] = [float(var[(var.label == r.label) & (var.m_pos == r.m_pos) & (var.reached == 1)].n.quantile(.9))
                            for r in sv.itertuples()]
    sv.to_csv(os.path.join(OUT, "summary_pos_target.csv"), index=False)

    try:
        commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "?"
    info = {"script": "revision/r3_calib_stability.py", "commit": commit, "source": os.path.relpath(V2, ROOT), "seed": SEED, "B": B, "ns": NS,
            "pos_targets": POS_TARGETS, "C": C_REG, "sep_threshold_b1": SEP, "bins": BINS,
            "criterion": {"K1": "median share of one-class bootstrap samples <= 0.05",
                          "K2": ">= 90% of repos have share(b1 <= 0) <= 0.05",
                          "K3": "median over repos of 90% bootstrap interval width of b1 <= 1.0",
                          "K4": "mean ECE after < mean ECE before and <= ECE after (whole earlier half) + 0.02"},
            "min_stable_n": min_stable, "python": sys.version.split()[0],
            "numpy": np.__version__, "pandas": pd.__version__, "seconds": round(time.time() - t0)}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print(json.dumps(min_stable))


if __name__ == "__main__":
    main()
