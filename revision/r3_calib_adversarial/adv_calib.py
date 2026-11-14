#!/usr/bin/env python3
"""Independent re-check of the chapter 3 calibration on short histories.

Independent of revision/r3_calib_stability.py: Platt via sklearn, shift-only via
a 1-D root, calibration measured several ways (ECE with 3 binnings, spline ICI,
Brier with Murphy decomposition, log-loss, O/E, calibration slope), a
rolling-origin evaluation instead of the bootstrap, and the rule threshold
(choose_threshold, MIN_FIRES = 20) on short histories.

Reads only revision/output/r3_b40d/v2; writes to revision/output/r3/calib_adversarial.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer

warnings.filterwarnings("ignore")
R3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # revision/
V2 = os.environ.get("R3_OUT") or os.path.join(R3, "output", "r3_b40d", "v2")
OUT = os.path.join(R3, "output", "r3", "calib_adversarial")
os.makedirs(OUT, exist_ok=True)
TGT = {"it": "IT", "st": "ST", "tr": "TFR", "tm": "TM"}
EPS = 1e-6
MIN_FIRES = 20


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
    parts = []
    for _, r in m[m.target == tg].iterrows():
        z = np.load(os.path.join(V2, "models", "oof", r.oof_file))
        parts.append(pd.DataFrame({"row_id": z["row_id"], "prob": z["prob"].astype(float), "test": r.test,
                                   "pos_train": float(r.pos_train)}))
    d = pd.concat(parts, ignore_index=True)
    rid = d.row_id.to_numpy()
    d["y"] = t[f"y_{tg}__main_D"].to_numpy()[rid]
    d["created"] = t["created"].to_numpy()[rid]
    d["merged"] = t["merged"].to_numpy()[rid]
    return d[~d.y.isna()].copy()


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def platt(p, y):
    y = np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return None
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def shift(p, y):
    y = np.asarray(y, float)
    if y.min() == y.max():
        return None
    x = logit(p)
    f = lambda a: sig(a + x).sum() - y.sum()
    return brentq(f, -40, 40)


# ── calibration measures ────────────────────────────────────────────────────

def ece_eqcount(y, p, bins=10):
    o = np.argsort(p, kind="stable")
    return float(sum(len(ix) * abs(y[ix].mean() - p[ix].mean()) for ix in np.array_split(o, min(bins, len(p))) if len(ix)) / len(p))


def ece_eqwidth(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((b == k).sum() * abs(y[b == k].mean() - p[b == k].mean()) for k in range(bins) if (b == k).any()) / len(p))


def murphy(y, p, bins=10):
    o = np.argsort(p, kind="stable")
    ybar = y.mean()
    rel = res = 0.0
    for ix in np.array_split(o, min(bins, len(p))):
        if len(ix):
            rel += len(ix) * (p[ix].mean() - y[ix].mean()) ** 2
            res += len(ix) * (y[ix].mean() - ybar) ** 2
    return rel / len(p), res / len(p), ybar * (1 - ybar)


def ici(y, p):
    """Integrated calibration index: mean |p - smoothed observed|, smoothed by a
    logistic regression on a cubic spline of logit p (flexible calibration curve)."""
    if y.min() == y.max():
        return np.nan
    x = logit(p).reshape(-1, 1)
    try:
        st = SplineTransformer(n_knots=4, degree=3, extrapolation="linear")
        X = st.fit_transform(x)
        lr = LogisticRegression(C=1e4, max_iter=2000).fit(X, y.astype(int))
        sm = lr.predict_proba(X)[:, 1]
    except Exception:
        return np.nan
    return float(np.mean(np.abs(p - sm)))


def logloss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def slope_citl(y, p):
    """Calibration slope (y ~ a + b logit p) and calibration-in-the-large
    (intercept with b fixed at 1) on the evaluation data."""
    ab = platt(p, y)
    s = shift(p, y)
    return (ab[1] if ab else np.nan), (s if s is not None else np.nan)


def measures(y, p, prefix):
    rel, res, unc = murphy(y, p)
    sl, citl = slope_citl(y, p)
    return {f"{prefix}ece10": ece_eqcount(y, p, 10), f"{prefix}ece20": ece_eqcount(y, p, 20),
            f"{prefix}ecew10": ece_eqwidth(y, p, 10), f"{prefix}ici": ici(y, p),
            f"{prefix}brier": float(np.mean((p - y) ** 2)), f"{prefix}logloss": logloss(y, p),
            f"{prefix}rel": rel, f"{prefix}res": res, f"{prefix}unc": unc,
            f"{prefix}oe": float(y.mean() / p.mean()) if p.mean() > 0 else np.nan,
            f"{prefix}meanp": float(p.mean()), f"{prefix}slope": sl, f"{prefix}citl": citl}


# ── threshold: the shared rule of r3_report_stats (strict p > theta, eq:rule) ──

sys.path.insert(0, R3)
from r3_report_stats import choose_threshold  # noqa: E402


def main():
    t, m = load()
    A, R, TH = [], [], []
    for k, tg in TGT.items():
        d = preds(t, m, tg)
        for repo, g in sorted(d.groupby("test"), key=lambda z: z[0]):
            g = g.sort_values("created", kind="stable").reset_index(drop=True)
            h = len(g) // 2
            early, late = g.iloc[:h], g.iloc[h:]
            if h < 20 or not len(late):
                continue
            pi = float(g.pos_train.iloc[0])
            yl = late.y.to_numpy(float); pl = late.prob.to_numpy()
            pe, ye = early.prob.to_numpy(), early.y.to_numpy(float)
            ab = platt(pe, ye)
            if ab is None:
                continue
            pp = sig(logit(pl) + np.log(pi / (1 - pi)))      # prior correction (zero project parameters)
            rec = {"label": k, "repo": repo, "n_late": len(late), "rate_late": yl.mean(), "rate_early": ye.mean(),
                   "pi_train": pi, "pos_late": int(yl.sum()), "b0_early": ab[0], "b1_early": ab[1]}
            rec.update(measures(yl, pl, "raw_"))
            rec.update(measures(yl, pp, "prior_"))
            rec.update(measures(yl, sig(ab[0] + ab[1] * logit(pl)), "plattE_"))
            rec.update(measures(yl, np.full_like(yl, pi), "constpi_"))
            rec.update(measures(yl, np.full_like(yl, min(max(ye.mean(), EPS), 1 - EPS)), "constE_"))
            A.append(rec)

            # ── rolling origin: origins at 30 %, 40 %, ..., 90 % of the repo's PRs by creation;
            #    each PR of block j gets the calibration fitted at origin j (history = merged before it)
            N = len(g)
            cuts = [int(N * q) for q in np.arange(0.3, 1.0001, 0.1)]
            for j in range(len(cuts) - 1):
                lo, hi = cuts[j], cuts[j + 1]
                if hi <= lo:
                    continue
                t0 = g.created.iloc[lo]
                pool = g.iloc[:lo]
                pool = pool[pool.merged < t0].sort_values("merged", kind="stable")
                blk = g.iloc[lo:hi]
                xb = blk.prob.to_numpy(); yb = blk.y.to_numpy(float)
                base = sig(logit(xb) + np.log(pi / (1 - pi)))
                for n in [30, 50, 100, 200, 400, 10**9]:
                    if len(pool) < min(n, 30):
                        continue
                    hist = pool.iloc[-n:] if n < 10**9 else pool
                    ph, yh = hist.prob.to_numpy(), hist.y.to_numpy(float)
                    nmin_cls = int(min(yh.sum(), len(yh) - yh.sum()))
                    fit = platt(ph, yh)
                    s0 = shift(ph, yh)
                    out = {"label": k, "repo": repo, "n": n if n < 10**9 else -1, "block": j, "hist_len": len(hist),
                           "minority": nmin_cls, "b1": fit[1] if fit else np.nan,
                           "y": yb, "raw": xb, "prior": base,
                           "platt": sig(fit[0] + fit[1] * logit(xb)) if fit else base,
                           "shift": sig(s0 + logit(xb)) if s0 is not None else base}
                    R.append(out)

            # ── point 4: project threshold on a short history at the start of the later half
            if k in ("it", "tr", "tm"):
                tb = late.created.iloc[0]
                pool = early[early.merged < tb].sort_values("merged", kind="stable")
                for c in (0.1, 0.2, 0.3):
                    for n in [30, 50, 100, 200, -1]:
                        hist = pool.iloc[-n:] if n > 0 else early
                        if n > 0 and len(pool) < n:
                            continue
                        th = choose_threshold(hist.prob.to_numpy(), hist.y.to_numpy(float), c)
                        r = {"label": k, "repo": repo, "c": c, "n": n, "reach": th is not None}
                        if th is not None:
                            f = pl > th
                            r.update(n_fire=int(f.sum()), prec=float(yl[f].mean()) if f.any() else np.nan,
                                     fire_share=float(f.mean()))
                        TH.append(r)
        print(k, "done", flush=True)

    A = pd.DataFrame(A); A.to_csv(os.path.join(OUT, "whole_half.csv"), index=False)
    TH = pd.DataFrame(TH); TH.to_csv(os.path.join(OUT, "threshold_short.csv"), index=False)

    # rolling: pool predictions of all blocks per (label, repo, n), then measure
    rows = []
    rr = pd.DataFrame([{kk: v for kk, v in r.items() if kk not in ("y", "raw", "prior", "platt", "shift")} for r in R])
    rr.to_csv(os.path.join(OUT, "rolling_fits.csv"), index=False)
    from collections import defaultdict
    acc = defaultdict(lambda: defaultdict(list))
    for r in R:
        key = (r["label"], r["repo"], r["n"])
        for f in ("y", "raw", "prior", "platt", "shift"):
            acc[key][f].append(r[f])
        # rules: r200 (n>=200 & >=10 each class & b1>0) and EPV-20 (minority >= 20 & b1 > 0)
        ok200 = r["hist_len"] >= 200 and r["minority"] >= 10 and r["b1"] > 0
        okepv = r["minority"] >= 20 and r["b1"] > 0
        acc[key]["rule200"].append(r["platt"] if ok200 else r["prior"])
        acc[key]["ruleEPV20"].append(r["platt"] if okepv else r["prior"])
        acc[key]["b1le0"].append(np.array([r["b1"] <= 0 if r["b1"] == r["b1"] else False]))
        acc[key]["oneclass"].append(np.array([r["b1"] != r["b1"]]))
    for (k, repo, n), v in acc.items():
        y = np.concatenate(v["y"])
        rec = {"label": k, "repo": repo, "n": n, "n_eval": len(y), "rate": y.mean(),
               "fits_b1le0": int(np.concatenate(v["b1le0"]).sum()), "fits_oneclass": int(np.concatenate(v["oneclass"]).sum()),
               "fits": len(v["y"])}
        for f in ("raw", "prior", "platt", "shift", "rule200", "ruleEPV20"):
            p = np.concatenate(v[f])
            rec[f"{f}_ece10"] = ece_eqcount(y, p); rec[f"{f}_brier"] = float(np.mean((p - y) ** 2))
            rec[f"{f}_logloss"] = logloss(y, p); rec[f"{f}_oe"] = float(y.mean() / p.mean())
            rec[f"{f}_ici"] = ici(y, p)
        rows.append(rec)
    RO = pd.DataFrame(rows); RO.to_csv(os.path.join(OUT, "rolling_measures.csv"), index=False)
    print("written", OUT)


if __name__ == "__main__":
    main()
