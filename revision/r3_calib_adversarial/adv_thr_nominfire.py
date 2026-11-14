"""Threshold by eq:threshold WITHOUT the 20-firing floor (as written in chapter 2),
on the last n merged PRs before the later half; plus sklearn re-fit of the point b1 of
r3_calib_stability.py on the same histories."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adv_calib as A
t, m = A.load()
rows, chk = [], []
per = pd.read_csv(os.path.join(A.R3, "output", "r3", "calib_stability", "per_repo_by_n.csv"))
for k, tg in A.TGT.items():
    d = A.preds(t, m, tg)
    for repo, g in d.groupby("test"):
        g = g.sort_values("created", kind="stable")
        h = len(g) // 2
        early, late = g.iloc[:h], g.iloc[h:]
        tb = late.created.iloc[0]
        pool = early[early.merged < tb].sort_values("merged", kind="stable")
        pl, yl = late.prob.to_numpy(), late.y.to_numpy(float)
        for n in (30, 50, 100, 200):
            if len(pool) < n:
                continue
            hh = pool.iloc[-n:]
            ab = A.platt(hh.prob.to_numpy(), hh.y.to_numpy(float))
            r = per[(per.label == k) & (per.repo == repo) & (per.n == n)]
            if len(r):
                chk.append({"label": k, "repo": repo, "n": n, "sk_b1": ab[1] if ab else np.nan, "their_b1": r.pt_b1.iloc[0]})
            if k == "st":
                continue
            for c in (0.1, 0.2, 0.3):
                th = A.choose_threshold(hh.prob.to_numpy(), hh.y.to_numpy(float), c, min_fires=1)
                rec = {"label": k, "repo": repo, "n": n, "c": c, "reach": th is not None}
                if th is not None:
                    f = pl > th
                    hf = hh.prob.to_numpy() > th
                    rec.update(fires_hist=int(hf.sum()), prec=float(yl[f].mean()) if f.any() else np.nan, fire=float(f.mean()))
                rows.append(rec)
C = pd.DataFrame(chk)
C["diff"] = (C.sk_b1 - C.their_b1).abs()
print("sklearn vs r3_calib_stability point b1: max |diff| =", C["diff"].max(), " n compared =", C["diff"].notna().sum(),
      " sign disagreements =", int(((C.sk_b1 <= 0) != (C.their_b1 <= 0))[C.sk_b1.notna()].sum()))
print(C.groupby(["label", "n"]).agg(sk_le0=("sk_b1", lambda v: int((v <= 0).sum())), their_le0=("their_b1", lambda v: int((v <= 0).sum()))).T.to_string())
T = pd.DataFrame(rows)
T["below"] = T.prec < 1 - T.c - 1e-12
G = T.groupby(["label", "c", "n"]).agg(reach=("reach", "sum"), fires_hist_med=("fires_hist", "median"),
      prec_med=("prec", "median"), below=("below", "sum"), fire_med=("fire", "median"))
print(G.round(3).to_string())
# written to files (the computation above is unchanged)
C.to_csv(os.path.join(A.OUT, "b1_sklearn_check.csv"), index=False)
T.to_csv(os.path.join(A.OUT, "threshold_nominfire.csv"), index=False)
G.reset_index().to_csv(os.path.join(A.OUT, "threshold_nominfire_summary.csv"), index=False)
