"""Rolling origin over ALL histories (no fixed n): at each origin the history is every PR
merged before it, as in the method. Rules compared by proper scores:
  prior    - class-weight correction only
  r200     - Platt if |H| >= 200, >= 10 of each class, b1 > 0; else prior
  epv20    - Platt if minority >= 20 and b1 > 0; else prior
  staged   - Platt if minority >= 20 and b1 > 0; shift if minority >= 10; else prior
  shift10  - shift if minority >= 10; else prior
  v100     - Platt if minority >= 100 and b1 > 0; else prior (m100 in r4_e5_numbers.py)
  v100s    - as v100, else shift if |H| >= 30 and minority >= 10; else prior
  s30      - shift if |H| >= 30 and minority >= 10; else prior
Origins every 5 % of a repo's PRs from 5 % on, so short histories are present.
Also: prior correction on the first PRs of each project (new-project period)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adv_calib as A
t, m = A.load()
out, first = [], []
for k, tg in A.TGT.items():
    d = A.preds(t, m, tg)
    for repo, g in d.groupby("test"):
        g = g.sort_values("created", kind="stable").reset_index(drop=True)
        pi = float(g.pos_train.iloc[0]); off = np.log(pi / (1 - pi))
        N = len(g)
        # new-project period: first 200 PRs and first 10 % by creation, prior correction only
        for lab, sl in (("first200", g.iloc[:200]), ("first10pct", g.iloc[:max(30, N // 10)])):
            y = sl.y.to_numpy(float); p = A.sig(A.logit(sl.prob.to_numpy()) + off)
            first.append({"label": k, "repo": repo, "part": lab, "n": len(y), "rate": y.mean(), "meanp": p.mean(),
                          "oe": y.mean() / p.mean(), "ece": A.ece_eqcount(y, p), "brier": np.mean((p - y) ** 2),
                          "brier_constpi": np.mean((pi - y) ** 2)})
        cuts = [int(N * q) for q in np.arange(0.05, 1.0001, 0.05)]
        for j in range(len(cuts) - 1):
            lo, hi = cuts[j], cuts[j + 1]
            t0 = g.created.iloc[lo]
            H = g.iloc[:lo]; H = H[H.merged < t0]
            blk = g.iloc[lo:hi]
            xb, yb = blk.prob.to_numpy(), blk.y.to_numpy(float)
            base = A.sig(A.logit(xb) + off)
            yh = H.y.to_numpy(float); ph = H.prob.to_numpy()
            mino = int(min(yh.sum(), len(yh) - yh.sum())) if len(yh) else 0
            fit = A.platt(ph, yh) if len(yh) >= 2 else None
            s0 = A.shift(ph, yh) if len(yh) >= 2 else None
            pl = A.sig(fit[0] + fit[1] * A.logit(xb)) if fit else base
            ps = A.sig(s0 + A.logit(xb)) if s0 is not None else base
            b1ok = fit is not None and fit[1] > 0
            r = {"prior": base,
                 "r200": pl if (len(yh) >= 200 and mino >= 10 and b1ok) else base,
                 "epv20": pl if (mino >= 20 and b1ok) else base,
                 "staged": pl if (mino >= 20 and b1ok) else (ps if mino >= 10 else base),
                 "shift10": ps if mino >= 10 else base, "v100": pl if (mino >= 100 and b1ok) else base, "v100s": pl if (mino >= 100 and b1ok) else (ps if (len(yh) >= 30 and mino >= 10) else base), "s30": ps if (len(yh) >= 30 and mino >= 10) else base}
            # bucket by history length at the origin
            hb = "<30" if len(yh) < 30 else "30-99" if len(yh) < 100 else "100-199" if len(yh) < 200 else ">=200"
            for rule, p in r.items():
                out.append({"label": k, "repo": repo, "hb": hb, "rule": rule, "n": len(yb),
                            "sse": float(((p - yb) ** 2).sum()),
                            "ll": float(-(yb * np.log(np.clip(p, 1e-6, 1)) + (1 - yb) * np.log(np.clip(1 - p, 1e-6, 1))).sum()),
                            "sum_p": float(p.sum()), "sum_y": float(yb.sum())})
O = pd.DataFrame(out)
O.to_csv(os.path.join(A.OUT, "rules_rolling2.csv"), index=False)
F = pd.DataFrame(first); F.to_csv(os.path.join(A.OUT, "first_prs2.csv"), index=False)
pd.set_option("display.width", 250)
# per repo Brier / log-loss, then mean over repos; by history bucket and overall
def agg(df):
    r = df.groupby(["label", "repo", "rule"]).agg(n=("n", "sum"), sse=("sse", "sum"), ll=("ll", "sum")).reset_index()
    r["brier"] = r.sse / r.n; r["logloss"] = r.ll / r.n
    return r
a = agg(O)
piv = a.pivot_table(index=["label", "repo"], columns="rule", values=["brier", "logloss"])
print("ALL ORIGINS (mean over repos):")
print(a.groupby(["label", "rule"])[["brier", "logloss"]].mean().unstack().round(4).to_string())
for col in ("r200", "epv20", "v100", "v100s", "s30"):
    dl = (piv[("logloss", col)] - piv[("logloss", "prior")]).groupby(level=0)
    print(col, "repos where log-loss worse than prior:", dl.apply(lambda v: int((v > 1e-9).sum())).to_dict())
print("BY HISTORY BUCKET (mean over repos):")
b = O.groupby(["label", "hb", "repo", "rule"]).agg(n=("n", "sum"), sse=("sse", "sum"), ll=("ll", "sum")).reset_index()
b["brier"] = b.sse / b.n; b["logloss"] = b.ll / b.n
print(b.groupby(["label", "hb", "rule"])["logloss"].mean().unstack().round(4).to_string())
print("FIRST PRs, prior correction:")
F["bss"] = 1 - F.brier / F.brier_constpi
print(F.groupby(["label", "part"]).agg(oe_med=("oe", "median"), oe_p10=("oe", lambda v: v.quantile(.1)), oe_p90=("oe", lambda v: v.quantile(.9)),
      out_08_125=("oe", lambda v: int(((v < .8) | (v > 1.25)).sum())), out_05_2=("oe", lambda v: int(((v < .5) | (v > 2)).sum())),
      ece=("ece", "mean"), rate_mean=("rate", "mean"), meanp=("meanp", "mean"), bss_vs_constpi=("bss", "median")).round(3).to_string())
