import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adv_calib as A
W = pd.read_csv(os.path.join(A.OUT, "whole_half.csv"))
rng = np.random.default_rng(1)
DIFF, DEC = [], []  # written to reliab_diff.csv and reliab_st_deciles.csv
for k, g in W.groupby("label"):
    a, b = g.prior_ece10.to_numpy(), g.plattE_ece10.to_numpy()
    bb = [np.mean((a - b)[rng.integers(0, len(a), len(a))]) for _ in range(10000)]
    a2, b2 = g.prior_brier.to_numpy(), g.constE_brier.to_numpy()
    bs = [np.mean((a2 - b2)[rng.integers(0, len(a2), len(a2))]) for _ in range(10000)]
    print(k, f"ECE prior - Platt(whole early) = {np.mean(a-b):.4f} [{np.quantile(bb,.025):.4f}; {np.quantile(bb,.975):.4f}]",
          f"| Brier prior - const early rate = {np.mean(a2-b2):.4f} [{np.quantile(bs,.025):.4f}; {np.quantile(bs,.975):.4f}]")
    DIFF.append({"label": k, "n_repos": len(a), "ece_prior_minus_plattE": float(np.mean(a - b)),
                 "ece_lo": float(np.quantile(bb, .025)), "ece_hi": float(np.quantile(bb, .975)),
                 "brier_prior_minus_constE": float(np.mean(a2 - b2)),
                 "brier_lo": float(np.quantile(bs, .025)), "brier_hi": float(np.quantile(bs, .975))})
pd.DataFrame(DIFF).to_csv(os.path.join(A.OUT, "reliab_diff.csv"), index=False)
# reliability of st after prior correction, pooled later halves, deciles of p
t, m = A.load(); d = A.preds(t, m, "ST")
rows = []
for repo, g in d.groupby("test"):
    g = g.sort_values("created", kind="stable"); late = g.iloc[len(g) // 2:]
    pi = float(g.pos_train.iloc[0])
    rows.append(pd.DataFrame({"repo": repo, "y": late.y.to_numpy(float), "p": A.sig(A.logit(late.prob.to_numpy()) + np.log(pi / (1 - pi)))}))
L = pd.concat(rows)
L["bin"] = pd.qcut(L.p, 10, labels=False, duplicates="drop")
def wil(k, n, z=1.96):
    ph = k / n; den = 1 + z*z/n; c = (ph + z*z/(2*n)) / den; h = z*np.sqrt(ph*(1-ph)/n + z*z/(4*n*n)) / den
    return c - h, c + h
print("st reliability, pooled later halves, prior correction (decile, n, mean p, observed [95% Wilson]); cluster CI not shown")
for b, g in L.groupby("bin"):
    lo, hi = wil(g.y.sum(), len(g))
    print(f"  {int(b)}  n={len(g)}  p={g.p.mean():.4f}  obs={g.y.mean():.4f} [{lo:.4f}; {hi:.4f}]  O/E={g.y.mean()/g.p.mean():.2f}")
    DEC.append({"decile": int(b), "n": len(g), "mean_p": float(g.p.mean()), "obs": float(g.y.mean()),
                "obs_lo": float(lo), "obs_hi": float(hi), "oe": float(g.y.mean() / g.p.mean())})
# cluster bootstrap over repos for the top decile O/E
top = L[L.bin == L.bin.max()]
reps = top.repo.unique()
oe = []
for _ in range(2000):
    s = rng.choice(reps, len(reps)); x = pd.concat([top[top.repo == r] for r in s])
    oe.append(x.y.mean() / x.p.mean())
print("top decile O/E cluster-bootstrap 95%:", np.round(np.quantile(oe, [.025, .975]), 2))
DEC = pd.DataFrame(DEC)
DEC["top_oe_cluster_lo"] = np.nan; DEC["top_oe_cluster_hi"] = np.nan
DEC.loc[DEC.decile == DEC.decile.max(), ["top_oe_cluster_lo", "top_oe_cluster_hi"]] = np.quantile(oe, [.025, .975])
DEC.to_csv(os.path.join(A.OUT, "reliab_st_deciles.csv"), index=False)
