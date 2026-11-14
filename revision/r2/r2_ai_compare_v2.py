#!/usr/bin/env python3
"""Chapter 2 (section 2.5.3, appendix B.5): PRs with and without an AI assistant review (corpus v2).

Descriptive comparison within each repository, main-sample PRs created in
2025-2026 only (assistants were almost absent before). A repository is included
if each group has at least MIN_GROUP PRs.

Measures: iterations over human events (n_it_human, literal rule), iterations
over the events of all accounts except the author (n_it_all_lit, the same rule
with a different account function), time to merge t_m (clock D), final added
lines, time to the first human review t_r (D) and over all accounts.

Methods: two-sided Mann-Whitney U test (scipy), rank-biserial correlation
r_rb = 2*U/(n1*n2) - 1 (positive if values are larger in the assistant group),
percentile 95 % bootstrap interval (B = 2 000, resampling within each group),
Holm correction. Size control: quartiles of added lines within the repository
(tests with their own Holm correction) and a linear probability model of
n_it >= 2 with the assistant indicator, log1p of added lines and changed files
and month fixed effects (and, separately, author fixed effects as well);
standard errors clustered by author.
Before/after enabling: the enabling month is the first month in which the
assistant reviewed at least ENABLE_SHARE of the repository's PRs; six months
before and six after are compared (the enabling month itself is left out).

compare(main) returns the dict for r2_numbers.json (key ai_compare).
The module only defines functions and constants (no side effects on import).
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

MIN_GROUP = 20
N_BOOT = 2000
SEED = 20261031
ENABLE_SHARE = 0.10
QUART_MIN = 10


def rrb(x, y):
    """(r_rb, p): x is the group with the assistant, y the group without it."""
    u, p = mannwhitneyu(x, y, alternative="two-sided")
    return 2.0 * u / (len(x) * len(y)) - 1.0, p


def _rrb_fast(x, y):
    allv = np.concatenate([x, y])
    ranks = pd.Series(allv).rank(method="average").to_numpy()
    r1 = ranks[:len(x)].sum()
    u = r1 - len(x) * (len(x) + 1) / 2.0
    return 2.0 * u / (len(x) * len(y)) - 1.0


def boot_ci(x, y, rng, n=N_BOOT):
    vals = np.empty(n)
    for b in range(n):
        vals[b] = _rrb_fast(rng.choice(x, len(x)), rng.choice(y, len(y)))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj


def ols_cluster(y, X, groups):
    """OLS with cluster-robust covariance (CR1). Returns (beta, se)."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    k = X.shape[1]
    meat = np.zeros((k, k))
    g = pd.Series(groups).astype("category").cat.codes.to_numpy()
    G = g.max() + 1
    S = np.zeros((G, k))
    np.add.at(S, g, X * u[:, None])
    meat = S.T @ S
    n = len(y)
    c = G / (G - 1) * (n - 1) / (n - k) if G > 1 else 1.0
    V = c * XtX_inv @ meat @ XtX_inv
    return beta, np.sqrt(np.maximum(np.diag(V), 0))


def lpm(d, author_fe=False):
    """Difference in the share of n_it >= 2 (percentage points) controlling for size and month; (coef, lo, hi, p)."""
    from scipy.stats import norm
    y = (d["n_it_human"] >= 2).astype(float).to_numpy()
    base = pd.DataFrame({"ai": d["u_ai"].astype(float).to_numpy(),
                         "la": np.log1p(d["final_additions"].clip(lower=0).to_numpy()),
                         "lf": np.log1p(d["final_changed_files"].clip(lower=0).to_numpy())})
    months = pd.get_dummies(d["month"].to_numpy(), drop_first=True, dtype=float)
    X = pd.concat([base.reset_index(drop=True), months.reset_index(drop=True)], axis=1)
    if author_fe:
        g = d["author_login"].fillna("").to_numpy()
        Xd = (X - X.groupby(g).transform("mean")).to_numpy()
        yd = y - pd.Series(y).groupby(g).transform("mean").to_numpy()
        keep = np.abs(Xd).sum(axis=0) > 1e-9
        if not keep[0]:   # assistant participation does not vary within authors
            return None, None, None, None
        Xd = Xd[:, keep]
        beta, se = ols_cluster(yd, Xd, g)
        j = 0
    else:
        Xm = np.column_stack([np.ones(len(y)), X.to_numpy()])
        beta, se = ols_cluster(y, Xm, d["author_login"].fillna("").to_numpy())
        j = 1
    b, s = beta[j], se[j]
    z = b / s if s > 0 else 0.0
    p = 2 * (1 - norm.cdf(abs(z)))
    return 100 * b, 100 * (b - 1.96 * s), 100 * (b + 1.96 * s), float(p)


def compare(main):
    """main: the main sample (pr_table_v2 + r2_pr_extra_v2), one row per PR."""
    rng = np.random.default_rng(SEED)
    d = main[main["created"].dt.year >= 2025].copy()
    d["month"] = d["created"].dt.strftime("%Y-%m")
    rows, pv = [], []
    metrics = [("it_h", "n_it_human"), ("it_all", "n_it_all_lit"), ("tm", "t_m_h_D"),
               ("add", "final_additions")]
    repos = []
    for repo, g in d.groupby("repo"):
        a, b = g[g.u_ai == 1], g[g.u_ai == 0]
        if len(a) >= MIN_GROUP and len(b) >= MIN_GROUP:
            repos.append(repo)
    for repo in repos:
        g = d[d.repo == repo]
        a, b = g[g.u_ai == 1], g[g.u_ai == 0]
        rec = {"repo": repo, "short": repo.split("/")[1], "n_ai": int(len(a)), "n_no": int(len(b))}
        for key, col in metrics:
            x = a[col].dropna().to_numpy(float)
            y = b[col].dropna().to_numpy(float)
            r, p = rrb(x, y)
            lo, hi = boot_ci(x, y, rng)
            rec[f"{key}_med_ai"] = float(np.median(x))
            rec[f"{key}_med_no"] = float(np.median(y))
            rec[f"{key}_r"] = float(r)
            rec[f"{key}_half"] = float((hi - lo) / 2)
            rec[f"{key}_p"] = float(p)
        rec["it2_ai_pct"] = float(100 * (a.n_it_human >= 2).mean())
        rec["it2_no_pct"] = float(100 * (b.n_it_human >= 2).mean())
        rec["add_ratio"] = float(rec["add_med_ai"] / rec["add_med_no"]) if rec["add_med_no"] > 0 else None
        # t_r over humans and over all accounts (only PRs with a human review)
        for key, col in (("tr_h", "t_r_h_D"), ("tr_all", "t_r_all_h_D")):
            x = a.loc[a.t_r_h_D.notna(), col].dropna().to_numpy(float)
            y = b.loc[b.t_r_h_D.notna(), col].dropna().to_numpy(float)
            r, p = rrb(x, y)
            rec[f"{key}_r"] = float(r)
            rec[f"{key}_p"] = float(p)
        # without PRs by AI agents
        ga = g[g.author_group != "ai_agent"]
        x = ga.loc[ga.u_ai == 1, "n_it_human"].to_numpy(float)
        y = ga.loc[ga.u_ai == 0, "n_it_human"].to_numpy(float)
        rec["it_h_noagent_r"], rec["it_h_noagent_p"] = (float(v) for v in rrb(x, y))
        rec["n_agent"] = int((g.author_group == "ai_agent").sum())
        rows.append(rec)
    # Holm for the main tests: repositories x (it_h, tm, add)
    keys = [(i, k) for i in range(len(rows)) for k in ("it_h", "tm", "add")]
    adj = holm([rows[i][f"{k}_p"] for i, k in keys])
    for (i, k), q in zip(keys, adj):
        rows[i][f"{k}_padj"] = float(q)
    for k in ("it_all", "tr_h", "tr_all"):
        adj = holm([r[f"{k}_p"] for r in rows])
        for r, q in zip(rows, adj):
            r[f"{k}_padj"] = float(q)
    # without agents: corrected for the same number of tests as the main ones (one test replaced)
    for r in rows:
        ps = [x[f"{k}_p"] for x in rows for k in ("it_h", "tm", "add") if not (x is r and k == "it_h")]
        r["it_h_noagent_padj"] = float(holm(ps + [r["it_h_noagent_p"]])[-1])

    # quartiles of added lines
    qt = []
    for repo in repos:
        g = d[d.repo == repo].copy()
        g["q"] = pd.qcut(g.final_additions.rank(method="first"), 4, labels=False)
        for q in range(4):
            s = g[g.q == q]
            x = s.loc[s.u_ai == 1, "n_it_human"].to_numpy(float)
            y = s.loc[s.u_ai == 0, "n_it_human"].to_numpy(float)
            if len(x) >= QUART_MIN and len(y) >= QUART_MIN:
                r, p = rrb(x, y)
                qt.append({"repo": repo, "q": q + 1, "r": float(r), "p": float(p)})
    if qt:
        adj = holm([x["p"] for x in qt])
        for x, a in zip(qt, adj):
            x["padj"] = float(a)
    sig = [x for x in qt if x["padj"] < 0.05]

    # regression
    reg = []
    for repo in repos:
        g = d[d.repo == repo]
        c, lo, hi, p = lpm(g)
        ca, loa, hia, pa = lpm(g, author_fe=True)
        reg.append({"repo": repo, "coef": c, "lo": lo, "hi": hi, "p": p,
                    "coef_a": ca, "lo_a": loa, "hi_a": hia, "p_a": pa})
    sig_reg = [r for r in reg if r["p"] < 0.05 and r["coef"] > 0]
    sig_reg_a = [r for r in reg if r["p_a"] is not None and r["p_a"] < 0.05 and r["coef_a"] > 0]

    # before/after enabling
    ba = []
    allm = main.copy()
    allm["month"] = allm["created"].dt.to_period("M")
    for repo in repos:
        g = allm[allm.repo == repo]
        sh = g.groupby("month").u_ai.mean()
        on = sh[sh >= ENABLE_SHARE]
        if on.empty:
            continue
        m0 = on.index[0]
        before = g[(g.month >= m0 - 6) & (g.month < m0)]
        after = g[(g.month > m0) & (g.month <= m0 + 6)]
        if len(before) < MIN_GROUP or len(after) < MIN_GROUP or (m0 - 6) < g.month.min():
            continue
        ba.append({"repo": repo, "short": repo.split("/")[1], "month": str(m0),
                   "before_pct": float(100 * (before.n_it_human >= 2).mean()),
                   "after_pct": float(100 * (after.n_it_human >= 2).mean()),
                   "n_before": int(len(before)), "n_after": int(len(after))})

    ratio = [r["it_all_r"] / r["it_h_r"] for r in rows if r["it_h_r"] > 0.02 and r["it_h_padj"] < 0.05]
    out = {
        "n_repos": len(rows),
        "rows": rows,
        "it_more_n": sum(r["it_h_r"] > 0 and r["it_h_padj"] < 0.05 for r in rows),
        "it_r_min": min(r["it_h_r"] for r in rows),
        "it_r_max": max(r["it_h_r"] for r in rows),
        "add_more_n": sum(r["add_r"] > 0 and r["add_padj"] < 0.05 for r in rows),
        "tm_more_n": sum(r["tm_r"] > 0 and r["tm_padj"] < 0.05 for r in rows),
        "tm_less_n": sum(r["tm_r"] < 0 and r["tm_padj"] < 0.05 for r in rows),
        "it_all_more_n": sum(r["it_all_r"] > 0 and r["it_all_padj"] < 0.05 for r in rows),
        "tr_h_later_n": sum(r["tr_h_r"] > 0 and r["tr_h_padj"] < 0.05 for r in rows),
        "tr_h_earlier_n": sum(r["tr_h_r"] < 0 and r["tr_h_padj"] < 0.05 for r in rows),
        "tr_all_earlier_n": sum(r["tr_all_r"] < 0 and r["tr_all_padj"] < 0.05 for r in rows),
        "ratio_min": float(min(ratio)) if ratio else None,
        "ratio_max": float(max(ratio)) if ratio else None,
        "ratio_n": len(ratio),
        "quart_n": len(qt), "quart_sig": len(sig),
        "quart_sig_pos": sum(x["r"] > 0 for x in sig),
        "quart_r_min": min((x["r"] for x in sig), default=None),
        "quart_r_max": max((x["r"] for x in sig), default=None),
        "quart_rows": qt,
        "reg": reg, "reg_sig_n": len(sig_reg), "reg_sig_a_n": len(sig_reg_a),
        "reg_min": min((r["coef"] for r in sig_reg), default=None),
        "reg_max": max((r["coef"] for r in sig_reg), default=None),
        "reg_a_min": min((r["coef_a"] for r in sig_reg_a), default=None),
        "reg_a_max": max((r["coef_a"] for r in sig_reg_a), default=None),
        "before_after": ba,
        "ba_up_n": sum(r["after_pct"] > r["before_pct"] for r in ba),
        "ba_n": len(ba),
    }
    return out
