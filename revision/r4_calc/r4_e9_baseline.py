#!/usr/bin/env python3
"""Chapter 4, preparation for E9: baseline efficiency Eff ("without the technology").

Definition (chapter 2, item 2.3.4, eq:eff, and appendix B.3):

  Eff(p) = alpha * F_r^tm(t_m(p)) + beta * F_r^nit(n_it(p)) + gamma * stale(p),

F_r = empirical distribution function of the characteristic in project r: the share of
the project's PRs with a smaller value, ties counted with the mid rank, i.e.
  F(x) = (#{h < x} + 0.5 * #{h == x}) / N   over the reference set H, |H| = N.
Characteristics by human events, as in chapters 2 and 3: t_m = t_m_h_D (hours from
readiness t_rd to merge), n_it = n_it_human, stale = stale30_D.
The period efficiency is the mean of Eff(p) over the PRs of the period.

Two normalisations:
  A (rolling, B.3): H = PRs of the same project and sample merged strictly before
    t_0(p) = created_at(p) (expanding window). If |H| < N_MIN = 30, Eff(p) is not
    computed.
  B (fixed on the base period, B.3 last sentence): per project, PRs sorted by
    (created_at, row_id); the earlier half = first n // 2 rows (the split of E8,
    r3_report.rule_thresholds: sort by created, h = len(g) // 2); H = the earlier half,
    Eff is computed for the later half. The earlier half is also scored against itself
    for the check mean F = 0.5 (not reported as Eff).

Sample: PR table r3_b40d (the one behind chapters 2-4), main sample
hait_filter == 1 & in_main == 1. row_id = position in the full table, as in r3_report.
Negative t_m (merge before the readiness event) is clipped to 0 and counted.
Weights: main alpha = beta = gamma = 1/3; each component alone is the component itself.
Diagnostic (not Eff, uses the future): rolling F over PRs CREATED before t_0(p) with their
final values (Dc_*); A minus Dc shows how much the "merged before t_0" window favours fast
PRs (long PRs created earlier are still open at t_0 and missing from the history).

Reconciliation (asserted): corpus.n_main, repo_total.it2_pct, repo_total.stale_pct and
n / it2_pct / stale_pct / tm_med of every row of repo_table in the chapter 4 numbers
(R4_NUMBERS_JSON).

Reads only; writes to revision/output/r4/e9_eff/baseline/:
  by_pr.parquet, by_repo.csv, numbers.json (flat keys e9.baseline.*).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # revision/
R3_ROOT = os.environ.get("R3_ROOT") or os.path.join(REV, "output", "r3_b40d")
TABLE = os.path.join(R3_ROOT, "v2", "pr_table_v2.csv")
# chapter 4 numbers of the dissertation (nested JSON), used only for the reconciliation checks
R4_NUMBERS = os.environ.get("R4_NUMBERS_JSON") or os.path.join(REV, "r4_calc", "r4_numbers.json")
OUT = os.path.join(REV, "output", "r4", "e9_eff", "baseline")
N_MIN = 30
W_MAIN = (1 / 3, 1 / 3, 1 / 3)
COMP = ("ftm", "fnit", "stale")
TOP = (0.10, 0.20)


# ── normalisation ───────────────────────────────────────────────────────────

def mid_cdf(ref, x):
    """F(x) over the reference values: (#{ref < x} + 0.5 #{ref == x}) / len(ref)."""
    ref = np.sort(np.asarray(ref, dtype=float))
    x = np.asarray(x, dtype=float)
    lo = np.searchsorted(ref, x, side="left")
    hi = np.searchsorted(ref, x, side="right")
    return (lo + 0.5 * (hi - lo)) / len(ref)


class _Fenwick:
    def __init__(self, n):
        self.n = n
        self.t = np.zeros(n + 1, dtype=np.int64)

    def add(self, i):          # i is 0-based
        i += 1
        t, n = self.t, self.n
        while i <= n:
            t[i] += 1
            i += i & -i

    def prefix(self, i):       # count of positions 0..i-1
        s, t = 0, self.t
        while i > 0:
            s += t[i]
            i -= i & -i
        return int(s)


def rolling_mid_cdf(hist_time, hist_val, q_time, q_val, n_min=N_MIN):
    """For each query q: F over the history items with hist_time < q_time (strict).

    Returns (F, k): F is NaN where k < n_min; k = size of the history.
    Times are any comparable numbers (int64 ns here).
    """
    hist_time = np.asarray(hist_time)
    hist_val = np.asarray(hist_val, dtype=float)
    q_time = np.asarray(q_time)
    q_val = np.asarray(q_val, dtype=float)
    order = np.argsort(hist_time, kind="mergesort")
    ht, hv = hist_time[order], hist_val[order]
    k = np.searchsorted(ht, q_time, side="left")            # merged strictly before t_0
    uniq = np.unique(np.concatenate([hv, q_val]))
    h_rank = np.searchsorted(uniq, hv)
    q_rank = np.searchsorted(uniq, q_val)
    fw = _Fenwick(len(uniq))
    F = np.full(len(q_val), np.nan)
    inserted = 0
    for qi in np.argsort(k, kind="mergesort"):
        kk = k[qi]
        while inserted < kk:
            fw.add(int(h_rank[inserted]))
            inserted += 1
        if kk < n_min:
            continue
        r = int(q_rank[qi])
        less = fw.prefix(r)
        eq = fw.prefix(r + 1) - less
        F[qi] = (less + 0.5 * eq) / kk
    return F, k


def eff(ftm, fnit, stale, w=W_MAIN):
    a, b, g = w
    return a * np.asarray(ftm) + b * np.asarray(fnit) + g * np.asarray(stale, dtype=float)


# ── data ────────────────────────────────────────────────────────────────────

def load(path=TABLE):
    cols = ["repo", "pr_number", "in_main", "hait_filter", "created_at", "merged_at",
            "t_m_h_D", "n_it_human", "stale30_D"]
    t = pd.read_csv(path, usecols=cols, low_memory=False)
    t["row_id"] = np.arange(len(t))
    s = t[(t.hait_filter == 1) & (t.in_main == 1)].copy()
    s["created"] = pd.to_datetime(s.created_at, utc=True)
    s["merged"] = pd.to_datetime(s.merged_at, utc=True)
    assert s.merged.notna().all(), "main sample must be merged PRs only"
    s["t_m_raw"] = s.t_m_h_D.astype(float)
    s["t_m"] = s.t_m_raw.clip(lower=0.0)
    s["n_it"] = s.n_it_human.astype(float)
    s["stale"] = s.stale30_D.astype(int)
    return s[["row_id", "repo", "pr_number", "created", "merged", "t_m_raw", "t_m",
              "n_it", "stale"]].reset_index(drop=True)


def reconcile(s, path=R4_NUMBERS):
    d = json.load(open(path, encoding="utf-8"))
    out = {"n_main": (len(s), d["corpus"]["n_main"]),
           "it2_pct": (100 * (s.n_it >= 2).mean(), d["repo_total"]["it2_pct"]),
           "stale_pct": (100 * (s.stale == 1).mean(), d["repo_total"]["stale_pct"])}
    assert out["n_main"][0] == out["n_main"][1], out
    for k in ("it2_pct", "stale_pct"):
        assert abs(out[k][0] - out[k][1]) < 1e-9, (k, out[k])
    rt = {r["repo"]: r for r in d["repo_table"]}
    assert set(rt) == set(s.repo.unique()), "repository sets differ"
    maxdiff = 0.0
    for repo, g in s.groupby("repo"):
        e = rt[repo]
        diffs = (len(g) - e["n"], 100 * (g.n_it >= 2).mean() - e["it2_pct"],
                 100 * (g.stale == 1).mean() - e["stale_pct"],
                 g.t_m_raw.median() - e["tm_med"])
        maxdiff = max(maxdiff, *map(abs, diffs))
    assert maxdiff < 1e-9, maxdiff
    out["repo_table_maxdiff"] = maxdiff
    out["n_repos"] = len(rt)
    return out


# ── computation ─────────────────────────────────────────────────────────────

def compute(s, n_min=N_MIN):
    """Per-PR components and Eff under A and B. Independent of the row order of s."""
    parts = []
    for repo, g in s.groupby("repo", sort=True):
        g = g.sort_values(["created", "row_id"], kind="mergesort").copy()
        ht = g.merged.astype("int64").to_numpy()
        qt = g.created.astype("int64").to_numpy()
        g["A_ftm"], g["k_hist"] = rolling_mid_cdf(ht, g.t_m, qt, g.t_m, n_min)
        g["A_fnit"], _ = rolling_mid_cdf(ht, g.n_it, qt, g.n_it, n_min)
        # diagnostic only (uses the future): history = PRs CREATED before t_0(p) with their
        # final values; the gap to A shows the selection of the "merged before t_0" window
        g["Dc_ftm"], _ = rolling_mid_cdf(qt, g.t_m, qt, g.t_m, n_min)
        g["Dc_fnit"], _ = rolling_mid_cdf(qt, g.n_it, qt, g.n_it, n_min)
        h = len(g) // 2
        g["half"] = np.r_[np.zeros(h, int), np.ones(len(g) - h, int)]
        early = g.iloc[:h]
        g["B_ftm"] = mid_cdf(early.t_m, g.t_m)
        g["B_fnit"] = mid_cdf(early.n_it, g.n_it)
        g["n_early"] = h
        parts.append(g)
    d = pd.concat(parts, ignore_index=True)
    d["A_eval"] = d.A_ftm.notna()
    d["A_eff"] = eff(d.A_ftm, d.A_fnit, d.stale)
    d["B_eff"] = eff(d.B_ftm, d.B_fnit, d.stale)
    return d.sort_values("row_id").reset_index(drop=True)


def top_share(x, q):
    """Share of the sum of x given by the top ceil(q * n) values."""
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    if len(x) == 0 or x.sum() == 0:
        return np.nan
    k = int(np.ceil(q * len(x)))
    return float(x[:k].sum() / x.sum())


def summarise(d):
    rows = []
    for repo, g in d.groupby("repo"):
        a = g[g.A_eval]
        b = g[g.half == 1]
        r = {"repo": repo, "n": len(g), "n_A": len(a), "n_A_excl": len(g) - len(a),
             "n_early": int(g.n_early.iloc[0]), "n_B": len(b),
             "stale_pct": 100 * g.stale.mean(), "it2_pct": 100 * (g.n_it >= 2).mean()}
        for v, x in (("A", a), ("B", b)):
            r[f"{v}_eff"] = x[f"{v}_eff"].mean()
            r[f"{v}_ftm"] = x[f"{v}_ftm"].mean()
            r[f"{v}_fnit"] = x[f"{v}_fnit"].mean()
            r[f"{v}_stale"] = x.stale.mean()
            for q in TOP:
                r[f"{v}_top{int(q * 100)}"] = top_share(x[f"{v}_eff"], q)
        al = b[b.A_eval]                          # A on the same later half as B
        r["Alate_eff"] = al.A_eff.mean()
        r["Alate_ftm"] = al.A_ftm.mean()
        r["Alate_fnit"] = al.A_fnit.mean()
        e = g[g.half == 0]                        # B: earlier half against itself
        r["Bself_ftm"] = e.B_ftm.mean()
        r["Bself_fnit"] = e.B_fnit.mean()
        rows.append(r)
    return pd.DataFrame(rows)


def describe(prefix, x, nums):
    x = pd.Series(x, dtype=float).dropna()
    nums[prefix + ".mean"] = float(x.mean())
    nums[prefix + ".sd"] = float(x.std(ddof=1))
    nums[prefix + ".min"] = float(x.min())
    nums[prefix + ".q1"] = float(x.quantile(0.25))
    nums[prefix + ".median"] = float(x.median())
    nums[prefix + ".q3"] = float(x.quantile(0.75))
    nums[prefix + ".max"] = float(x.max())


def numbers(s, d, R, rec):
    P = "e9.baseline."
    n = {}
    n[P + "n_main"] = int(len(s))
    n[P + "n_repos"] = int(R.shape[0])
    n[P + "n_min"] = N_MIN
    n[P + "weights_main"] = "1/3, 1/3, 1/3"
    n[P + "recon.n_main_expected"] = int(rec["n_main"][1])
    n[P + "recon.it2_pct"] = float(rec["it2_pct"][0])
    n[P + "recon.stale_pct"] = float(rec["stale_pct"][0])
    n[P + "recon.repo_table_maxdiff"] = float(rec["repo_table_maxdiff"])
    n[P + "tm_neg.n"] = int((s.t_m_raw < 0).sum())
    n[P + "tm_neg.min_h"] = float(s.t_m_raw.min())
    a = d[d.A_eval]
    b = d[d.half == 1]
    n[P + "A.n_eval"] = int(len(a))
    n[P + "A.n_excluded"] = int((~d.A_eval).sum())
    n[P + "A.pct_excluded"] = float(100 * (~d.A_eval).mean())
    n[P + "A.n_excluded_max_repo"] = int(R.n_A_excl.max())
    n[P + "A.n_excluded_max_repo_name"] = R.loc[R.n_A_excl.idxmax(), "repo"].split("/", 1)[1]
    n[P + "B.n_eval"] = int(len(b))
    n[P + "B.n_early"] = int((d.half == 0).sum())
    for v, x in (("A", a), ("B", b)):
        for c in ("eff", "ftm", "fnit"):
            n[f"{P}{v}.{c}.mean_pr"] = float(x[f"{v}_{c}"].mean())
        n[f"{P}{v}.stale.mean_pr"] = float(x.stale.mean())
        for c in ("eff", "ftm", "fnit", "stale"):
            describe(f"{P}{v}.{c}.repo", R[f"{v}_{c}"], n)
        # concentration of losses: share of the sum of Eff from the top 10 % / 20 % PRs
        for q in TOP:
            t = int(q * 100)
            n[f"{P}{v}.eff.top{t}_share_pooled"] = top_share(x[f"{v}_eff"], q)
            n[f"{P}{v}.eff.top{t}_share_repo_median"] = float(R[f"{v}_top{t}"].median())
            n[f"{P}{v}.eff.top{t}_share_repo_min"] = float(R[f"{v}_top{t}"].min())
            n[f"{P}{v}.eff.top{t}_share_repo_max"] = float(R[f"{v}_top{t}"].max())
            for c, col in (("ftm", f"{v}_ftm"), ("fnit", f"{v}_fnit"), ("stale", "stale")):
                n[f"{P}{v}.{c}.top{t}_share_pooled"] = top_share(x[col], q)
        # composition of the sum: part of the stale term; stale PRs among the top 10 %
        e = x[f"{v}_eff"].to_numpy()
        n[f"{P}{v}.eff.stale_part_of_sum"] = float((x.stale.sum() / 3) / e.sum())
        top = x.loc[x[f"{v}_eff"].rank(method="first", ascending=False) <= np.ceil(0.1 * len(x))]
        n[f"{P}{v}.eff.top10_pct_stale"] = float(100 * top.stale.mean())
        n[f"{P}{v}.eff.top10_min_eff"] = float(top[f"{v}_eff"].min())
        # share of PRs with Eff above 2/3 (at least two components at the top)
        n[f"{P}{v}.eff.pct_gt_2_3"] = float(100 * (x[f"{v}_eff"] > 2 / 3).mean())
    for c in ("eff", "ftm", "fnit"):
        describe(f"{P}Alate.{c}.repo", R[f"Alate_{c}"], n)
    # the "expected" mean of A under stationarity: 1/3 * (0.5 + 0.5 + stale share)
    exp_a = (0.5 + 0.5 + R.A_stale) / 3
    describe(P + "A.eff.repo_minus_stationary", R.A_eff - exp_a, n)
    exp_b = (0.5 + 0.5 + R.B_stale) / 3
    describe(P + "B.eff.repo_minus_stationary", R.B_eff - exp_b, n)
    # between-project spread: how much of it the stale component explains
    for v in ("A", "B"):
        n[f"{P}{v}.eff.repo_var"] = float(R[f"{v}_eff"].var(ddof=1))
        n[f"{P}{v}.eff.repo_var_stale_part"] = float((R[f"{v}_stale"] / 3).var(ddof=1))
        n[f"{P}{v}.eff.repo_spearman_with_stale"] = float(
            R[f"{v}_eff"].rank().corr(R[f"{v}_stale"].rank()))
    n[P + "B.ftm.n_repos_gt_05"] = int((R.B_ftm > 0.5).sum())
    n[P + "B.fnit.n_repos_gt_05"] = int((R.B_fnit > 0.5).sum())
    n[P + "A.ftm.n_repos_gt_05"] = int((R.A_ftm > 0.5).sum())
    n[P + "A.fnit.n_repos_gt_05"] = int((R.A_fnit > 0.5).sum())
    n[P + "AB.eff.repo_spearman"] = float(R.A_eff.rank().corr(R.B_eff.rank()))
    # diagnostic: window "merged before t_0" (A) vs "created before t_0" (uses the future)
    both = d[d.A_eval & d.Dc_ftm.notna()]
    for c in ("ftm", "fnit"):
        n[f"{P}diag.A.{c}.mean_pr_common"] = float(both[f"A_{c}"].mean())
        n[f"{P}diag.created_hist.{c}.mean_pr_common"] = float(both[f"Dc_{c}"].mean())
        rd = both.groupby("repo").apply(lambda g: g[f"A_{c}"].mean() - g[f"Dc_{c}"].mean())
        n[f"{P}diag.A_minus_created.{c}.repo_median"] = float(rd.median())
        n[f"{P}diag.A_minus_created.{c}.n_repos_pos"] = int((rd > 0).sum())
    n[P + "diag.created_hist.ftm.n_repos_gt_05"] = int(
        (both.groupby("repo").Dc_ftm.mean() > 0.5).sum())
    n[P + "check.Bself_ftm_max_abs_dev"] = float((R.Bself_ftm - 0.5).abs().max())
    n[P + "check.Bself_fnit_max_abs_dev"] = float((R.Bself_fnit - 0.5).abs().max())
    return n


def checks(s, d):
    # 1. the earlier half scored against itself: mean F = 0.5 exactly (mid ranks)
    e = d[d.half == 0]
    for c in ("B_ftm", "B_fnit"):
        dev = (e.groupby("repo")[c].mean() - 0.5).abs().max()
        assert dev < 1e-12, (c, dev)
    # 2. permutation of the rows does not change Eff
    perm = s.sample(frac=1.0, random_state=20261115).reset_index(drop=True)
    d2 = compute(perm)
    for c in ("A_eff", "B_eff"):
        assert np.allclose(d[c].to_numpy(), d2[c].to_numpy(), equal_nan=True), c
    # 3. monotone transform of t_m does not change the t_m component
    s3 = s.assign(t_m=np.log1p(s.t_m) ** 3)
    d3 = compute(s3)
    for c in ("A_ftm", "B_ftm"):
        assert np.allclose(d[c].to_numpy(), d3[c].to_numpy(), equal_nan=True), c
    # 4. A: history size >= N_MIN wherever Eff is defined; no PR in its own history
    assert (d.loc[d.A_eval, "k_hist"] >= N_MIN).all()
    return {"bself_mean_05": True, "permutation": True, "monotone_tm": True}


def main():
    t0 = time.time()
    s = load()
    rec = reconcile(s)
    d = compute(s)
    chk = checks(s, d)
    R = summarise(d)
    nums = numbers(s, d, R, rec)
    nums["e9.baseline.checks"] = ", ".join(k for k, v in chk.items() if v)
    os.makedirs(OUT, exist_ok=True)
    cols = ["row_id", "repo", "pr_number", "created", "merged", "half", "k_hist", "A_eval",
            "t_m_raw", "t_m", "n_it", "stale", "A_ftm", "A_fnit", "A_eff",
            "B_ftm", "B_fnit", "B_eff"]
    out = d[cols].rename(columns={"created": "t_0"})
    out.loc[out.half == 0, ["B_eff"]] = np.nan     # B is defined for the later half only
    out.to_parquet(os.path.join(OUT, "by_pr.parquet"), index=False)
    R.to_csv(os.path.join(OUT, "by_repo.csv"), index=False)
    nums = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in nums.items()}
    nums["_meta"] = {"table": os.path.relpath(TABLE, REV), "script": "r4_calc/r4_e9_baseline.py",
                     "date": time.strftime("%Y-%m-%d %H:%M")}
    with open(os.path.join(OUT, "numbers.json"), "w", encoding="utf-8") as f:
        json.dump(nums, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in nums.items() if k != "_meta"}, ensure_ascii=False, indent=1))
    print(f"done in {time.time() - t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()
