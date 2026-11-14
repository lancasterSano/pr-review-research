#!/usr/bin/env python3
"""Chapter 4, P4: the rollback rule of chapter 2 (2.4.4, 2.4.5, B.4) replayed on the
corpus history.

Label y(c) (symmetric, events only):
  1 – the suggestion was applied, or a revision before the merge changed the commented
      line (isOutdated in raw);
  0 – the PR was merged and the commented line did not change (no resolution needed);
  null – the PR was not merged (does not occur here: all threads are in merged PRs).
The same label for threads started by human reviewers (baseline ybar_H).

Rule (B.4): ratio R = ybar_A / ybar_H; MOVER-R interval (Newcombe 2016; Donner, Zou
2012) from Wilson intervals of each group with n / deff, deff by PR author (Rao-Scott,
cluster = author of the PR, floored at 1); level 1 - alpha / K of every check, alpha =
0.05, K = maximal number of cycles fixed in advance; a check runs only if each group has
labelled examples in >= n_f PRs and n / deff >= n_f; rollback if the upper bound < kappa,
keep if the lower bound >= kappa; the first decision is final; no decision within K
cycles -> keep (by default).

Corpus replay:
  - cycles = quarters (an example joins the quarter its PR was merged); the pilot of a
    repository starts at its first quarter with an assistant thread; examples of both
    groups accumulate from that quarter;
  - K = 8: the number of quarters of the corpus window (2024Q4–2026Q3), the same for all
    repositories and fixed before looking at the data; no repository has more cycles;
  - unit (B.4): one assistant (tool) in a repository – a pilot of its own
    from its first quarter, with its own K cycles; all assistants of a repository together
    is a sensitivity;
  - control group (main): threads of human reviewers in PRs WITHOUT assistant threads in
    the same quarters (B.4: control PRs of the same period). Sensitivity: human threads
    in the same PRs as the assistant;
  - kappa = 0.6 (main), 0.5 and 0.7 (sensitivity); deff by PR (sensitivity).
Power table: Monte Carlo of the rule with cluster sizes from the corpus and author and
PR random effects (logit-normal), for a grid of true ratios and PRs per cycle.

Input (read only): the same as r4_p4_feedback.py, including its raw caches
  revision/output/r4/p4_feedback/raw_thread_flags.csv and raw_thread_flags_base.csv.
Output: revision/output/r4/p4_ratio/ numbers.json (keys e2.p4r.*), by_repo.csv,
  by_repo_quarter.csv, power.csv.

Run from the repository root:
  python revision/r4_calc/r4_p4_ratio.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import r4_p4_feedback as F  # noqa: E402  (paths, raw caches, design_effect, wilson)

OUT = os.path.join(F.REV, "output", "r4", "p4_ratio")
PFX = "e2.p4r."
SEED = 20261114
N_F = F.N_F
ALPHA = 0.05
K = 8
KAPPA = 0.6
KAPPA_SENS = (0.5, 0.7)


# ── data ─────────────────────────────────────────────────────────────────────
def load():
    th = pd.read_csv(F.THREADS)
    pt = pd.read_csv(F.PR_TABLE, usecols=["repo", "pr_number", "merged_at", "author_login"])

    ai = th[th.starter_class == "ai_assistant"].reset_index(drop=True)
    ai["pos"] = ai.groupby(["repo", "pr_number"]).cumcount()
    fl = pd.read_csv(os.path.join(F.OUT, "raw_thread_flags.csv"), keep_default_na=False)
    a = ai.merge(fl.drop(columns="created_at"), on=["repo", "pr_number", "pos"], how="left", validate="1:1")
    assert a.is_outdated.notna().all()
    a["applied"] = a.sugg_applied == 1
    a["group"] = "ai"
    a["tool"] = a["tool"].fillna("").astype(str)

    hb = th[th.starter_class.isin(["human", "other_bot"])].reset_index(drop=True)
    hb["pos"] = hb.groupby(["repo", "pr_number"]).cumcount()   # positions as in raw_flags_base
    flb = pd.read_csv(os.path.join(F.OUT, "raw_thread_flags_base.csv"), keep_default_na=False)
    b = hb.drop(columns=["is_resolved", "commits_after", "has_suggestion", "apply_commit_after",
                         "sugg_applied", "reacted"], errors="ignore")
    b = b.merge(flb, on=["repo", "pr_number", "pos"], how="left", validate="1:1")
    assert b.is_outdated.notna().all()
    b = b.copy()
    b["applied"] = (b.has_suggestion == 1) & (b.apply_commit_after == 1) & (b.author_group == "human")
    b["group"] = b.starter_class.map({"human": "human", "other_bot": "bot"})
    b["tool"] = ""

    a["created_at"] = a["created_at"].astype(str)
    b["created_at"] = b["created_at"].astype(str)
    cols = ["repo", "pr_number", "group", "tool", "applied", "is_outdated", "created_at"]
    x = pd.concat([a[cols], b[cols]], ignore_index=True)
    x = x.merge(pt, on=["repo", "pr_number"], how="left", validate="m:1")
    assert x.merged_at.notna().all(), "threads in unmerged PRs: label null not handled"
    x["y"] = (x.applied | (x.is_outdated.astype(int) == 1)).astype(int)
    x["quarter"] = pd.to_datetime(x.merged_at, utc=True).dt.tz_localize(None).dt.to_period("Q")
    x["author"] = x.author_login.fillna("").astype(str)
    ai_prs = set(map(tuple, x.loc[x.group == "ai", ["repo", "pr_number"]].drop_duplicates().to_numpy()))
    x["pr_has_ai"] = [(r, p) in ai_prs for r, p in zip(x.repo, x.pr_number)]
    return x, len(a), int((b.group == "human").sum())


# ── rule ─────────────────────────────────────────────────────────────────────
def mover_r(p1, l1, u1, p2, l2, u2):
    """MOVER-R interval for p1 / p2 (Donner, Zou 2012; Newcombe 2016)."""
    a = p1 * p2
    dl = a * a - l1 * u2 * (2 * p1 - l1) * (2 * p2 - u2)
    du = a * a - u1 * l2 * (2 * p1 - u1) * (2 * p2 - l2)
    lo = (a - np.sqrt(max(dl, 0.0))) / (u2 * (2 * p2 - u2))
    hi = (a + np.sqrt(max(du, 0.0))) / (l2 * (2 * p2 - l2)) if l2 * (2 * p2 - l2) > 0 else np.inf
    return max(lo, 0.0), hi


def group_stats(g, cluster, z):
    """n, clusters, deff, ybar, Wilson (lo, hi) with n / deff; cluster column given."""
    n = len(g)
    pc = g.groupby(cluster).y.agg(["size", "sum"])
    _, deff = F.design_effect(pc["size"], pc["sum"])
    p = g.y.mean()
    lo, hi = F.wilson(p, n / deff, z)
    return dict(n=n, m=g.pr_number.nunique(), clusters=len(pc), deff=deff, p=p, lo=lo, hi=hi)


def check(ca, ch, cluster, kappa, z):
    """One check: gate per group, ratio interval, state."""
    A, H = group_stats(ca, cluster, z), group_stats(ch, cluster, z)
    gate = all(s["m"] >= N_F and s["n"] / s["deff"] >= N_F for s in (A, H))
    rec = {"n_a": A["n"], "m_a": A["m"], "deff_a": A["deff"], "ybar_a": A["p"],
           "n_h": H["n"], "m_h": H["m"], "deff_h": H["deff"], "ybar_h": H["p"], "gate": gate}
    if not gate:
        return rec, None
    lo, hi = mover_r(A["p"], A["lo"], A["hi"], H["p"], H["lo"], H["hi"])
    st = "rollback" if hi < kappa else ("keep" if lo >= kappa else "undecided")
    rec.update(ratio=A["p"] / H["p"], r_lo=lo, r_hi=hi, state=st)
    return rec, st


def replay(x, kappa=KAPPA, control="other_prs", cluster="author", k_max=K, by_tool=True):
    z = norm.ppf(1 - ALPHA / k_max / 2)
    rows, summ = [], []
    end = x.quarter.max()
    units = []
    for repo, g in x.groupby("repo"):
        gh = g[(g.group == "human") & (~g.pr_has_ai if control == "other_prs" else g.pr_has_ai)]
        gai = g[g.group == "ai"]
        if by_tool:
            units += [(f"{repo}::{t}", gt, gh) for t, gt in gai.groupby("tool")]
        elif len(gai):
            units.append((repo, gai, gh))
    for repo, ga, gh in units:
        q0 = ga.quarter.min()
        qs = pd.period_range(q0, end, freq="Q")[:k_max]
        dec, n_checks, last = None, 0, None
        for k, q in enumerate(qs, start=1):
            ca, ch = ga[ga.quarter <= q], gh[(gh.quarter >= q0) & (gh.quarter <= q)]
            if ca.empty or ch.empty:
                rows.append({"repo": repo, "cycle": k, "quarter": str(q), "gate": False})
                continue
            rec, st = check(ca, ch, cluster, kappa, z)
            rec.update(repo=repo, cycle=k, quarter=str(q))
            rows.append(rec)
            if st is None:
                continue
            n_checks += 1
            last = rec
            if st != "undecided":
                dec = (st, k, rec)
                break
        r = {"repo": repo, "n_cycles": len(qs), "n_checks": n_checks, "first_quarter": str(q0),
             "last_thread": ga.created_at.max()[:10]}
        if dec:
            st, k, rec = dec
            r.update(state=st, cycle_decision=k, quarter_decision=rec["quarter"], n_a=rec["n_a"], m_a=rec["m_a"],
                     n_h=rec["n_h"], m_h=rec["m_h"], neff_a=rec["n_a"] / rec["deff_a"],
                     ratio=rec["ratio"], r_lo=rec["r_lo"], r_hi=rec["r_hi"])
        else:
            r.update(state="keep_default" if n_checks else "no_check")
            if last:
                r.update(ratio=last["ratio"], r_lo=last["r_lo"], r_hi=last["r_hi"])
        tot_a = ga
        tot_h = gh[gh.quarter >= q0]
        r.update(ybar_a_total=tot_a.y.mean(), n_a_total=len(tot_a),
                 ybar_h_total=tot_h.y.mean() if len(tot_h) else np.nan, n_h_total=len(tot_h))
        summ.append(r)
    return pd.DataFrame(rows), pd.DataFrame(summ)


def summarise(sm, pre, num):
    st = sm.state
    dec = sm[st.isin(["rollback", "keep"])]
    num[pre + "n_repos"] = int(len(sm))
    num[pre + "n_units_checked"] = int((sm.n_checks > 0).sum())
    num[pre + "n_distinct_repos_checked"] = int(sm[sm.n_checks > 0].repo.str.split("::").str[0].nunique())
    num[pre + "n_distinct_repos"] = int(sm.repo.str.split("::").str[0].nunique())
    for i, (_, r) in enumerate(sm[st == "rollback"].sort_values("quarter_decision").iterrows(), start=1):
        rp, _, tl = r.repo.partition("::")
        num.update({pre + f"rb{i}.repo": rp.split("/")[1], pre + f"rb{i}.tool": (tl.replace(" code review", "").replace(" (GitHub app)", "") or "усі асистенти"),
                    pre + f"rb{i}.quarter": r.quarter_decision.replace("Q", " Q"), pre + f"rb{i}.ratio": float(r.ratio),
                    pre + f"rb{i}.r_lo": float(r.r_lo), pre + f"rb{i}.r_hi": float(r.r_hi),
                    pre + f"rb{i}.n_a": int(r.n_a), pre + f"rb{i}.m_a": int(r.m_a),
                    pre + f"rb{i}.last_thread": ".".join(reversed(r.last_thread.split("-")))})
    for _, r in sm[sm.repo == "woocommerce/woocommerce"].iterrows():   # by_repo: the early keep
        if isinstance(r.get("quarter_decision"), str):
            num[pre + "woocommerce.quarter"] = r.quarter_decision.replace("Q", " Q")
            num[pre + "woocommerce.state"] = r.state
    for s in ("rollback", "keep", "keep_default", "no_check"):
        num[pre + f"n_repos_{s}"] = int((st == s).sum())
    num[pre + "n_repos_keep_all"] = int(st.isin(["keep", "keep_default", "no_check"]).sum())
    num[pre + "n_repos_decided"] = int(st.isin(["rollback", "keep"]).sum())
    if len(dec):
        num[pre + "median_cycle_decision"] = float(dec.cycle_decision.median())
        num[pre + "dec_n_a_min"] = int(dec.n_a.min())
        num[pre + "dec_m_a_min"] = int(dec.m_a.min())
        num[pre + "dec_n_a_median"] = float(dec.n_a.median())
        num[pre + "dec_m_a_median"] = float(dec.m_a.median())
        num[pre + "dec_neff_a_min"] = float(dec.neff_a.min())
    num[pre + "rollback_repos"] = ", ".join(sorted(sm[st == "rollback"].repo.str.split("/").str[1]))
    num[pre + "undecided_ratio_below_kappa"] = int(((st == "keep_default") & (sm.ratio < KAPPA)).sum())


# ── power ────────────────────────────────────────────────────────────────────
def power(x, rng, n_sim=1000):
    """P(rollback), P(keep), P(no decision) of the rule over K cycles for true ratios."""
    sizes = {g: x[x.group == g].groupby(["repo", "pr_number"]).size().to_numpy() for g in ("ai", "human")}
    p_h = 0.8
    sig_pr, sig_au, n_auth = 1.2, 0.8, 10
    z = norm.ppf(1 - ALPHA / K / 2)

    def intercept(p):  # logit intercept so that the marginal mean is p under the random effects
        e = rng.normal(0, np.hypot(sig_pr, sig_au), 200_000)
        lo, hi = -10.0, 10.0
        for _ in range(60):
            mid = (lo + hi) / 2
            m = (1 / (1 + np.exp(-(mid + e)))).mean()
            lo, hi = (mid, hi) if m < p else (lo, mid)
        return (lo + hi) / 2

    rows = []
    for m_cyc in (15, 40):
        for ratio in (0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
            b = {"ai": intercept(ratio * p_h), "human": intercept(p_h)}
            out = {"rollback": 0, "keep": 0, "none": 0}
            cyc = []
            for _ in range(n_sim):
                eff_au = {g: rng.normal(0, sig_au, n_auth) for g in b}
                acc = {g: {"y": [], "au": []} for g in b}
                res = None
                for k in range(1, K + 1):
                    for g in b:
                        au = rng.integers(0, n_auth, m_cyc)
                        sz = rng.choice(sizes[g], m_cyc)
                        pr = rng.normal(0, sig_pr, m_cyc)
                        lin = np.repeat(b[g] + eff_au[g][au] + pr, sz)
                        acc[g]["y"].append(rng.random(lin.size) < 1 / (1 + np.exp(-lin)))
                        acc[g]["au"].append(np.repeat(au, sz))
                    st = {}
                    for g in b:
                        y = np.concatenate(acc[g]["y"]).astype(float)
                        au = np.concatenate(acc[g]["au"])
                        s = np.bincount(au, minlength=n_auth)
                        t = np.bincount(au, weights=y, minlength=n_auth)
                        keep = s > 0
                        _, deff = F.design_effect(s[keep], t[keep])
                        p = y.mean()
                        lo, hi = F.wilson(p, y.size / deff, z)
                        st[g] = (p, lo, hi, y.size / deff, k * m_cyc)
                    if not all(v[4] >= N_F and v[3] >= N_F for v in st.values()):
                        continue
                    (pa, la, ua, _, _), (ph, lh, uh, _, _) = st["ai"], st["human"]
                    lo, hi = mover_r(pa, la, ua, ph, lh, uh)
                    if hi < KAPPA:
                        res = ("rollback", k)
                        break
                    if lo >= KAPPA:
                        res = ("keep", k)
                        break
                if res:
                    out[res[0]] += 1
                    cyc.append(res[1])
                else:
                    out["none"] += 1
            rows.append({"prs_per_cycle": m_cyc, "true_ratio": ratio, "p_rollback": out["rollback"] / n_sim,
                         "p_keep": out["keep"] / n_sim, "p_no_decision": out["none"] / n_sim,
                         "median_cycle": float(np.median(cyc)) if cyc else np.nan})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    x, n_ai_threads, n_h_threads = load()
    num = {PFX + "n_threads_ai": n_ai_threads, PFX + "n_threads_human": n_h_threads,
           PFX + "K": K, PFX + "kappa": KAPPA, PFX + "alpha": ALPHA, PFX + "n_f": N_F,
           PFX + "rule": ("MOVER-R для ȳ_A/ȳ_H з Вілсона з n/deff (deff за автором запиту), рівень 1 − α/K, "
                          "умова перевірки ≥ n_f запитів і n/deff ≥ n_f у кожній групі, без рішення за K – лишити"),
           PFX + "label": "1 – пропозицію застосовано або рядок змінено до злиття; 0 – злито, рядок не змінено"}
    for g in ("ai", "human", "bot"):
        num[PFX + f"ybar_pooled.{g}"] = float(x[x.group == g].y.mean())
        num[PFX + f"n_threads_pooled.{g}"] = int((x.group == g).sum())
    num[PFX + "ybar_pooled.human_other_prs"] = float(x[(x.group == "human") & ~x.pr_has_ai].y.mean())
    num[PFX + "ybar_pooled.human_same_prs"] = float(x[(x.group == "human") & x.pr_has_ai].y.mean())
    num[PFX + "ratio_pooled"] = num[PFX + "ybar_pooled.ai"] / num[PFX + "ybar_pooled.human_other_prs"]
    # same-PR ratio as in the opponent's check: mean over PRs with both groups
    both = x[x.group.isin(["ai", "human"])].groupby(["repo", "pr_number", "group"]).y.mean().unstack()
    both = both.dropna()
    num[PFX + "same_pr.n_prs"] = int(len(both))
    num[PFX + "same_pr.ratio_of_means"] = float(both["ai"].mean() / both["human"].mean())

    bq, sm = replay(x)                       # main: pairs repository x assistant (B.4)
    summarise(sm, PFX + "main.", num)
    variants = {f"kappa{str(k).replace('.', '')}": dict(kappa=k) for k in KAPPA_SENS}
    variants.update({"same_prs": dict(control="same_prs"), "deff_pr": dict(cluster="pr_number"),
                     "by_repo": dict(by_tool=False)})
    for name, kw in variants.items():
        _, s2 = replay(x, **kw)
        summarise(s2, PFX + name + ".", num)
        if name == "by_repo":
            s2.to_csv(os.path.join(OUT, "by_repo_all_tools.csv"), index=False, float_format="%.4f")
    # per repository point ratio at the end of the window (descriptive)
    ok = sm.dropna(subset=["ybar_h_total"])
    r_end = ok.ybar_a_total / ok.ybar_h_total
    num[PFX + "repo_ratio_end.min"] = float(r_end.min())
    num[PFX + "repo_ratio_end.median"] = float(r_end.median())
    num[PFX + "repo_ratio_end.n_below_kappa"] = int((r_end < KAPPA).sum())

    pw = power(x, rng)
    for _, r in pw.iterrows():
        key = PFX + f"power.m{int(r.prs_per_cycle)}.r{str(r.true_ratio).replace('.', '')}."
        for c in ("p_rollback", "p_keep", "p_no_decision", "median_cycle"):
            num[key + c] = None if pd.isna(r[c]) else float(r[c])

    bq.to_csv(os.path.join(OUT, "by_repo_quarter.csv"), index=False, float_format="%.4f")
    sm.to_csv(os.path.join(OUT, "by_repo.csv"), index=False, float_format="%.4f")
    pw.to_csv(os.path.join(OUT, "power.csv"), index=False, float_format="%.4f")
    with open(os.path.join(OUT, "numbers.json"), "w", encoding="utf-8") as f:
        json.dump(num, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in num.items() if "power" not in k}, ensure_ascii=False, indent=1))
    print(pw.to_string(index=False))


if __name__ == "__main__":
    main()
