#!/usr/bin/env python3
"""Chapter 4, E9 extension added AFTER the E9 results were seen;
its variants, seeds, measures and criterion were fixed in the author's notes before this calculation.

  B1  the same number k of requests as rule R1/R2 (c = 0.2) in every project and variant, chosen
      by a simple t0 heuristic instead of the forecast P_it: largest change size
      (lines_added_t0 + lines_deleted_t0), most changed files (commit_files_t0), newest author
      (-author_pr_count). Missing value = lowest priority; ties by a fixed random stream.
      Criterion (named in advance): lower bound of the 95 % cluster bootstrap interval of
      G_model / G_size > 1. The other heuristics are reported without a criterion.
  B3  sensitivity of the bound: on the requests with an R1/R2 action and n_it >= 2 also
      t_m -> min(t_m, Q_r), Q_r = median t_m of the history requests with n_it = 1. No criterion.
  descr  descriptive numbers of the existing E9 result on the SAME E9 variants (seed 20261115):
      the oracle ceiling (all n_it >= 2 -> 1), G as a share of it, and the realised precision of
      R1/R2 on the next period. Reproduces e9.shuf.it.{c}.w3.A_pct / Arand_pct exactly (else stop).

B1 and B3: scheme S of E9 (200 permutations per project) on the new seed 20261116, c = 0.2,
weights 1/3, the same Eff, normalisation, choose_threshold and removal as E9.

    PY=python   # interpreter with requirements.txt
    $PY revision/r4_calc/r4_e9c.py --dev    # seed 1, 3 projects, 10 (B) / 20 (descr) variants
    $PY revision/r4_calc/r4_e9c.py          # main -> revision/output/r4/e9_eff/*_e9c.*
"""
import argparse
import json
import multiprocessing as mp
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
from r4_e9b import cluster_summary, pct  # noqa: E402

SEED_C = 20261116
STREAM_TIE = 7
C_MAIN = 0.2
HEUR = ("size", "files", "newauth")
HEUR_COLS = ["lines_added_t0", "lines_deleted_t0", "commit_files_t0", "author_pr_count"]
N_PROC = max(1, (os.cpu_count() or 2) - 2)


# ── selection ───────────────────────────────────────────────────────────────

def top_k(score, u, k):
    """Boolean mask of the k rows with the highest score; NaN = lowest priority; ties by the
    random key u (smaller u first)."""
    s = np.asarray(score, dtype=float)
    s = np.where(np.isnan(s), -np.inf, s)
    order = np.lexsort((u, -s))          # last key is primary: -s ascending = s descending
    f = np.zeros(len(s), bool)
    f[order[:k]] = True
    return f


def heuristic_scores(t, P):
    """t0 heuristic scores aligned with the project arrays P (by row_id)."""
    x = t.set_index("row_id").loc[P["row_id"]]
    return {"size": (x.lines_added_t0 + x.lines_deleted_t0).to_numpy(dtype=float),
            "files": x.commit_files_t0.to_numpy(dtype=float),
            "newauth": -x.author_pr_count.to_numpy(dtype=float)}


# ── one variant ─────────────────────────────────────────────────────────────

def variant_c(P, H, hist, nxt, u, c_r=C_MAIN):
    """B1 and B3 for one variant: G of the model, of every heuristic in H (same k), B3."""
    comp = E.components(P, hist, nxt)
    n = len(nxt)
    a = b = g = 1 / 3
    L = a * comp["ftm"].sum() + b * comp["fnit"].sum() + g * comp["stale"].sum()
    th = E.choose_threshold(P["p_it"][hist], P["y_it"][hist], c_r)
    fire = (P["p_it"][nxt] > th) if th is not None else np.zeros(n, bool)
    k = int(fire.sum())
    d = b * comp["d_nit"]

    def share(x):
        # same guard as E.measures: an empty loss sum gives a zero share
        return x / L if L > 0 else 0.0

    row = {"k": k, "n": n, "reach": th is not None,
           "G_model": share(d[fire].sum()), "R": (k / n) * share(d.sum()) if n else 0.0}
    for h, s in H.items():
        row["G_" + h] = share(d[top_k(s[nxt], u, k)].sum())
    # B3: t_m -> min(t_m, Q_r) on top of n_it -> 1 for requests with n_it >= 2
    one = P["nit"][hist] == 1
    q = float(np.median(P["tm"][hist][one])) if one.any() else np.inf
    tm, nit = P["tm"][nxt], P["nit"][nxt]
    d_tm = np.where(nit >= 2, comp["ftm"] - mid_cdf(P["tm"][hist], np.minimum(tm, q)), 0.0)
    d3 = d + a * d_tm
    row.update({"Q_r": q, "G_b3": share(d3[fire].sum()),
                "R_b3": (k / n) * share(d3.sum()) if n else 0.0})
    return row


def variant_descr(P, hist, nxt):
    """Descriptive numbers of E9 on one E9 variant, for c = 0.1, 0.2, 0.3 (weights 1/3)."""
    comp = E.components(P, hist, nxt)
    a, b, g = E.WTS["w3"]
    L = a * comp["ftm"].sum() + b * comp["fnit"].sum() + g * comp["stale"].sum()
    ceil = b * comp["d_nit"].sum() / L if L > 0 else 0.0
    y = P["y_it"][nxt] == 1
    z = np.zeros(len(nxt), bool)
    rows = []
    for c, cr in E.CR.items():
        th = E.choose_threshold(P["p_it"][hist], P["y_it"][hist], cr)
        fire = (P["p_it"][nxt] > th) if th is not None else z
        m = E.measures(comp, fire, z)["it"]["w3"]
        rows.append({"c": c, "k": int(fire.sum()), "n": len(nxt), "tp": int((fire & y).sum()),
                     "n_pos": int(y.sum()), "A": m[0], "R": m[1], "C": ceil})
    return rows


# ── projects (parallel over repositories) ──────────────────────────────────

_G = {}


def _run_repo_c(args):
    repo, i_r, seed, n_var = args
    P, H = _G["PA"][repo], _G["H"][repo]
    n = len(P["row_id"])
    h = n // 2
    rng = np.random.default_rng([seed, E.STREAM["shuf"], i_r])
    rng_u = np.random.default_rng([seed, STREAM_TIE, i_r])
    Hx = {**H, "pit": P["p_it"]}             # K4c: the forecast itself as a "heuristic"
    rows = []
    for v in range(n_var):
        perm = rng.permutation(n)
        u = rng_u.random(n - h)
        rows.append({"v": v, "repo": repo, **variant_c(P, Hx, perm[:h], perm[h:], u)})
    return rows


def _run_repo_descr(args):
    repo, i_r, seed, n_var = args
    P = _G["PA"][repo]
    n = len(P["row_id"])
    h = n // 2
    rng = np.random.default_rng([seed, E.STREAM["shuf"], i_r])   # the permutations of E9
    rows = []
    for v in range(n_var):
        perm = rng.permutation(n)
        for r in variant_descr(P, perm[:h], perm[h:]):
            rows.append({"v": v, "repo": repo, **r})
    return rows


def run_parallel(fn, jobs):
    with mp.get_context("fork").Pool(min(N_PROC, len(jobs))) as pool:
        parts = pool.map(fn, jobs)
    return pd.DataFrame([r for p in parts for r in p])


# ── summaries ───────────────────────────────────────────────────────────────

def mat(V, col, nv, R):
    return V.sort_values(["v", "repo"])[col].to_numpy(dtype=float).reshape(nv, R)


def ratio_summary(A, Bm, idx, prefix, nums):
    """Ratio of the corpus means of two measures (paired cluster bootstrap) and its spread
    over variants. A, Bm: (n_variants, n_repos)."""
    a_r, b_r = A.mean(axis=0), Bm.mean(axis=0)
    mA, mB = a_r[idx].mean(axis=1), b_r[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = np.where(mB > 0, mA / mB, np.nan)
        rv = np.where(Bm.mean(axis=1) > 0, A.mean(axis=1) / Bm.mean(axis=1), np.nan)
    nums[f"{prefix}.ratio"] = float(a_r.mean() / b_r.mean()) if b_r.mean() > 0 else None
    nums[f"{prefix}.ratio_lo"] = pct(rb, 2.5)
    nums[f"{prefix}.ratio_hi"] = pct(rb, 97.5)
    nums[f"{prefix}.ratio_boot_undefined"] = int(np.isnan(rb).sum())
    nums[f"{prefix}.var.ratio_med"] = pct(rv, 50)
    nums[f"{prefix}.var.ratio_p025"] = pct(rv, 2.5)
    nums[f"{prefix}.var.ratio_p975"] = pct(rv, 97.5)
    acting = a_r > 0
    nums[f"{prefix}.repo.n_acting"] = int(acting.sum())
    nums[f"{prefix}.repo.n_gt"] = int((a_r[acting] > b_r[acting]).sum())


def summarise_c(V, repos, seed, nums):
    R = len(repos)
    nv = int(V.v.max()) + 1
    idx = np.random.default_rng(seed).integers(0, R, size=(E.B, R))
    Rm = mat(V, "R", nv, R)
    Gm = mat(V, "G_model", nv, R)
    cluster_summary(Gm, Rm, idx, "e9.c.b1.model", nums)
    for h in HEUR:
        Gh = mat(V, "G_" + h, nv, R)
        cluster_summary(Gh, Rm, idx, f"e9.c.b1.{h}", nums)
        ratio_summary(Gm, Gh, idx, f"e9.c.b1.model_vs_{h}", nums)
    lo = nums["e9.c.b1.model_vs_size.ratio_lo"]
    nums["e9.c.b1.criterion_met"] = int(lo is not None and lo > 1)
    cluster_summary(mat(V, "G_b3", nv, R), mat(V, "R_b3", nv, R), idx, "e9.c.b3", nums)
    reach = mat(V, "reach", nv, R).astype(bool)
    nums["e9.c.n_unreach_med"] = float(np.median((~reach).sum(axis=1)))
    nums["e9.c.fire_pct"] = 100 * float((mat(V, "k", nv, R) / mat(V, "n", nv, R)).mean())
    q = V.groupby("repo").Q_r.median()
    nums["e9.c.b3.n_repos_q_undefined"] = int(np.isinf(V.Q_r).groupby(V.repo).any().sum())
    nums["e9.c.b3.q_med_h"] = float(np.median(q[np.isfinite(q)]))
    nums["e9.c.n_variants"] = nv
    nums["e9.c.n_repos"] = R
    nums["e9.c.seed"] = seed
    nums["e9.c.n_boot"] = E.B
    return nums


def summarise_descr(D, repos, seed, nums):
    R = len(repos)
    nv = int(D.v.max()) + 1
    idx = np.random.default_rng(seed).integers(0, R, size=(E.B, R))   # same draw as E9
    for c, cr in E.CR.items():
        s = D[D.c == c]
        A, Rr, C = mat(s, "A", nv, R), mat(s, "R", nv, R), mat(s, "C", nv, R)
        p = f"e9.c.descr.{c}"
        a_r, c_r = A.mean(axis=0), C.mean(axis=0)
        mA, mC = a_r[idx].mean(axis=1), c_r[idx].mean(axis=1)
        nums[f"{p}.A_pct"] = 100 * float(a_r.mean())
        nums[f"{p}.Arand_pct"] = 100 * float(Rr.mean(axis=0).mean())
        nums[f"{p}.ceil_pct"] = 100 * float(c_r.mean())
        nums[f"{p}.ceil_lo_pct"] = 100 * pct(mC, 2.5)
        nums[f"{p}.ceil_hi_pct"] = 100 * pct(mC, 97.5)
        nums[f"{p}.ceil_min_pct"] = 100 * float(c_r.min())
        nums[f"{p}.ceil_max_pct"] = 100 * float(c_r.max())
        nums[f"{p}.share_of_ceil_pct"] = 100 * float(a_r.mean() / c_r.mean())
        sh = mA / mC
        nums[f"{p}.share_of_ceil_lo_pct"] = 100 * pct(sh, 2.5)
        nums[f"{p}.share_of_ceil_hi_pct"] = 100 * pct(sh, 97.5)
        # realised precision on the next period
        g = s.groupby("v")[["tp", "k"]].sum()
        prec = (g.tp / g.k).where(g.k > 0)
        nums[f"{p}.prec_pool_med_pct"] = 100 * pct(prec, 50) if prec.notna().any() else None
        nums[f"{p}.prec_pool_p025_pct"] = 100 * pct(prec, 2.5) if prec.notna().any() else None
        nums[f"{p}.prec_pool_p975_pct"] = 100 * pct(prec, 97.5) if prec.notna().any() else None
        pm = nums[f"{p}.prec_pool_med_pct"]
        nums[f"{p}.unneeded_pool_med_pct"] = None if pm is None else 100 - pm
        act = s[s.k > 0]
        gb = act.groupby("v")[["n_pos", "n"]].sum()
        nums[f"{p}.base_acting_med_pct"] = 100 * pct(gb.n_pos / gb.n, 50) if len(gb) else None
        gr = s.groupby("repo")[["tp", "k"]].sum()
        gr = gr[gr.k > 0]
        nums[f"{p}.prec_repo_med_pct"] = 100 * float((gr.tp / gr.k).median()) if len(gr) else None
        nums[f"{p}.prec_repo_n"] = int(len(gr))
        below = (act.tp / act.k) < (1 - cr - 1e-12)
        nums[f"{p}.pairs_below_target_pct"] = 100 * float(below.mean()) if len(act) else None
        nums[f"{p}.pairs_acting"] = int(len(act))
    nums["e9.c.descr.seed"] = seed
    nums["e9.c.descr.n_variants"] = nv
    return nums


# ── main ────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="seed 1, 3 projects")
    args = ap.parse_args(argv)
    t0 = time.time()
    seed_c = E.SEED_DEV if args.dev else SEED_C
    seed_d = E.SEED_DEV if args.dev else E.SEED_MAIN
    nv_c = 10 if args.dev else E.N_SHUF
    nv_d = E.N_SHUF_DEV if args.dev else E.N_SHUF
    out = os.path.join(E.OUT, "dev") if args.dev else E.OUT
    os.makedirs(out, exist_ok=True)

    t = E.load_table()
    x = pd.read_csv(E.TABLE, usecols=HEUR_COLS, low_memory=False)
    for col in HEUR_COLS:
        t[col] = x[col].to_numpy()
    preds = {k: E.load_preds(t, tg) for k, tg in (("it", "IT"), ("st", "ST"), ("tr", "TFR"))}
    k5, _ = E.reproduce_e8(t, preds)
    assert k5["ok"], k5["mismatches"]
    all_repos = sorted(t.loc[(t.hait_filter == 1) & (t.in_main == 1), "repo"].unique())
    assert len(all_repos) == 40
    repos = sorted(E.DEV_REPOS) if args.dev else all_repos
    PA = E.project_arrays(t, preds, set(repos))
    _G["PA"] = PA
    _G["H"] = {r: heuristic_scores(t, PA[r]) for r in repos}

    jobs_c = [(r, all_repos.index(r), seed_c, nv_c) for r in repos]
    V = run_parallel(_run_repo_c, jobs_c)
    jobs_d = [(r, all_repos.index(r), seed_d, nv_d) for r in repos]
    D = run_parallel(_run_repo_descr, jobs_d)
    print(f"variants done: {time.time() - t0:.0f} s", flush=True)

    checks = {"k5_e8_reproduction_ok": True}
    # K1c: no action -> every G is 0
    z = V[V.k == 0]
    checks["k1c_null_action"] = bool((z[["G_model", "G_size", "G_files", "G_newauth", "G_pit",
                                         "G_b3", "R", "R_b3"]] == 0).all().all())
    # K4c: the forecast as a heuristic reproduces the model exactly
    checks["k4c_forecast_as_heuristic"] = bool((V.G_pit == V.G_model).all())
    # B3 never below the model on the same requests
    checks["b3_not_below_model"] = bool((V.G_b3 >= V.G_model - 1e-15).all())
    nums = {}
    summarise_c(V, repos, seed_c, nums)
    summarise_descr(D, repos, seed_d, nums)
    # K6: the descriptive part reproduces E9 exactly
    e9 = json.load(open(os.path.join(out, "numbers.json"), encoding="utf-8"))
    k6 = {}
    for c in E.CR:
        for mine, theirs in (("A_pct", "A_pct"), ("Arand_pct", "Arand_pct")):
            got, exp = nums[f"e9.c.descr.{c}.{mine}"], e9.get(f"e9.shuf.it.{c}.w3.{theirs}")
            k6[f"{c}.{mine}"] = {"got": got, "expected": exp,
                                 "equal": exp is not None and got == exp,
                                 "close": exp is not None and abs(got - exp) <= 1e-12 * max(1, abs(exp))}
    checks["k6_e9_reproduction"] = k6
    checks["k6_ok"] = all(v["close"] for v in k6.values())
    ok = checks["k1c_null_action"] and checks["k4c_forecast_as_heuristic"] and checks["b3_not_below_model"]
    if not args.dev and not (ok and checks["k6_ok"]):
        json.dump(checks, open(os.path.join(out, "checks_e9c.json"), "w", encoding="utf-8"),
                  indent=1, ensure_ascii=False)
        sys.exit("E9c controls failed: stop (see checks_e9c.json)")

    run = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": ("dev" if args.dev else "main") + "_e9c",
           "seed": seed_c, "seed_descr": seed_d, "n_var": nv_c, "n_var_descr": nv_d,
           "repos": repos if args.dev else len(repos), "git": E.git_head(),
           "seconds": round(time.time() - t0, 1),
           "checks": {k: v for k, v in checks.items() if isinstance(v, bool)},
           "summary": {k: nums.get(k) for k in (
               "e9.c.b1.model.G_pct", "e9.c.b1.model.Grand_pct", "e9.c.b1.model.lift",
               "e9.c.b1.size.lift", "e9.c.b1.model_vs_size.ratio",
               "e9.c.b1.model_vs_size.ratio_lo", "e9.c.b1.model_vs_size.ratio_hi",
               "e9.c.b1.criterion_met", "e9.c.b3.G_pct", "e9.c.b3.lift",
               "e9.c.descr.c02.ceil_pct", "e9.c.descr.c02.share_of_ceil_pct",
               "e9.c.descr.c02.prec_pool_med_pct")}}
    with open(os.path.join(E.OUT, "runs.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(run, ensure_ascii=False) + "\n")
    checks["runs"] = [json.loads(x) for x in open(os.path.join(E.OUT, "runs.jsonl"), encoding="utf-8")]
    json.dump(checks, open(os.path.join(out, "checks_e9c.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    json.dump({k: nums[k] for k in sorted(nums)},
              open(os.path.join(out, "numbers_e9c.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    br = V.groupby("repo")[["k", "n", "G_model", "R", "G_size", "G_files", "G_newauth",
                            "G_b3", "R_b3"]].mean()
    br.to_csv(os.path.join(out, "by_repo_e9c.csv"))
    V.to_csv(os.path.join(out, "variants_e9c.csv"), index=False)
    D.to_csv(os.path.join(out, "variants_e9c_descr.csv"), index=False)
    print(json.dumps(run, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
