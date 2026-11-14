#!/usr/bin/env python3
"""P5 for chapter 4: does the event definition change the assistant conclusion?

Premise of principle P5 (measurability) of the chapter 2 methodology (item 2.4.1).
Historical data only, no new collection.

Question: does choosing the event definition (human events only vs events of all
accounts) change the conclusion about the difference between PRs with and without
an AI assistant review? The answer is about the definition, not an independent
estimate of the assistant effect.

Within-repository r_rb come from the chapter 2 code, imported, not copied:
`make_tables_r2.load()` (PR table r3_b40d + r2_pr_extra_v2.csv) and
`r2_ai_compare_v2.compare(main)` (2025-2026 PRs, groups by u_ai, MIN_GROUP = 20,
r_rb = 2U/(n1*n2) - 1 from Mann-Whitney U). The script recomputes r_rb with the
imported `rrb()` on the same samples and asserts equality with `compare()`.

Characteristics with two definitions in compare():
  it  iterations: n_it_human vs n_it_all_lit;
  tr  time to first review: t_r_h_D vs t_r_all_h_D, both on PRs with a human
      review (t_r_h_D not empty), as in compare().
t_m and added lines do not depend on the event definition, so they are not compared.

Pooled effect: mean r_rb over repositories where both groups have >= MIN_GROUP PRs
under both definitions at once. Interval: cluster bootstrap over repositories
(N_BOOT draws, the same draws for both definitions), 95 % percentile, fixed seed.
No Holm correction.

The conclusion changed if (a) the means have different signs and neither interval
contains 0, or (b) one interval contains 0 and the other does not. One
characteristic is enough. Per-repository changes are descriptive: the same rule
on the within-repository bootstrap interval (`boot_ci` of compare(), resampling
within groups, 2 000 draws), plus the count of sign flips.

Reads only; writes to revision/output/r4/p5_definition/ (numbers.json, by_repo.csv).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # revision/
sys.path.insert(0, os.path.join(REV, "r2"))                        # chapter 2 tables (revision/r2)
import make_tables_r2 as mt          # noqa: E402
import r2_ai_compare_v2 as ac        # noqa: E402

OUT = os.path.join(REV, "output", "r4", "p5_definition")
SEED = 20261114
N_BOOT = 10_000
MIN_GROUP = ac.MIN_GROUP
# key -> (human column, all-accounts column, row filter as in compare(), compare() row keys)
CHARS = {
    "it": ("n_it_human", "n_it_all_lit", None, ("it_h_r", "it_all_r")),
    "tr": ("t_r_h_D", "t_r_all_h_D", "t_r_h_D", ("tr_h_r", "tr_all_r")),
}
DEFS = ("human", "all")


def samples(g, col, filt):
    """(x, y): x with the assistant, y without; selection exactly as in compare()."""
    if filt is not None:
        g = g[g[filt].notna()]
    x = g.loc[g.u_ai == 1, col].dropna().to_numpy(float)
    y = g.loc[g.u_ai == 0, col].dropna().to_numpy(float)
    return x, y


def contains0(lo, hi):
    return bool(lo <= 0.0 <= hi)


def verdict(m1, lo1, hi1, m2, lo2, hi2):
    """(changed, condition) by the author's rule; condition is 'a', 'b' or None."""
    z1, z2 = contains0(lo1, hi1), contains0(lo2, hi2)
    if np.sign(m1) != np.sign(m2) and not z1 and not z2:
        return True, "a"
    if z1 != z2:
        return True, "b"
    return False, None


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    _, m = mt.load()
    cmp_res = ac.compare(m)
    cmp_rows = {r["repo"]: r for r in cmp_res["rows"]}
    d = m[m["created"].dt.year >= 2025]          # the same period as compare()
    repos_all = sorted(d.repo.unique())

    rng_repo = np.random.default_rng(SEED + 1)    # within-repository intervals
    rng_clu = np.random.default_rng(SEED)         # cluster bootstrap
    nums = {}
    rows = []
    max_diff = 0.0
    sets_equal_all = True
    any_changed, conds = False, []
    for ch, (col_h, col_a, filt, (key_h, key_a)) in CHARS.items():
        # 1. repository sets under each definition (all 2025-2026 repos, not only compare's)
        ok = {k: set() for k in DEFS}
        smp = {}
        for repo in repos_all:
            g = d[d.repo == repo]
            for k, col in zip(DEFS, (col_h, col_a)):
                x, y = samples(g, col, filt)
                smp[(repo, k)] = (x, y)
                if len(x) >= MIN_GROUP and len(y) >= MIN_GROUP:
                    ok[k].add(repo)
        both = sorted(ok["human"] & ok["all"])
        eq = ok["human"] == ok["all"]
        sets_equal_all &= eq
        nums[f"e2.p5.{ch}.n_repos_set_human"] = len(ok["human"])
        nums[f"e2.p5.{ch}.n_repos_set_all"] = len(ok["all"])
        nums[f"e2.p5.{ch}.repo_sets_equal"] = eq
        if not eq:
            nums[f"e2.p5.{ch}.only_human"] = sorted(ok["human"] - ok["all"])
            nums[f"e2.p5.{ch}.only_all"] = sorted(ok["all"] - ok["human"])

        # 2. point r_rb per repository, checked against compare()
        R = np.empty((len(both), 2))
        for i, repo in enumerate(both):
            rec = {"characteristic": ch, "repo": repo, "short": repo.split("/")[1]}
            for j, (k, ck) in enumerate(zip(DEFS, (key_h, key_a))):
                x, y = smp[(repo, k)]
                r, p = ac.rrb(x, y)
                max_diff = max(max_diff, abs(r - cmp_rows[repo][ck]))
                lo, hi = ac.boot_ci(x, y, rng_repo)
                R[i, j] = r
                rec.update({f"n_ai_{k}": len(x), f"n_no_{k}": len(y), f"r_{k}": r,
                            f"lo_{k}": lo, f"hi_{k}": hi, f"p_{k}": float(p)})
            rec["sign_flip"] = bool(np.sign(rec["r_human"]) != np.sign(rec["r_all"]))
            c, cond = verdict(rec["r_human"], rec["lo_human"], rec["hi_human"],
                              rec["r_all"], rec["lo_all"], rec["hi_all"])
            rec["changed"], rec["condition"] = c, cond or ""
            rows.append(rec)

        # 3. cluster bootstrap over repositories, the same draws for both definitions
        idx = rng_clu.integers(0, len(both), size=(N_BOOT, len(both)))
        boot = R[idx].mean(axis=1)                       # N_BOOT x 2
        diff = boot[:, 1] - boot[:, 0]
        est = {}
        for j, k in enumerate(DEFS):
            mean = float(R[:, j].mean())
            lo, hi = (float(v) for v in np.percentile(boot[:, j], [2.5, 97.5]))
            est[k] = (mean, lo, hi)
            nums[f"e2.p5.{ch}.{k}.mean"] = mean
            nums[f"e2.p5.{ch}.{k}.lo"] = lo
            nums[f"e2.p5.{ch}.{k}.hi"] = hi
            nums[f"e2.p5.{ch}.{k}.n_repos"] = len(both)
            nums[f"e2.p5.{ch}.{k}.n_pos"] = int((R[:, j] > 0).sum())
        nums[f"e2.p5.{ch}.diff_all_minus_human.mean"] = float(R[:, 1].mean() - R[:, 0].mean())
        nums[f"e2.p5.{ch}.diff_all_minus_human.lo"] = float(np.percentile(diff, 2.5))
        nums[f"e2.p5.{ch}.diff_all_minus_human.hi"] = float(np.percentile(diff, 97.5))
        c, cond = verdict(*est["human"], *est["all"])
        nums[f"e2.p5.{ch}.changed"] = c
        nums[f"e2.p5.{ch}.condition"] = cond
        any_changed |= c
        if cond:
            conds.append(f"{ch}:{cond}")
        rch = [r for r in rows if r["characteristic"] == ch]
        nums[f"e2.p5.{ch}.n_repos_changed"] = sum(r["changed"] for r in rch)
        nums[f"e2.p5.{ch}.n_repos_changed_a"] = sum(r["condition"] == "a" for r in rch)
        nums[f"e2.p5.{ch}.n_repos_changed_b"] = sum(r["condition"] == "b" for r in rch)
        nums[f"e2.p5.{ch}.n_repos_sign_flip"] = sum(r["sign_flip"] for r in rch)

    assert max_diff < 1e-12, f"r_rb differs from compare(): {max_diff}"
    nums["e2.p5.changed"] = any_changed
    nums["e2.p5.condition"] = ", ".join(conds) if conds else None
    nums["e2.p5.repo_sets_equal"] = sets_equal_all
    nums["e2.p5.n_repos_compare"] = cmp_res["n_repos"]
    nums["e2.p5.check_max_abs_diff_vs_compare"] = max_diff
    nums["_meta"] = {
        "script": "revision/r4_calc/r4_p5_definition.py",
        "data": "make_tables_r2.load(): " + mt.R3_RUN + "/v2/pr_table_v2.csv + rozdil_2/tables/v2/r2_pr_extra_v2.csv; "
                "2025-2026 PRs of the main sample; group u_ai",
        "r_rb": "2U/(n1*n2) - 1, positive = larger values with the assistant; r2_ai_compare_v2.rrb",
        "tr_rows": "PRs with a human review (t_r_h_D not empty) for both definitions, as in compare()",
        "min_group": MIN_GROUP, "n_boot": N_BOOT, "seed": SEED,
        "repo_ci": f"r2_ai_compare_v2.boot_ci, {ac.N_BOOT} draws within groups, seed {SEED + 1}",
        "holm": "not applied",
        "rule": "changed if (a) signs of means differ and neither CI contains 0, or (b) exactly one CI contains 0",
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT, "numbers.json"), "w", encoding="utf-8") as f:
        json.dump(nums, f, ensure_ascii=False, indent=1)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "by_repo.csv"), index=False)
    print(json.dumps({k: v for k, v in nums.items() if k != "_meta"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
