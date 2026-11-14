"""P2 premise for chapter 4: share of review threads that start at a model trigger
(<= 1 h after PR creation, leaving draft or a new revision), assistants vs deterministic
bots vs human reviewers, with a cluster bootstrap over repositories.

Input (read only): $R2_DIR/v2/r2_threads_v2.csv
(output of revision/r2/r2_raw_v2.py; columns starter_class, hours_since_trigger).
Reference numbers: $R2_DIR/r2_numbers.json (revision/r2/make_tables_r2.py)
(react.total.trig_pct, react.trig_bot_pct, react.trig_h_pct).

Measure, as in make_tables_r2.reactions(): pooled shares (all threads of a group together).
Assistants are pooled over the "shown" repositories with >= REACT_MIN assistant threads;
humans and bots are pooled over all repositories. The bootstrap resamples all repositories
(cluster = repository) and recomputes the same quantities, so repository membership in the
"shown" set travels with the repository.

Criterion: confirmed if the 95 % percentile interval of p_ai - p_human excludes 0
AND the interval of |p_ai - p_human| - |p_ai - p_bot| lies above 0.

Output: revision/output/r4/p2_trigger/{numbers.json, by_repo.csv}.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CLONE = os.path.dirname(os.path.dirname(HERE))
R2 = os.environ.get("R2_DIR") or os.path.join(CLONE, "revision", "output", "r2")
THREADS = os.path.join(R2, "v2", "r2_threads_v2.csv")
R2_NUMBERS = os.path.join(R2, "r2_numbers.json")
OUT = os.path.join(CLONE, "revision", "output", "r4", "p2_trigger")

REACT_MIN = 100      # make_tables_r2.REACT_MIN: repositories shown in the reactions table
MIN_REPO_MEAN = 20   # sensitivity: repository enters the per-repository mean with >= 20 threads
TRIG_H = 1.0
B = 10_000
SEED = 20261114
GROUPS = {"ai": "ai_assistant", "human": "human", "bot": "other_bot"}
P = "e2.p2."


def wilson(k, n, z=1.959964):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def per_repo(th, trig_h=TRIG_H):
    th = th.assign(trig=th.hours_since_trigger <= trig_h)
    repos = sorted(th.repo.unique())
    tab = pd.DataFrame({"repo": repos})
    for g, cls in GROUPS.items():
        s = th[th.starter_class == cls].groupby("repo").trig.agg(["size", "sum"])
        tab[f"n_{g}"] = tab.repo.map(s["size"]).fillna(0).astype(int)
        tab[f"trig_{g}"] = tab.repo.map(s["sum"]).fillna(0).astype(int)
        tab[f"pct_{g}"] = np.where(tab[f"n_{g}"] > 0, 100 * tab[f"trig_{g}"] / tab[f"n_{g}"].where(tab[f"n_{g}"] > 0), np.nan)
        ci = [wilson(k, n) for k, n in zip(tab[f"trig_{g}"], tab[f"n_{g}"])]
        tab[f"pct_{g}_lo"] = [c[0] for c in ci]
        tab[f"pct_{g}_hi"] = [c[1] for c in ci]
    tab["shown_ai"] = tab.n_ai >= REACT_MIN
    return tab


def pooled(W, k, n):
    """W: (B, R) repository multiplicities; k, n: (R,) trigger counts and thread counts."""
    num, den = W @ k, W @ n
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, 100 * num / den, np.nan)


def repo_mean(W, pct, elig):
    w = W * elig
    den = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (w @ np.nan_to_num(pct)) / den, np.nan)


def stats(pa, ph, pb):
    d = pa - ph
    gap = np.abs(pa - ph) - np.abs(pa - pb)
    return d, gap


def interval(x):
    x = x[~np.isnan(x)]
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)), int(B - len(x))


def verdict(d_lo, d_hi, g_lo):
    ok1 = d_lo > 0 or d_hi < 0
    ok2 = g_lo > 0
    return "підтверджено" if (ok1 and ok2) else "ні"


def variant(name, W1, Wb, pa_fn, ph_fn, pb_fn):
    """Point estimate with W1 (ones), bootstrap with Wb; returns flat dict with prefix."""
    pa, ph, pb = pa_fn(W1)[0], ph_fn(W1)[0], pb_fn(W1)[0]
    d0, g0 = stats(pa, ph, pb)
    ba, bh, bb = pa_fn(Wb), ph_fn(Wb), pb_fn(Wb)
    d, g = stats(ba, bh, bb)
    dlo, dhi, dnan = interval(d)
    glo, ghi, gnan = interval(g)
    alo, ahi, _ = interval(ba)
    hlo, hhi, _ = interval(bh)
    blo, bhi, bnan = interval(bb)
    p = P + name
    return {
        p + "ai_trig_pct": float(pa), p + "ai_trig_lo_pct": alo, p + "ai_trig_hi_pct": ahi,
        p + "human_trig_pct": float(ph), p + "human_trig_lo_pct": hlo, p + "human_trig_hi_pct": hhi,
        p + "bot_trig_pct": float(pb), p + "bot_trig_lo_pct": blo, p + "bot_trig_hi_pct": bhi,
        p + "diff_ai_human_pct": float(d0), p + "diff_ai_human_lo_pct": dlo, p + "diff_ai_human_hi_pct": dhi,
        p + "dist_gap_pct": float(g0), p + "dist_gap_lo_pct": glo, p + "dist_gap_hi_pct": ghi,
        p + "boot_nan": int(max(dnan, gnan, bnan)),
        p + "verdict": verdict(dlo, dhi, glo),
    }


def boot_weights(R):
    rng = np.random.default_rng(SEED)
    draw = rng.integers(0, R, size=(B, R))
    Wb = np.zeros((B, R))
    np.add.at(Wb, (np.repeat(np.arange(B), R), draw.ravel()), 1.0)
    return Wb


def main_measure(name, tab, Wb):
    """The chapter 2 measure (assistants pooled over shown repositories, others over all) on tab."""
    k = {g: tab[f"trig_{g}"].to_numpy(float) for g in GROUPS}
    n = {g: tab[f"n_{g}"].to_numpy(float) for g in GROUPS}
    shown = tab.shown_ai.to_numpy(float)
    return variant(name, np.ones((1, len(tab))), Wb,
                   lambda W: pooled(W, k["ai"] * shown, n["ai"] * shown),
                   lambda W: pooled(W, k["human"], n["human"]),
                   lambda W: pooled(W, k["bot"], n["bot"]))


WINDOWS = {"h0083": 5 / 60, "h025": 0.25, "h05": 0.5, "h2": 2.0, "h3": 3.0}


def main():
    th = pd.read_csv(THREADS, usecols=["repo", "pr_number", "starter_class", "hours_since_trigger"])
    assert th.hours_since_trigger.notna().all(), "missing hours_since_trigger"
    tab = per_repo(th)
    R = len(tab)
    Wb = boot_weights(R)
    W1 = np.ones((1, R))

    k = {g: tab[f"trig_{g}"].to_numpy(float) for g in GROUPS}
    n = {g: tab[f"n_{g}"].to_numpy(float) for g in GROUPS}
    shown = tab.shown_ai.to_numpy(float)
    pct = {g: tab[f"pct_{g}"].to_numpy(float) for g in GROUPS}
    elig = {g: (tab[f"n_{g}"] >= MIN_REPO_MEAN).to_numpy(float) for g in GROUPS}

    out = {
        P + "measure": "pooled; assistants over repositories with >= 100 assistant threads, "
                       "humans and bots over all repositories (as make_tables_r2.reactions)",
        P + "trigger_h": TRIG_H, P + "boot_b": B, P + "boot_seed": SEED,
        P + "boot_cluster": "repository", P + "boot_ci": "percentile 95 %",
        P + "n_repos_cluster": int(R),
        P + "n_repos_ai": int(tab.shown_ai.sum()), P + "n_repos_ai_any": int((tab.n_ai > 0).sum()),
        P + "n_repos_human": int((tab.n_human > 0).sum()), P + "n_repos_bot": int((tab.n_bot > 0).sum()),
        P + "n_repos_all_three": int(((tab.n_ai > 0) & (tab.n_human > 0) & (tab.n_bot > 0)).sum()),
        P + "n_threads_ai": int(tab.n_ai[tab.shown_ai].sum()), P + "n_threads_ai_any": int(tab.n_ai.sum()),
        P + "n_threads_human": int(tab.n_human.sum()), P + "n_threads_bot": int(tab.n_bot.sum()),
        P + "n_prs_ai": int(th[(th.starter_class == "ai_assistant") & th.repo.isin(tab.repo[tab.shown_ai])]
                            [["repo", "pr_number"]].drop_duplicates().shape[0]),
        P + "bot_top_repo": tab.loc[tab.n_bot.idxmax(), "repo"],
        P + "bot_top_repo_share_pct": float(100 * tab.n_bot.max() / tab.n_bot.sum()),
    }
    # descriptive, per repository (for the forest plot)
    s = tab[tab.shown_ai]
    out[P + "repo_ai_gt_human_n"] = int((s.pct_ai > s.pct_human).sum())
    out[P + "repo_ai_gt_human_of"] = int(len(s))
    c = tab[(tab.n_ai >= MIN_REPO_MEAN) & (tab.n_bot >= MIN_REPO_MEAN)]
    closer = (c.pct_ai - c.pct_human).abs() > (c.pct_ai - c.pct_bot).abs()
    out[P + "repo_closer_bot_n"] = int(closer.sum())
    out[P + "repo_closer_bot_of"] = int(len(c))
    out[P + "repo_closer_human"] = ", ".join(c.repo[~closer])
    # main: the measure used in chapter 2
    main_v = variant("", W1, Wb,
                     lambda W: pooled(W, k["ai"] * shown, n["ai"] * shown),
                     lambda W: pooled(W, k["human"], n["human"]),
                     lambda W: pooled(W, k["bot"], n["bot"]))
    out.update(main_v)
    # sensitivity 1: assistants pooled over all repositories (react.trig_all_pct)
    out.update(variant("sens_all.", W1, Wb,
                       lambda W: pooled(W, k["ai"], n["ai"]),
                       lambda W: pooled(W, k["human"], n["human"]),
                       lambda W: pooled(W, k["bot"], n["bot"])))
    # sensitivity 2: all three pooled over the same repositories (all three groups present, ai shown)
    common = shown * (n["human"] > 0) * (n["bot"] > 0)
    out[P + "sens_common.n_repos"] = int(common.sum())
    out.update(variant("sens_common.", W1, Wb,
                       lambda W: pooled(W, k["ai"] * common, n["ai"] * common),
                       lambda W: pooled(W, k["human"] * common, n["human"] * common),
                       lambda W: pooled(W, k["bot"] * common, n["bot"] * common)))
    # sensitivity 3: unweighted mean of per-repository shares (repository with >= 20 threads of the group)
    for g in GROUPS:
        out[P + f"sens_repomean.n_repos_{g}"] = int(elig[g].sum())
    out.update(variant("sens_repomean.", W1, Wb,
                       lambda W: repo_mean(W, pct["ai"], elig["ai"]),
                       lambda W: repo_mean(W, pct["human"], elig["human"]),
                       lambda W: repo_mean(W, pct["bot"], elig["bot"])))

    # sensitivity 4: trigger window (same shown set, same bootstrap draws)
    for key, h in WINDOWS.items():
        out[P + f"sens_window.{key}.trigger_h"] = h
        out.update(main_measure(f"sens_window.{key}.", per_repo(th, h), Wb))
    # median hours from the trigger, threads of the main measure (assistants: shown repositories)
    shown_repos = set(tab.repo[tab.shown_ai])
    for g, cls in GROUPS.items():
        sel = th[th.starter_class == cls]
        if g == "ai":
            out[P + "median_h.ai_any"] = float(sel.hours_since_trigger.median())
            sel = sel[sel.repo.isin(shown_repos)]
        out[P + f"median_h.{g}"] = float(sel.hours_since_trigger.median())
    # sensitivity 5: without the repository with most assistant threads / most bot threads
    for key, col in (("sens_no_top_ai.", "n_ai"), ("sens_no_top_bot.", "n_bot")):
        top = tab.loc[tab[col].idxmax(), "repo"]
        t2 = tab[tab.repo != top].reset_index(drop=True)
        out[P + key + "dropped_repo"] = top
        out[P + key + "n_repos_ai"] = int(t2.shown_ai.sum())
        out[P + key + "n_repos_bot"] = int((t2.n_bot > 0).sum())
        out.update(main_measure(key, t2, boot_weights(len(t2))))

    # check against chapter 2 numbers
    r2 = json.load(open(R2_NUMBERS, encoding="utf-8"))["react"]
    ref = {"ai_trig_pct": r2["total"]["trig_pct"], "human_trig_pct": r2["trig_h_pct"],
           "bot_trig_pct": r2["trig_bot_pct"], "sens_all.ai_trig_pct": r2["trig_all_pct"]}
    diffs = {kk: abs(out[P + kk] - v) for kk, v in ref.items()}
    out[P + "r2_match"] = bool(max(diffs.values()) < 1e-9)
    out[P + "r2_match_max_abs_diff"] = float(max(diffs.values()))
    assert out[P + "n_threads_ai"] == r2["total"]["threads"], "shown-repo thread count differs from chapter 2"
    assert out[P + "n_repos_ai"] == r2["n_shown"]

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "numbers.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    cols = ["repo", "shown_ai"] + [f"{a}_{g}" for g in GROUPS for a in ("n", "trig")] + \
           [c for g in GROUPS for c in (f"pct_{g}", f"pct_{g}_lo", f"pct_{g}_hi")]
    tab[cols].sort_values("n_ai", ascending=False).to_csv(os.path.join(OUT, "by_repo.csv"), index=False,
                                                          float_format="%.4f")
    for kk, v in out.items():
        print(kk, round(v, 3) if isinstance(v, float) else v)


if __name__ == "__main__":
    main()
