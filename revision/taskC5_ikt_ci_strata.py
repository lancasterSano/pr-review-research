#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task C5 (стаття ІКТ-2026) — доповнення до описового порівняння C4:

  1. поправка Холма на 24 тести (8 репозиторіїв × 3 величини);
  2. 95 % бутстреп-інтервали (percentile, 2 000 повторень) для рангової
     бісеріальної кореляції;
  3. контроль розміру зміни: повторення порівняння кількості ітерацій
     усередині квартилів доданих рядків (межі квартилів — за всіма злитими
     запитами репозиторію, створеними у 2025–2026 роках);
  4. кількість запитів, створених агентами, у групах табл. 3.

Той самий корпус і ті самі фільтри, що й у taskC4_observational.py
(злиті запити аналітичного набору, створені у 2025–2026 роках, тести лише
де обидві групи мають n ≥ 20). Суто описово, без причинного прочитання.

Вихід: <DISSERTATION_DIR>/conf_papers/october_2026/IKT2026/data/{C5-table3-ci-holm.csv,
C5-strata-lines-added.csv, C5-summary.md}
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu, rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taskC_common import DISSERTATION, OUT_DIR, MERGED_CSV, load_processed  # noqa: E402

FLAGS_CSV = os.path.join(OUT_DIR, "C-ai-pr-flags.csv")
OUT = os.path.join(DISSERTATION, "conf_papers", "october_2026", "IKT2026", "data")
os.makedirs(OUT, exist_ok=True)
MIN_N = 20
B = 2000
SEED = 20260903
# порядок табл. 2 статті ІКТ (за спаданням частки)
ORDER = ["microsoft/vscode", "woocommerce/woocommerce", "vercel/next.js", "supabase/supabase",
         "mattermost/mattermost", "grafana/grafana", "strapi/strapi", "discourse/discourse"]
METRICS = {"rounds": "review_rounds", "ttm": "time_to_merge_hours", "lines_added": "lines_added"}


def rb_and_p(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    res = mannwhitneyu(a, b, alternative="two-sided", use_continuity=True, method="asymptotic")
    rb = 2.0 * res.statistic / (len(a) * len(b)) - 1.0
    return rb, res.pvalue


def rb_fast(a, b):
    """рангова бісеріальна через ранги (для бутстрепу)."""
    n1 = len(a)
    r = rankdata(np.concatenate([a, b]))
    u1 = r[:n1].sum() - n1 * (n1 + 1) / 2.0
    return 2.0 * u1 / (n1 * len(b)) - 1.0


def boot_ci(a, b, rng):
    a = np.asarray(a, float); b = np.asarray(b, float)
    vals = np.empty(B)
    for i in range(B):
        vals[i] = rb_fast(rng.choice(a, len(a)), rng.choice(b, len(b)))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def holm(pvals):
    """повертає скориговані p (Holm–Bonferroni) у порядку входу."""
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); adj = np.empty(m)
    running = 0.0
    for k, idx in enumerate(order):
        running = max(running, (m - k) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def main():
    flags = {}
    for r in load_processed(FLAGS_CSV):
        if r["pr_type"] == "merged":
            flags[(r["repo"], int(r["pr_number"]))] = (int(r["has_ai_review"]), int(r["ai_authored"]))
    groups = defaultdict(lambda: {"ai": defaultdict(list), "no": defaultdict(list)})
    agent_authored = defaultdict(lambda: {"ai": 0, "no": 0})
    for r in load_processed(MERGED_CSV):
        if (r["created_at"] or "")[:4] not in ("2025", "2026"):
            continue
        key = (r["repo"], int(r["pr_number"]))
        if key not in flags:
            continue
        has_ai, authored = flags[key]
        g = "ai" if has_ai else "no"
        agent_authored[r["repo"]][g] += authored
        try:
            vals = {m: float(r[c]) for m, c in METRICS.items()}
        except (TypeError, ValueError):
            continue
        for m, v in vals.items():
            groups[r["repo"]][g][m].append(v)

    rng = np.random.default_rng(SEED)
    rows, ptests = [], []
    for repo in ORDER:
        g = groups[repo]
        n_ai, n_no = len(g["ai"]["rounds"]), len(g["no"]["rounds"])
        row = {"repo": repo, "n_ai": n_ai, "n_no_ai": n_no,
               "agent_authored_ai": agent_authored[repo]["ai"], "agent_authored_no": agent_authored[repo]["no"]}
        for m in METRICS:
            a, b = g["ai"][m], g["no"][m]
            row[f"med_{m}_ai"] = float(np.median(a)); row[f"med_{m}_no"] = float(np.median(b))
            rb, p = rb_and_p(a, b)
            lo, hi = boot_ci(a, b, rng)
            row[f"rb_{m}"] = rb; row[f"ci_lo_{m}"] = lo; row[f"ci_hi_{m}"] = hi; row[f"p_{m}"] = p
            ptests.append((repo, m, p))
        rows.append(row)
    adj = holm([p for _, _, p in ptests])
    for (repo, m, p), pa in zip(ptests, adj):
        next(r for r in rows if r["repo"] == repo)[f"p_holm_{m}"] = pa

    with open(os.path.join(OUT, "C5-table3-ci-holm.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})

    # ── страти за доданими рядками ────────────────────────────────────────
    srows, sp = [], []
    for repo in ORDER:
        g = groups[repo]
        la = np.array(g["ai"]["lines_added"] + g["no"]["lines_added"])
        q = np.quantile(la, [0.25, 0.5, 0.75])
        def stratum(v):
            return 0 if v <= q[0] else 1 if v <= q[1] else 2 if v <= q[2] else 3
        for s in range(4):
            a = [r for r, l in zip(g["ai"]["rounds"], g["ai"]["lines_added"]) if stratum(l) == s]
            b = [r for r, l in zip(g["no"]["rounds"], g["no"]["lines_added"]) if stratum(l) == s]
            row = {"repo": repo, "stratum": f"Q{s+1}", "lines_added_range": (
                f"<= {q[0]:.0f}" if s == 0 else f"{q[s-1]:.0f} < x <= {q[s]:.0f}" if s < 3 else f"> {q[2]:.0f}"),
                "n_ai": len(a), "n_no_ai": len(b),
                "med_rounds_ai": float(np.median(a)) if a else None,
                "med_rounds_no": float(np.median(b)) if b else None}
            if len(a) >= MIN_N and len(b) >= MIN_N:
                rb, p = rb_and_p(a, b); lo, hi = boot_ci(a, b, rng)
                row.update({"rb_rounds": rb, "ci_lo": lo, "ci_hi": hi, "p": p}); sp.append((len(srows), p))
            else:
                row.update({"rb_rounds": None, "ci_lo": None, "ci_hi": None, "p": None})
            srows.append(row)
    adj = holm([p for _, p in sp])
    for (i, _), pa in zip(sp, adj):
        srows[i]["p_holm"] = pa
    with open(os.path.join(OUT, "C5-strata-lines-added.csv"), "w", newline="", encoding="utf-8") as f:
        cols = ["repo", "stratum", "lines_added_range", "n_ai", "n_no_ai", "med_rounds_ai", "med_rounds_no",
                "rb_rounds", "ci_lo", "ci_hi", "p", "p_holm"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in srows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items() if k in cols})

    # ── підсумок ──────────────────────────────────────────────────────────
    L = ["# C5 — інтервали, поправка Холма, страти за розміром (ІКТ-2026)", "",
         f"B = {B} бутстреп-повторень, seed {SEED}; тести лише де обидві групи n >= {MIN_N}; 24 тести табл. 3 і "
         f"{len(sp)} тестів у стратах скориговано за Холмом окремо.", "",
         "## Табл. 3 з інтервалами", "",
         "| repo | n ai/no | agent-authored ai/no | rounds med ai/no rb [CI] p_holm | ttm med ai/no rb [CI] p_holm | lines med ai/no rb [CI] p_holm |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        cells = []
        for m in METRICS:
            cells.append(f"{r[f'med_{m}_ai']:g} / {r[f'med_{m}_no']:g}; {r[f'rb_{m}']:+.2f} [{r[f'ci_lo_{m}']:+.2f}; {r[f'ci_hi_{m}']:+.2f}]; p_H={r[f'p_holm_{m}']:.2g}")
        L.append(f"| {r['repo']} | {r['n_ai']} / {r['n_no_ai']} | {r['agent_authored_ai']} / {r['agent_authored_no']} | " + " | ".join(cells) + " |")
    L += ["", "## Ітерації всередині квартилів доданих рядків", "",
          "| repo | Q | range | n ai/no | med ai/no | rb [CI] | p_holm |", "|---|---|---|---|---|---|---|"]
    for r in srows:
        if r["rb_rounds"] is None:
            L.append(f"| {r['repo']} | {r['stratum']} | {r['lines_added_range']} | {r['n_ai']} / {r['n_no_ai']} | — | n < {MIN_N} | |")
        else:
            L.append(f"| {r['repo']} | {r['stratum']} | {r['lines_added_range']} | {r['n_ai']} / {r['n_no_ai']} | "
                     f"{r['med_rounds_ai']:g} / {r['med_rounds_no']:g} | {r['rb_rounds']:+.2f} [{r['ci_lo']:+.2f}; {r['ci_hi']:+.2f}] | {r.get('p_holm', float('nan')):.2g} |")
    open(os.path.join(OUT, "C5-summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
