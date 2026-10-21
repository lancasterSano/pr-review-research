#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task I: числа Розділу 2, що залежать від кількості ітерацій, за буквальним правилом.

Правило (рішення автора 14.10.2026, буквальне означення HAIT Def. 2): нова
ітерація настає лише тоді, коли після рецензії людини автор зробив нові коміти
і потім була нова рецензія людини. Коміти після останньої рецензії людини, за
якими йде злиття без нової рецензії, ітерації не додають. Реалізація одна:
review_iterations.count_iterations(pr, is_human) (rule="literal" за замовчуванням).
Для порівняння рахується і попереднє правило коду (rule="code").

Події лише людей: автор події не автор запиту і
taskF1_duration_language.participant_class(login) == "human" (без ШІ-асистентів,
інших ботів, логінів `*[bot]` і службових облікових записів EXTRA_AUTOMATION).

Корпус: 12 репозиторіїв HAIT, 64 820 злитих запитів аналітичного набору
(all_merged_features.csv) і сирі вузли data/raw/ (не raw_v2).

Частина 2.5.3: протокол табл. 5 і 6 ІКТ (taskH_ikt_table5_human.run_protocol,
taskC5: 2025–2026, 8 репозиторіїв, Манн–Вітні, rb, бутстреп 2 000, Холм) з
ітераціями людей за буквальним правилом; поруч правило коду (як у H-*.csv).

Вихід: <DISSERTATION_DIR>/dissertation/rozdil_2/tables/r2_iterations_literal.json
(інші файли Розділу 2 не змінюються).

Запуск: .venv/bin/python revision/taskI_r2_iterations_literal.py
"""
import csv
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from review_iterations import (  # noqa: E402
    RULE_CODE, RULE_LITERAL, count_iterations, include_all, iteration_trace, parse_dt,
    reviewer_events,
)
from taskC_common import MERGED_CSV, RAW_DIR, REPO_DIRS, dir_to_repo  # noqa: E402
from taskF1_duration_language import AI_ERA_START, LANG, PR_TABLE, is_human  # noqa: E402
import taskH_ikt_table5_human as H  # noqa: E402
from paths import DISSERTATION  # noqa: E402

csv.field_size_limit(10 ** 9)
OUT_JSON = os.path.join(DISSERTATION, "dissertation", "rozdil_2", "tables",
                        "r2_iterations_literal.json")


def per_pr_table():
    """Ітерації кожного запиту аналітичного набору за обома правилами."""
    with open(MERGED_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    want = {(r["repo"], int(r["pr_number"])): r for r in rows}
    out = {}
    for d in REPO_DIRS:
        repo = dir_to_repo(d)
        with open(os.path.join(RAW_DIR, d, "merged_prs.json")) as f:
            prs = json.load(f)
        for pr in prs:
            key = (repo, int(pr["number"]))
            if key not in want or key in out:
                continue
            lit_h, rework_h = iteration_trace(pr, is_human)
            ev_h = reviewer_events(pr, is_human)
            out[key] = {
                "repo": repo,
                "created": parse_dt(want[key]["created_at"]),
                "lit_h": lit_h,
                "code_h": count_iterations(pr, is_human, rule=RULE_CODE),
                "lit_all": count_iterations(pr, include_all, rule=RULE_LITERAL),
                "code_all": count_iterations(pr, include_all, rule=RULE_CODE),
                "pub": int(float(want[key]["review_rounds"])),
                "rework_h": rework_h,
                "last_h_state": ev_h[-1][2] if ev_h else None,
            }
        print(f"  {repo}: {sum(k[0] == repo for k in out)}", flush=True)
    missing = set(want) - set(out)
    if missing:
        raise SystemExit(f"{len(missing)} PRs not found in raw JSON")
    return [out[(r["repo"], int(r["pr_number"]))] for r in rows]


def _np(o):
    """json default: numpy scalars to Python numbers, datetimes to ISO strings."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(type(o))


def pct(x):
    return round(100.0 * float(np.mean(x)), 2) if len(x) else None


def dist(v):
    v = np.asarray(v)
    return {"n": int(len(v)),
            "share_0_pct": pct(v == 0), "share_1_pct": pct(v == 1),
            "share_2_pct": pct(v == 2), "share_3plus_pct": pct(v >= 3),
            "share_ge2_pct": pct(v >= 2),
            "n_0": int((v == 0).sum()), "n_1": int((v == 1).sum()),
            "n_2": int((v == 2).sum()), "n_3plus": int((v >= 3).sum()),
            "n_ge2": int((v >= 2).sum()),
            "mean": round(float(v.mean()), 3),
            "median": float(np.median(v)), "q1": float(np.percentile(v, 25)),
            "q3": float(np.percentile(v, 75)), "p90": float(np.percentile(v, 90)),
            "max": int(v.max())}


def check_against_pr_level(tab):
    """Звірка з F-pr-level.csv (taskF1 prep), якщо він уже за буквальним правилом."""
    if not os.path.exists(PR_TABLE):
        return {"pr_level_found": False}
    with open(PR_TABLE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if "n_it_human_code" not in rows[0]:
        return {"pr_level_found": True, "pr_level_rule": "code (stale file)"}
    return {"pr_level_found": True, "pr_level_rule": "literal",
            "mismatch_literal_vs_n_it_human": sum(int(r["n_it_human"]) != t["lit_h"] for r, t in zip(rows, tab)),
            "mismatch_code_vs_n_it_human_code": sum(int(r["n_it_human_code"]) != t["code_h"] for r, t in zip(rows, tab))}


def section_253():
    """Табл. 5/6 ІКТ (протокол taskH/taskC5) з ітераціями людей за обома правилами."""
    rows = H.load_corpus(rule=RULE_LITERAL)
    t5_lit, t6_lit = H.run_protocol(rows, "rounds_human_literal")
    t5_code, t6_code = H.run_protocol(rows, "rounds_human_code")

    def r5(r):
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}

    by_repo = []
    for a, b in zip(t5_lit, t5_code):
        if a["metric"] != "rounds":
            continue
        by_repo.append({
            "repo": a["repo"], "n_ai": a["n_ai"], "n_no_ai": a["n_no_ai"],
            "med_ai": a["med_ai"], "med_no_ai": a["med_no_ai"],
            "rb": round(a["rb"], 4), "ci_lo": round(a["ci_lo"], 4), "ci_hi": round(a["ci_hi"], 4),
            "p": a["p"], "p_holm": a["p_holm"], "significant_holm_005": bool(a["p_holm"] < 0.05),
            "code_rule": {"med_ai": b["med_ai"], "med_no_ai": b["med_no_ai"], "rb": round(b["rb"], 4),
                          "ci_lo": round(b["ci_lo"], 4), "ci_hi": round(b["ci_hi"], 4),
                          "p_holm": b["p_holm"], "significant_holm_005": bool(b["p_holm"] < 0.05)},
        })
    diag = []
    for repo in H.ORDER:
        for g in ("ai", "no"):
            rr = [r for r in rows if r["repo"] == repo and r["g"] == g]
            lit = np.array([r["rounds_human_literal"] for r in rr])
            code = np.array([r["rounds_human_code"] for r in rr])
            pub = np.array([r["rounds_pub"] for r in rr])
            diag.append({"repo": repo, "group": g, "n": len(rr),
                         "ge2_literal_pct": pct(lit >= 2), "ge2_code_human_pct": pct(code >= 2),
                         "ge2_published_pct": pct(pub >= 2),
                         "mean_literal": round(float(lit.mean()), 3),
                         "zero_human_pct": pct(lit == 0)})
    return {
        "protocol": ("taskH_ikt_table5_human.run_protocol = taskC5: merged PRs created 2025-2026, "
                     "8 repos, groups by C-ai-pr-flags.csv has_ai_review, two-sided Mann-Whitney, "
                     "rank-biserial rb (positive = more iterations with assistant), bootstrap B=2000 "
                     "seed 20260903, Holm over 24 tests (table 5: 8 repos x {rounds, ttm, lines_added}); "
                     "table 6: strata by added-lines quartiles, n>=20 in both groups, separate Holm"),
        "n_rows": len(rows),
        "rounds_all_vs_published_mismatch": int(sum(r["rounds_all"] != r["rounds_pub"] for r in rows)),
        "table5_iterations_literal": by_repo,
        "table5_all_metrics_literal": [r5(r) for r in t5_lit],
        "table6_strata_literal": [r5(r) for r in t6_lit if r["rb"] is not None],
        "table6_strata_code_rule": [r5(r) for r in t6_code if r["rb"] is not None],
        "group_diagnostics": diag,
        "summary": {
            "n_repos_sig_positive_literal": sum(r["significant_holm_005"] and r["rb"] > 0 for r in by_repo),
            "n_repos_sig_negative_literal": sum(r["significant_holm_005"] and r["rb"] < 0 for r in by_repo),
            "n_repos_sig_positive_code_rule": sum(r["code_rule"]["significant_holm_005"] and r["code_rule"]["rb"] > 0
                                                  for r in by_repo),
            "rb_range_literal": [min(r["rb"] for r in by_repo), max(r["rb"] for r in by_repo)],
            "rb_range_code_rule": [min(r["code_rule"]["rb"] for r in by_repo),
                                   max(r["code_rule"]["rb"] for r in by_repo)],
            "table6_n_tests_literal": sum(r["rb"] is not None for r in t6_lit),
            "table6_n_sig_literal": sum(r["rb"] is not None and r["p_holm"] < 0.05 for r in t6_lit),
            "table6_n_sig_positive_literal": sum(r["rb"] is not None and r["p_holm"] < 0.05 and r["rb"] > 0
                                                 for r in t6_lit),
            "table6_n_sig_code_rule": sum(r["rb"] is not None and r["p_holm"] < 0.05 for r in t6_code),
        },
    }


def main():
    tab = per_pr_table()
    n = len(tab)
    lit = np.array([t["lit_h"] for t in tab])
    code = np.array([t["code_h"] for t in tab])
    lit_all = np.array([t["lit_all"] for t in tab])
    code_all = np.array([t["code_all"] for t in tab])
    pub = np.array([t["pub"] for t in tab])
    repo = np.array([t["repo"] for t in tab])
    era = np.array([t["created"] >= AI_ERA_START for t in tab])
    rework = np.array([t["rework_h"] for t in tab])
    last_appr = np.array([t["last_h_state"] == "APPROVED" for t in tab])
    diff = lit != code
    assert (code - lit == diff.astype(int)).all(), "code rule must add at most one iteration"
    assert (code_all == pub).all(), "code rule with all events must reproduce published review_rounds"

    repos = sorted(set(repo))
    per_repo = []
    for rp in repos:
        m = repo == rp
        d = dist(lit[m])
        per_repo.append({
            "repo": rp, "lang": LANG[rp], **d,
            "share_ge2_code_rule_human_pct": pct(code[m] >= 2),
            "share_ge2_all_events_literal_pct": pct(lit_all[m] >= 2),
            "share_ge2_published_pct": pct(pub[m] >= 2),
            "rule_changes_n_it_n": int(diff[m].sum()), "rule_changes_n_it_pct": pct(diff[m]),
            "rule_changes_label_ge2_n": int(((lit[m] >= 2) != (code[m] >= 2)).sum()),
            "share_ge2_before_2025_02_pct": pct(lit[m & ~era] >= 2),
            "share_ge2_from_2025_02_pct": pct(lit[m & era] >= 2),
        })
    ge2 = {r["repo"]: r["share_ge2_pct"] for r in per_repo}
    lo_r, hi_r = min(ge2, key=ge2.get), max(ge2, key=ge2.get)

    # Колонки r2_repo_characteristics.csv, що залежать від ітерацій (ті самі назви).
    char_cols = [{"repo": r["repo"], "it2_human_pct": r["share_ge2_pct"],
                  "it2_all_pct": r["share_ge2_all_events_literal_pct"],
                  "it2_all_pct_published_code_rule": r["share_ge2_published_pct"],
                  "no_human_review_n": r["n_0"],
                  "n_it_median": r["median"], "n_it_q1": r["q1"], "n_it_q3": r["q3"]}
                 for r in per_repo]
    char_cols.append({"repo": "Разом", "it2_human_pct": pct(lit >= 2),
                      "it2_all_pct": pct(lit_all >= 2),
                      "it2_all_pct_published_code_rule": pct(pub >= 2),
                      "no_human_review_n": int((lit == 0).sum()),
                      "n_it_median": float(np.median(lit)), "n_it_q1": float(np.percentile(lit, 25)),
                      "n_it_q3": float(np.percentile(lit, 75))})

    lab_all_h = (lit_all >= 2) != (lit >= 2)
    out = {
        "rule": ("Буквальне означення HAIT Def. 2: ітерацію відкриває перша рецензія людини; нова "
                 "ітерація настає лише тоді, коли після рецензії людини автор зробив нові коміти "
                 "(authoredDate > t0) і ПОТІМ була нова рецензія людини. Коміти після останньої "
                 "рецензії людини, за якими йде злиття без нової рецензії, ітерації не додають. "
                 "Рецензії: усі три вердикти; самоперевірки, ШІ-асистенти, інші боти, *[bot], "
                 "grafana-pr-approver, woocommercebot, angular-builds не враховуються. За рівних "
                 "моментів рецензія йде раніше коміту."),
        "rule_code_previous": ("Правило коду (compute_review_rounds, рядок 11 алгоритму 2.3.2): "
                               "додатково +1 ітерація, якщо після останньої рецензії були коміти "
                               "і запит злито без нової рецензії."),
        "source_script": "revision/taskI_r2_iterations_literal.py",
        "implementation": "revision/review_iterations.py::count_iterations (rule='literal')",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corpus": "12 репозиторіїв HAIT, 64 820 злитих запитів (all_merged_features.csv + data/raw)",
        "checks": {"n_rows": n, "code_all_vs_published_review_rounds_mismatch": int((code_all != pub).sum()),
                   **check_against_pr_level(tab)},
        "pool": {
            "distribution_literal": dist(lit),
            "distribution_code_rule_human": dist(code),
            "positive_class_ge2_pct": pct(lit >= 2),
            "positive_class_ge2_code_rule_pct": pct(code >= 2),
            "positive_class_ge3_pct": pct(lit >= 3),
            "by_period": {"before_2025_02": {"n": int((~era).sum()), "share_ge2_pct": pct(lit[~era] >= 2),
                                             "share_ge2_all_events_literal_pct": pct(lit_all[~era] >= 2)},
                          "from_2025_02": {"n": int(era.sum()), "share_ge2_pct": pct(lit[era] >= 2),
                                           "share_ge2_all_events_literal_pct": pct(lit_all[era] >= 2)}},
        },
        "rule_difference": {
            "n_prs_trailing_rework_after_last_human_review": int(rework.sum()),
            "n_prs_n_it_differs": int(diff.sum()), "pct_prs_n_it_differs": pct(diff),
            "of_them_last_human_review_approved_n": int((diff & last_appr).sum()),
            "n_label_ge2_changes": int(((lit >= 2) != (code >= 2)).sum()),
            "note": "literal = code - 1 рівно в цих запитах (перевірено assert)",
        },
        "human_vs_all_events_literal": {
            "share_ge2_all_events_literal_pct": pct(lit_all >= 2),
            "share_ge2_human_literal_pct": pct(lit >= 2),
            "label_changes_n": int(lab_all_h.sum()), "label_changes_pct": pct(lab_all_h),
            "label_changes_from_2025_02_n": int((lab_all_h & era).sum()),
            "vscode_share_ge2_all_events_literal_pct": next(r["share_ge2_all_events_literal_pct"]
                                                            for r in per_repo if r["repo"] == "microsoft/vscode"),
            "vscode_share_ge2_human_literal_pct": ge2["microsoft/vscode"],
            "published_hait_share_ge2_pct": pct(pub >= 2),
        },
        "repo_extremes_share_ge2": {"min_repo": lo_r, "min_pct": ge2[lo_r], "max_repo": hi_r, "max_pct": ge2[hi_r]},
        "per_repo": per_repo,
        "r2_repo_characteristics_columns": char_cols,
        "delta_sensitivity": ("Не стосується: поріг Δ визначає лише мітку застарілості, яка від "
                              "ітерацій не залежить (r2_delta_sensitivity.csv без змін)."),
        "section_2_5_3": section_253(),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=_np)
    print(json.dumps({k: out[k] for k in ("checks", "pool", "rule_difference", "human_vs_all_events_literal",
                                          "repo_extremes_share_ge2")}, ensure_ascii=False, indent=1,
                     default=_np))
    print(json.dumps(out["section_2_5_3"]["summary"], indent=1, default=_np))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
