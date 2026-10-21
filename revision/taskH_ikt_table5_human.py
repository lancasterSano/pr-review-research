#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task H: перерахунок табл. 5 і 6 статті ІКТ-2026 з ітераціями лише за подіями людей.

Рішення автора 14.10.2026 (для всієї дисертації): ітерації, час до першої
реакції і застарілість рахуються тільки за подіями людей. Подія перевірки
враховується, якщо її автор не є автором запиту і
taskF1_duration_language.participant_class(login) == "human" (без ШІ-асистентів,
інших ботів, логінів `*[bot]` і службових акаунтів EXTRA_AUTOMATION).

Протокол точно як у taskC5_ikt_ci_strata.py (табл. 5 і 6 ІКТ, подана v14):
злиті запити аналітичного набору (all_merged_features.csv), створені у
2025–2026 роках; групи з / без перевірки асистентом за C-ai-pr-flags.csv;
8 репозиторіїв; двобічний Манн–Вітні; рангова бісеріальна кореляція;
бутстреп 2 000 повторень, seed 20260903; Холм на 24 тести (табл. 5) і окремо
на тести у квартилях доданих рядків (табл. 6, n >= 20 в обох групах).

Варіанти:
  pub   — опубліковане означення (review_rounds з матриці, усі події);
          має збігтися з C5-*.csv і v14 до останнього знака.
  human — ітерації лише за подіями людей (count_iterations(pr, is_human, rule)).
          H-*.csv у IKT2026/data пораховано 14.10 з rule="code" (правило коду,
          за замовчуванням load_corpus); буквальне правило (rule="literal",
          рішення автора 14.10 увечері) рахує taskI_r2_iterations_literal.py,
          який повторно використовує load_corpus і run_protocol звідси.
Час до злиття і додані рядки від подій не залежать; їхні міри ефекту й
інтервали в обох варіантах однакові, але p після Холма перераховано, бо
змінюються p ітерацій у тій самій сім'ї з 24 тестів.

Додатково (окрема сім'я Холма, 8 тестів): час до першої реакції людини
серед запитів, що мають хоча б одну перевірку людини, і частка запитів без
жодної перевірки людини.

Наявні скрипти (enrich/review_rounds.py, taskC4/C5, taskF1) не змінюються;
функції подій імпортуються з taskF1_duration_language.py, а збіг
count_iterations(include_all) з опублікованим review_rounds перевіряється тут.

Вихід (нові файли, наявні не перезаписуються):
  <DISSERTATION_DIR>/conf_papers/october_2026/IKT2026/data/H-*.csv, H-summary.md, H-checks.json
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taskC_common import MERGED_CSV, RAW_DIR, REPO_DIRS, dir_to_repo, load_processed  # noqa: E402
from taskC5_ikt_ci_strata import (  # noqa: E402
    B, FLAGS_CSV, MIN_N, ORDER, OUT, SEED, boot_ci, holm, rb_and_p,
)
from taskF1_duration_language import (  # noqa: E402
    first_reaction_hours, is_human,
)
from review_iterations import RULE_CODE, RULE_LITERAL, count_iterations, include_all  # noqa: E402

METRICS = ["rounds", "ttm", "lines_added"]
SHORT = {r: r.split("/")[1] for r in ORDER}


def out_path(name):
    p = os.path.join(OUT, name)
    if os.path.exists(p) and not os.environ.get("H_OVERWRITE"):
        raise SystemExit(f"refusing to overwrite existing {p} (set H_OVERWRITE=1 to rerun task H)")
    return p


def load_corpus(rule=RULE_CODE):
    flags = {}
    for r in load_processed(FLAGS_CSV):
        if r["pr_type"] == "merged":
            flags[(r["repo"], int(r["pr_number"]))] = int(r["has_ai_review"])
    rows = []
    for r in load_processed(MERGED_CSV):
        if (r["created_at"] or "")[:4] not in ("2025", "2026"):
            continue
        key = (r["repo"], int(r["pr_number"]))
        if key not in flags or r["repo"] not in ORDER:
            continue
        try:
            vals = {"rounds_pub": float(r["review_rounds"]), "ttm": float(r["time_to_merge_hours"]),
                    "lines_added": float(r["lines_added"])}
        except (TypeError, ValueError):
            continue
        vals.update({"repo": r["repo"], "pr": key[1], "g": "ai" if flags[key] else "no"})
        rows.append(vals)
    want = {(r["repo"], r["pr"]) for r in rows}
    ev = {}
    for d in REPO_DIRS:
        repo = dir_to_repo(d)
        if repo not in ORDER:
            continue
        with open(os.path.join(RAW_DIR, d, "merged_prs.json")) as f:
            for pr in json.load(f):
                key = (repo, int(pr["number"]))
                if key not in want or key in ev:
                    continue
                ev[key] = {
                    "rounds_all": count_iterations(pr, include_all, rule=RULE_CODE),
                    "rounds_human": count_iterations(pr, is_human, rule=rule),
                    "rounds_human_literal": count_iterations(pr, is_human, rule=RULE_LITERAL),
                    "rounds_human_code": count_iterations(pr, is_human, rule=RULE_CODE),
                    "tfr_any": first_reaction_hours(pr, include_all),
                    "tfr_human": first_reaction_hours(pr, is_human),
                }
    miss = want - set(ev)
    if miss:
        raise SystemExit(f"{len(miss)} PRs not found in raw JSON")
    for r in rows:
        r.update(ev[(r["repo"], r["pr"])])
    return rows


def run_protocol(rows, rounds_col):
    """Те саме, що taskC5.main(), з параметризованим стовпцем ітерацій."""
    groups = defaultdict(lambda: {"ai": defaultdict(list), "no": defaultdict(list)})
    for r in rows:
        g = groups[r["repo"]][r["g"]]
        g["rounds"].append(r[rounds_col]); g["ttm"].append(r["ttm"]); g["lines_added"].append(r["lines_added"])
    rng = np.random.default_rng(SEED)
    t5, ptests = [], []
    for repo in ORDER:
        g = groups[repo]
        for m in METRICS:
            a, b = g["ai"][m], g["no"][m]
            rb, p = rb_and_p(a, b)
            lo, hi = boot_ci(a, b, rng)
            t5.append({"repo": repo, "metric": m, "n_ai": len(a), "n_no_ai": len(b),
                       "med_ai": float(np.median(a)), "med_no_ai": float(np.median(b)),
                       "rb": rb, "ci_lo": lo, "ci_hi": hi, "p": p})
            ptests.append(p)
    for r, pa in zip(t5, holm(ptests)):
        r["p_holm"] = pa
    t6, sp = [], []
    for repo in ORDER:
        g = groups[repo]
        la = np.array(g["ai"]["lines_added"] + g["no"]["lines_added"])
        q = np.quantile(la, [0.25, 0.5, 0.75])

        def stratum(v):
            return 0 if v <= q[0] else 1 if v <= q[1] else 2 if v <= q[2] else 3
        for s in range(4):
            a = [x for x, l in zip(g["ai"]["rounds"], g["ai"]["lines_added"]) if stratum(l) == s]
            b = [x for x, l in zip(g["no"]["rounds"], g["no"]["lines_added"]) if stratum(l) == s]
            row = {"repo": repo, "stratum": f"Q{s + 1}", "n_ai": len(a), "n_no_ai": len(b),
                   "med_ai": float(np.median(a)) if a else None, "med_no_ai": float(np.median(b)) if b else None,
                   "rb": None, "ci_lo": None, "ci_hi": None, "p": None, "p_holm": None}
            if len(a) >= MIN_N and len(b) >= MIN_N:
                rb, p = rb_and_p(a, b); lo, hi = boot_ci(a, b, rng)
                row.update({"rb": rb, "ci_lo": lo, "ci_hi": hi, "p": p}); sp.append((len(t6), p))
            t6.append(row)
    for (i, _), pa in zip(sp, holm([p for _, p in sp])):
        t6[i]["p_holm"] = pa
    return t5, t6


def supp_tfr(rows):
    """Час до першої реакції людини (лише запити з перевіркою людини) і частка без неї."""
    rng = np.random.default_rng(SEED)
    out = []
    for repo in ORDER:
        rr = [r for r in rows if r["repo"] == repo]
        a = [r["tfr_human"] for r in rr if r["g"] == "ai" and r["tfr_human"] is not None]
        b = [r["tfr_human"] for r in rr if r["g"] == "no" and r["tfr_human"] is not None]
        aa = [r["tfr_any"] for r in rr if r["g"] == "ai" and r["tfr_any"] is not None]
        ba = [r["tfr_any"] for r in rr if r["g"] == "no" and r["tfr_any"] is not None]
        n_ai = sum(r["g"] == "ai" for r in rr); n_no = sum(r["g"] == "no" for r in rr)
        rb, p = rb_and_p(a, b); lo, hi = boot_ci(a, b, rng)
        rb_any, _ = rb_and_p(aa, ba)
        out.append({"repo": repo, "n_ai": n_ai, "n_no_ai": n_no,
                    "no_human_review_ai_pct": 100 * (1 - len(a) / n_ai),
                    "no_human_review_no_ai_pct": 100 * (1 - len(b) / n_no),
                    "n_ai_with_human": len(a), "n_no_ai_with_human": len(b),
                    "med_tfr_human_ai": float(np.median(a)), "med_tfr_human_no_ai": float(np.median(b)),
                    "rb_tfr_human": rb, "ci_lo": lo, "ci_hi": hi, "p": p,
                    "med_tfr_any_ai": float(np.median(aa)), "med_tfr_any_no_ai": float(np.median(ba)),
                    "rb_tfr_any": rb_any})
    for r, pa in zip(out, holm([r["p"] for r in out])):
        r["p_holm"] = pa
    return out


def fmtv(v):
    return f"{v:.4g}" if isinstance(v, float) else ("" if v is None else v)


def write_csv(name, rows, cols=None):
    cols = cols or list(rows[0].keys())
    with open(out_path(name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: fmtv(r.get(k)) for k in cols})


def check_against_c5(t5, t6):
    """Порівняння варіанта pub з опублікованими C5-*.csv (до 4 значущих цифр)."""
    diffs = []
    c5 = {r["repo"]: r for r in load_processed(os.path.join(OUT, "C5-table3-ci-holm.csv"))}
    key = {"rounds": "rounds", "ttm": "ttm", "lines_added": "lines_added"}
    for r in t5:
        ref = c5[r["repo"]]; m = key[r["metric"]]
        pairs = [("n_ai", ref["n_ai"]), ("n_no_ai", ref["n_no_ai"]), ("med_ai", ref[f"med_{m}_ai"]),
                 ("med_no_ai", ref[f"med_{m}_no"]), ("rb", ref[f"rb_{m}"]), ("ci_lo", ref[f"ci_lo_{m}"]),
                 ("ci_hi", ref[f"ci_hi_{m}"]), ("p_holm", ref[f"p_holm_{m}"])]
        for k, v in pairs:
            mine = str(r[k]) if k in ("n_ai", "n_no_ai") else fmtv(float(r[k]))
            if mine != (v if k in ("n_ai", "n_no_ai") else fmtv(float(v))):
                diffs.append((r["repo"], m, k, r[k], v))
    s5 = load_processed(os.path.join(OUT, "C5-strata-lines-added.csv"))
    for r, ref in zip(t6, s5):
        for k, rk in [("rb", "rb_rounds"), ("ci_lo", "ci_lo"), ("ci_hi", "ci_hi"), ("p_holm", "p_holm")]:
            if (ref[rk] or "") != fmtv(r[k]):
                diffs.append((r["repo"], r["stratum"], k, r[k], ref[rk]))
    return diffs


def sig_label(p):
    return "p<0.001" if p < 0.001 else f"p={p:.2g}"


def main():
    rows = load_corpus()
    mism = sum(int(r["rounds_all"] != r["rounds_pub"]) for r in rows)
    t5p, t6p = run_protocol(rows, "rounds_pub")
    t5h, t6h = run_protocol(rows, "rounds_human")
    diffs = check_against_c5(t5p, t6p)
    sup = supp_tfr(rows)

    # діагностика: як змінюються ітерації в групах
    diag = []
    for repo in ORDER:
        for g in ("ai", "no"):
            rr = [r for r in rows if r["repo"] == repo and r["g"] == g]
            diag.append({"repo": repo, "group": g, "n": len(rr),
                         "changed_pct": 100 * np.mean([r["rounds_human"] != r["rounds_pub"] for r in rr]),
                         "decreased_pct": 100 * np.mean([r["rounds_human"] < r["rounds_pub"] for r in rr]),
                         "zero_human_pct": 100 * np.mean([r["rounds_human"] == 0 for r in rr]),
                         "mean_pub": float(np.mean([r["rounds_pub"] for r in rr])),
                         "mean_human": float(np.mean([r["rounds_human"] for r in rr])),
                         "ge2_pub_pct": 100 * np.mean([r["rounds_pub"] >= 2 for r in rr]),
                         "ge2_human_pct": 100 * np.mean([r["rounds_human"] >= 2 for r in rr])})

    long5 = [dict(variant="pub", **r) for r in t5p] + [dict(variant="human", **r) for r in t5h]
    long6 = [dict(variant="pub", **r) for r in t6p] + [dict(variant="human", **r) for r in t6h]
    write_csv("H-table5-pub-vs-human.csv", long5)
    write_csv("H-table6-strata-pub-vs-human.csv", long6)
    write_csv("H-tfr-human-supp.csv", sup)
    write_csv("H-rounds-diagnostics.csv", diag)

    # було / стало
    cmp_rows = []
    for a, b in zip([r for r in t5p], [r for r in t5h]):
        cmp_rows.append({"table": "5", "repo": a["repo"], "cell": a["metric"],
                         "med_pub": f"{a['med_ai']:g}/{a['med_no_ai']:g}", "med_human": f"{b['med_ai']:g}/{b['med_no_ai']:g}",
                         "rb_pub": a["rb"], "ci_pub": f"[{a['ci_lo']:+.2f}; {a['ci_hi']:+.2f}]", "p_holm_pub": a["p_holm"],
                         "rb_human": b["rb"], "ci_human": f"[{b['ci_lo']:+.2f}; {b['ci_hi']:+.2f}]", "p_holm_human": b["p_holm"],
                         "sig_pub": int(a["p_holm"] < 0.05), "sig_human": int(b["p_holm"] < 0.05),
                         "sign_flip": int(np.sign(a["rb"]) != np.sign(b["rb"])),
                         "sig_change": int((a["p_holm"] < 0.05) != (b["p_holm"] < 0.05))})
    for a, b in zip(t6p, t6h):
        if a["rb"] is None and b["rb"] is None:
            continue
        cmp_rows.append({"table": "6", "repo": a["repo"], "cell": a["stratum"],
                         "med_pub": f"{a['med_ai']:g}/{a['med_no_ai']:g}", "med_human": f"{b['med_ai']:g}/{b['med_no_ai']:g}",
                         "rb_pub": a["rb"], "ci_pub": f"[{a['ci_lo']:+.2f}; {a['ci_hi']:+.2f}]", "p_holm_pub": a["p_holm"],
                         "rb_human": b["rb"], "ci_human": f"[{b['ci_lo']:+.2f}; {b['ci_hi']:+.2f}]", "p_holm_human": b["p_holm"],
                         "sig_pub": int(a["p_holm"] < 0.05), "sig_human": int(b["p_holm"] < 0.05),
                         "sign_flip": int(np.sign(a["rb"]) != np.sign(b["rb"])),
                         "sig_change": int((a["p_holm"] < 0.05) != (b["p_holm"] < 0.05))})
    write_csv("H-compare-before-after.csv", cmp_rows)

    checks = {"n_rows_2025_2026_8repos": len(rows), "rounds_all_vs_published_mismatch": mism,
              "pub_variant_vs_C5_csv_diffs": [list(map(str, d)) for d in diffs],
              "bootstrap_B": B, "seed": SEED, "min_n": MIN_N,
              "human_rule": "non-author AND taskF1.participant_class(login)=='human'",
              "n_tests_table5": len(t5p), "n_tests_table6_pub": sum(r["rb"] is not None for r in t6p),
              "n_tests_table6_human": sum(r["rb"] is not None for r in t6h)}
    with open(out_path("H-checks.json"), "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2, ensure_ascii=False)

    L = ["# H: табл. 5 і 6 ІКТ-2026, усі події (pub) і лише люди (human)", "",
         f"Рядків: {len(rows)}; розбіжностей count_iterations(all) з review_rounds: {mism}; "
         f"розбіжностей pub з C5-*.csv: {len(diffs)}.", "",
         "## Табл. 5 (ітерації): було / стало", "",
         "| repo | n ai/no | pub: med; rb [CI]; p_H | human: med; rb [CI]; p_H |", "|---|---|---|---|"]
    for a, b in zip(t5p, t5h):
        if a["metric"] != "rounds":
            continue
        L.append(f"| {SHORT[a['repo']]} | {a['n_ai']}/{a['n_no_ai']} | {a['med_ai']:g}/{a['med_no_ai']:g}; "
                 f"{a['rb']:+.2f} [{a['ci_lo']:+.2f}; {a['ci_hi']:+.2f}]; {sig_label(a['p_holm'])} | "
                 f"{b['med_ai']:g}/{b['med_no_ai']:g}; {b['rb']:+.2f} [{b['ci_lo']:+.2f}; {b['ci_hi']:+.2f}]; {sig_label(b['p_holm'])} |")
    L += ["", "## Табл. 5 (час до злиття, рядки): p після Холма pub / human", "",
          "| repo | metric | rb | p_H pub | p_H human |", "|---|---|---|---|---|"]
    for a, b in zip(t5p, t5h):
        if a["metric"] == "rounds":
            continue
        L.append(f"| {SHORT[a['repo']]} | {a['metric']} | {a['rb']:+.2f} | {sig_label(a['p_holm'])} | {sig_label(b['p_holm'])} |")
    L += ["", "## Табл. 6: було / стало", "", "| repo | Q | n ai/no | pub rb [CI] p_H | human med; rb [CI] p_H |",
          "|---|---|---|---|---|"]
    for a, b in zip(t6p, t6h):
        if b["rb"] is None:
            continue
        L.append(f"| {SHORT[a['repo']]} | {a['stratum']} | {a['n_ai']}/{a['n_no_ai']} | {a['rb']:+.2f} [{a['ci_lo']:+.2f}; "
                 f"{a['ci_hi']:+.2f}] {sig_label(a['p_holm'])} | {b['med_ai']:g}/{b['med_no_ai']:g}; {b['rb']:+.2f} "
                 f"[{b['ci_lo']:+.2f}; {b['ci_hi']:+.2f}] {sig_label(b['p_holm'])} |")
    L += ["", "## Діагностика ітерацій", "",
          "| repo | group | n | змінено % | зменшено % | 0 за людьми % | >=2 pub % | >=2 human % |", "|---|---|---|---|---|---|---|---|"]
    for d in diag:
        L.append(f"| {SHORT[d['repo']]} | {d['group']} | {d['n']} | {d['changed_pct']:.1f} | {d['decreased_pct']:.1f} | "
                 f"{d['zero_human_pct']:.1f} | {d['ge2_pub_pct']:.1f} | {d['ge2_human_pct']:.1f} |")
    L += ["", "## Додатково: час до першої реакції людини (окремий Холм, 8 тестів)", "",
          "| repo | без перевірки людини ai/no % | med t_fr human ai/no, год | rb [CI]; p_H | rb за всіма подіями |",
          "|---|---|---|---|---|"]
    for s in sup:
        L.append(f"| {SHORT[s['repo']]} | {s['no_human_review_ai_pct']:.1f} / {s['no_human_review_no_ai_pct']:.1f} | "
                 f"{s['med_tfr_human_ai']:.1f} / {s['med_tfr_human_no_ai']:.1f} | {s['rb_tfr_human']:+.2f} "
                 f"[{s['ci_lo']:+.2f}; {s['ci_hi']:+.2f}]; {sig_label(s['p_holm'])} | {s['rb_tfr_any']:+.2f} |")
    with open(out_path("H-summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(json.dumps(checks, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
