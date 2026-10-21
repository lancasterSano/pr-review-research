#!/usr/bin/env python3
"""E1, step 2 — Table «Реакції на вихід асистентів» by repository (+ a by-tool companion for the text).

Run: .venv/bin/python revision/taskE1_reactions_table.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from taskE1_common import OUT, THREADS_CSV, REVIEWS_CSV, TABLE3_REPOS  # noqa: E402

SHORT = {r: r.split("/")[1] for r in TABLE3_REPOS}
# draft_ua.md (v11) numbers the quartile table as «Таблиця 6»; this one follows it.
TABLE_NO = 7


def plural(n, one, few, many):
    """Ukrainian numeral agreement: 1 обговорення, 3 обговорення, 5 обговорень, 104 обговорення."""
    if n % 10 == 1 and n % 100 != 11:
        return one
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return few
    return many


def pct(x):
    if x != x:  # NaN: no threads with a suggestion block (e.g. OpenAI Codex)
        return "н/д"
    return f"{100 * x:.1f}".replace(".", ",")


def rows_for(df, rv, keys, name):
    out = []
    for k in keys:
        d = df[df[name] == k]
        r = rv[rv[name] == k]
        human = d[d["author_is_bot"] == 0]
        sugg = human[human["has_suggestion"] == 1]
        out.append({
            "key": k, "n_threads": len(d), "n_reviews": len(r),
            "within_1h": (d["hours_since_trigger"] <= 1).mean(),
            "reacted": d["reacted"].mean(), "resolved": d["is_resolved"].mean(),
            "has_sugg": d["has_suggestion"].mean(),
            "applied": sugg["sugg_applied"].mean() if len(sugg) else float("nan"),
            "verdict": r["state"].isin(["APPROVED", "CHANGES_REQUESTED"]).mean() if len(r) else float("nan"),
        })
    return out


def table(rows, label):
    L = ["| " + label + " | Обговорень / подій перевірки | ≤ 1 год після тригера, % | Відповідь людини / вирішено, % | Готове виправлення / з них застосовано, % | Подій перевірки з вердиктом, % |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['key']} | {r['n_threads']:,} / {r['n_reviews']:,}".replace(",", " ") +
                 f" | {pct(r['within_1h'])} | {pct(r['reacted'])} / {pct(r['resolved'])} | {pct(r['has_sugg'])} / {pct(r['applied'])} | {pct(r['verdict'])} |")
    return L


def main():
    df = pd.read_csv(THREADS_CSV)
    rv = pd.read_csv(REVIEWS_CSV)
    rows = rows_for(df, rv, TABLE3_REPOS, "repo")
    for r in rows:
        r["key"] = SHORT[r["key"]]
    total = rows_for(df.assign(all="Разом"), rv.assign(all="Разом"), ["Разом"], "all")
    # «Разом» covers all twelve repositories; the rows show only the eight of Table 5. Name the rest in the note.
    rest = df[~df["repo"].isin(TABLE3_REPOS)]
    rest_names = [r.split("/")[1] for r in rest["repo"].value_counts().index]
    n_rest_threads, n_rest_reviews = len(rest), int((~rv["repo"].isin(TABLE3_REPOS)).sum())
    rest_note = (f"Разом: усі дванадцять репозиторіїв; {', '.join(rest_names[:-1])} і {rest_names[-1]} (разом {n_rest_threads} "
                 f"{plural(n_rest_threads, 'обговорення', 'обговорення', 'обговорень')} і {n_rest_reviews} "
                 f"{plural(n_rest_reviews, 'подія', 'події', 'подій')} перевірки) окремо не показано.")
    L = [f"Таблиця {TABLE_NO}. Реакції людей на вихід асистентів у злитих запитах, 2025–2026 роки", ""] + table(rows + total, "Репозиторій") + [
        "",
        "Джерело: складено авторами. " + rest_note + " Тригер: відкриття запиту або нова ревізія. Застосовано: частка зауважень із блоком suggestion у запитах, створених людьми, "
        "після яких є коміт «Apply suggestions from code review» або зі співавторством асистента. Вердикт: стан APPROVED або CHANGES_REQUESTED"]
    open(os.path.join(OUT, "E1-table4.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    tools = df["tool"].value_counts()
    tools = [t for t in tools.index if tools[t] >= 300]
    by_tool = table(rows_for(df, rv, tools, "tool"), "Інструмент")
    open(os.path.join(OUT, "E1-table4-by-tool.md"), "w", encoding="utf-8").write("\n".join(by_tool) + "\n")
    print("\n".join(L))
    print("\n".join(by_tool))


if __name__ == "__main__":
    main()
