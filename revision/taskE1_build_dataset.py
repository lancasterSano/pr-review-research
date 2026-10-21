#!/usr/bin/env python3
"""E1, step 1 — per-thread and per-review tables plus a Markdown summary.

Run: .venv/bin/python revision/taskE1_build_dataset.py
"""
import csv
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from taskE1_common import OUT, THREADS_CSV, REVIEWS_CSV, iter_ai_threads, iter_ai_reviews  # noqa: E402

OUTCOMES = ["reacted", "is_resolved", "any_signal", "has_suggestion", "sugg_applied", "human_same_file"]


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    t = time.time()
    os.makedirs(OUT, exist_ok=True)
    threads = list(iter_ai_threads())
    reviews = list(iter_ai_reviews())
    write_csv(THREADS_CSV, threads)
    write_csv(REVIEWS_CSV, reviews)
    df = pd.DataFrame(threads)
    rv = pd.DataFrame(reviews)
    df["year"] = df["created_at"].str[:4]
    df["within_1h"] = (df["hours_since_trigger"] <= 1).astype(int)
    df["any_signal"] = ((df["reacted"] == 1) | (df["is_resolved"] == 1)).astype(int)  # platform recorded a reaction

    L = ["# E1 — обговорення та рев'ю асистентів у злитих запитах (12 репозиторіїв)", "",
         f"Обговорень: {len(df)}; запитів: {df.groupby(['repo', 'pr_number']).ngroups}; рев'ю асистентів: {len(rv)}",
         f"Усього: ≤1h after trigger {df['within_1h'].mean():.3f}; " + "; ".join(f"{o} {df[o].mean():.3f}" for o in OUTCOMES),
         f"Рев'ю з вердиктом (APPROVED або CHANGES_REQUESTED): {rv['state'].isin(['APPROVED', 'CHANGES_REQUESTED']).mean():.4f}; "
         f"COMMENTED: {(rv['state'] == 'COMMENTED').mean():.4f}"]
    # Resolution has no polarity: split resolved threads by what happened after the assistant's comment.
    res = df[df["is_resolved"] == 1]
    L += ["Вирішені обговорення за групами (частка від усіх обговорень): "
          f"після нового коміту без відповіді {((res['reacted'] == 0) & (res['commits_after'] > 0)).sum() / len(df):.3f}; "
          f"з відповіддю людини {(res['reacted'] == 1).sum() / len(df):.3f}; "
          f"мовчазно, без відповіді і без коміту {((res['reacted'] == 0) & (res['commits_after'] == 0)).sum() / len(df):.3f}", ""]
    for key in ["repo", "tool", "year"]:
        g = df.groupby(key).agg(n=("reacted", "size"), within_1h=("within_1h", "mean"),
                                **{o: (o, "mean") for o in OUTCOMES}).sort_values("n", ascending=False)
        L += [f"## Обговорення за {key}", "", "| " + key + " | n | ≤1h after trigger | " + " | ".join(OUTCOMES) + " |",
              "|---|---|---|" + "---|" * len(OUTCOMES)]
        for k, r in g.iterrows():
            L.append(f"| {k} | {int(r['n'])} | {r['within_1h']:.3f} | " + " | ".join(f"{r[o]:.3f}" for o in OUTCOMES) + " |")
        L.append("")
    ct = pd.crosstab(rv["tool"], rv["state"])
    L += ["## Стани рев'ю асистентів за інструментами", "", ct.to_markdown(), ""]
    L.append(f"Час побудови: {time.time() - t:.0f} с")
    open(os.path.join(OUT, "E1-dataset-summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("threads:", len(df), "reviews:", len(rv))


if __name__ == "__main__":
    main()
