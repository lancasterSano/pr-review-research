#!/usr/bin/env python3
"""Chapter 4, E1: spread of the deviation categories over repeated alignment runs.

Among equal-cost alignments pm4py returns one arbitrarily (it depends on object
addresses, so a fixed PYTHONHASHSEED does not help). Fitness, the share of fitting
traces and precision do not depend on that choice; the categories of the non-fitting
traces of the main_noreapprove variant do. This script reads the r2_alignment_v2.json
files of several runs of r2_alignment_v2.py on the same sample and reports, for every
category share, the value of the reference run and the minimum and maximum over runs.

Input (read only): every *.json in --runs (default revision/output/r4/e1_repeats/runs/);
the reference run is the file named by --ref (default ref.json in the same folder).
Output: revision/output/r4/e1_repeats/numbers.json (keys e1.rep.*).

Run from the repository root:
  python revision/r4_calc/r4_e1_repeats.py
"""
import argparse
import glob
import json
import os

REV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REV, "output", "r4", "e1_repeats")


def categories(path):
    with open(path, encoding="utf-8") as f:
        a = json.load(f)["align"]
    nr = a["main_noreapprove"]
    cat = nr["categories"]
    v = {"fit_pct": nr["fit_pct"], "fitness": nr["fitness"], "nonfit_n": cat["nonfit_n"],
         "explained": cat["explained_pct"], "explained_abce": cat["explained_abce_pct"]}
    for c in "abcde":
        v[f"cat_{c}"] = cat["traces_with_pct"][c]
    return a["n_traces"], v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(OUT, "runs"))
    ap.add_argument("--ref", default="ref.json", help="file name of the reference run inside --runs")
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.runs, "*.json")))
    ref_path = os.path.join(args.runs, args.ref)
    assert ref_path in paths, f"reference run {ref_path} not found"
    runs = {os.path.basename(p): categories(p) for p in paths}
    sizes = {n for n, _ in runs.values()}
    assert len(sizes) == 1, f"runs differ in the sample size: {sizes}"
    num = {"e1.rep.n_runs": len(runs), "e1.rep.n_traces": sizes.pop(),
           "e1.rep.runs": sorted(runs), "e1.rep.ref": args.ref,
           "e1.rep.note": "categories of non-fitting traces, variant main_noreapprove; "
                          "pm4py returns one of several equal-cost alignments arbitrarily"}
    for k in runs[args.ref][1]:
        vals = [v[k] for _, v in runs.values()]
        num[f"e1.rep.{k}.ref"] = runs[args.ref][1][k]
        num[f"e1.rep.{k}.min"] = min(vals)
        num[f"e1.rep.{k}.max"] = max(vals)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "numbers.json"), "w", encoding="utf-8") as f:
        json.dump(num, f, ensure_ascii=False, indent=1)
    for k in ("fit_pct", "explained", "explained_abce", "cat_a", "cat_b", "cat_c", "cat_e"):
        print(f"{k}: ref {num[f'e1.rep.{k}.ref']:.2f}, "
              f"{num[f'e1.rep.{k}.min']:.2f}..{num[f'e1.rep.{k}.max']:.2f} over {len(runs)} runs")


if __name__ == "__main__":
    main()
