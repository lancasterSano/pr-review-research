#!/usr/bin/env python3
"""Chapter 4, E3: robustness of the E3 verdicts to the CatBoost hyperparameters (the sets,
targets and criteria were fixed in the author's notes before the run).

LORO AUC (40 repositories, target IT, TFR, ST, TM; features t0, clock D, population main) for
three alternative sets fixed before the run. Everything except the named parameters is the main
model of chapter 3 (`r3_v2_models`: table, features, train-median imputation, folds, seed 42,
class weights). The main set H0 is not refitted: its numbers come from the stored r3_b40d
predictions, and two folds per target are refitted only to check the reproduction.

    PY=python   # interpreter with requirements.txt
    $PY revision/r4_calc/r4_hparams.py --set H0check     # reproduction of 2 folds per target
    $PY revision/r4_calc/r4_hparams.py --set H1          # one process per set; resumable
    $PY revision/r4_calc/r4_hparams.py --set H2
    $PY revision/r4_calc/r4_hparams.py --set H3
    $PY revision/r4_calc/r4_hparams.py --summarise       # numbers.json, checks.json

Output: revision/output/r4/hparams/{folds_<set>.jsonl, oof/, numbers.json, checks.json, runs.jsonl}.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
sys.path.insert(0, REV)
import r3_report_stats as S  # noqa: E402
import r3_v2_models as M  # noqa: E402

RUN = os.path.join(os.environ.get("R3_ROOT") or os.path.join(REV, "output", "r3_b40d"), "v2")
TABLE = os.path.join(RUN, "pr_table_v2.csv")
METRICS = os.path.join(RUN, "models", "metrics.jsonl")
OUT = os.path.join(REV, "output", "r4", "hparams")
# chapter 4 numbers of the dissertation (nested JSON), used only for the reconciliation checks
R4_NUMBERS = os.environ.get("R4_NUMBERS_JSON") or os.path.join(REV, "r4_calc", "r4_numbers.json")

THREADS = 6
B, BOOT_SEED = 10000, 42
SIGNIF = 0.02                       # sens.signif_diff of chapter 4
BASE = dict(iterations=500, depth=6, learning_rate=0.05, verbose=0, random_seed=M.SEED,
            auto_class_weights="Balanced", allow_writing_files=False)
SETS = {"H0": {}, "H1": {"depth": 4}, "H2": {"depth": 8},
        "H3": {"learning_rate": 0.025, "iterations": 1000, "l2_leaf_reg": 10}}
TARGETS = {"it": "IT", "tr": "TFR", "st": "ST", "tm": "TM"}


def params(name):
    return dict(BASE, **SETS[name])


def check_base_equals_main():
    """H0 must be exactly the chapter 3 model (thread count aside)."""
    main = M.make_model("CatBoost").get_params()
    mine = dict(params("H0"), thread_count=main.get("thread_count"))
    assert mine == main, (mine, main)
    return True


def fit_predict(name, Xtr, ytr, Xte, threads=THREADS):
    from catboost import CatBoostClassifier
    Xtr, Xte = M.impute_train_median(Xtr, Xte)
    m = CatBoostClassifier(**params(name), thread_count=threads)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else math.nan


def verdict(lo):
    return "transfers" if lo >= 0.70 else "weak" if lo >= 0.60 else "no"


def stored_main():
    """{target: [records]} of the stored main LORO CatBoost folds (r3_b40d)."""
    out = {t: [] for t in TARGETS.values()}
    with open(METRICS) as f:
        for line in f:
            r = json.loads(line)
            if (r["variant"] == "main" and r["protocol"] == "loro" and r["model"] == "CatBoost"
                    and r["clock"] == "D" and r["pop"] == "main" and r["features"] == "t0"
                    and r["label"] == r["target"] and r["target"] in out):
                out[r["target"]].append(r)
    return out


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                keys.add((r["target"], r["test"]))
            except json.JSONDecodeError:
                pass        # torn last line after a crash: refit that fold
    return keys


def git_head():
    try:
        return subprocess.check_output(["git", "-C", REV, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def log_run(run):
    with open(os.path.join(OUT, "runs.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(run, ensure_ascii=False) + "\n")


def run_set(name, threads):
    os.makedirs(os.path.join(OUT, "oof"), exist_ok=True)
    check_base_equals_main()
    t_start = time.time()
    tab = M.Table(TABLE)
    path = os.path.join(OUT, f"folds_{name}.jsonl")
    done = done_keys(path)
    stored = stored_main()
    n_fit = 0
    for k, tg in TARGETS.items():
        idx, y = tab.rows("main", "D", tg)
        X = tab.X(idx, M.FEATURE_SETS["t0"], "D")
        repo = tab.t["repo"].to_numpy()[idx]
        row_id = tab.t["row_id"].to_numpy()[idx]
        if name == "H0check":     # two smallest stored folds per target
            recs = sorted(stored[tg], key=lambda r: (r["n_test"], r["test"]))[:2]
            folds = [r["test"] for r in recs]
        else:
            folds = sorted(set(repo))
        for r in folds:
            if (tg, r) in done:
                continue
            tr, te = np.where(repo != r)[0], np.where(repo == r)[0]
            if len(te) == 0 or len(tr) == 0 or len(np.unique(y[tr])) < 2:
                continue
            t0 = time.time()
            prob = fit_predict("H0" if name == "H0check" else name, X[tr], y[tr], X[te], threads)
            rec = {"set": name, "target": tg, "test": r, "n_test": int(len(te)),
                   "pos_test": float(np.mean(y[te])), "n_train": int(len(tr)),
                   "auc_roc": auc(y[te], prob), "seconds": round(time.time() - t0, 1)}
            if name == "H0check":
                st = next(s for s in stored[tg] if s["test"] == r)
                z = np.load(os.path.join(RUN, "models", "oof", st["oof_file"]))
                assert (z["row_id"] == row_id[te]).all()
                rec["max_abs_diff_prob"] = float(np.max(np.abs(z["prob"].astype(float) - prob)))
                rec["stored_auc_roc"] = st["auc_roc"]
            fn = hashlib.sha1(f"{name}|{tg}|{r}".encode()).hexdigest()[:16] + ".npz"
            np.savez_compressed(os.path.join(OUT, "oof", fn), row_id=row_id[te].astype(np.int64),
                                prob=prob.astype(np.float32))
            rec["oof_file"] = fn
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            n_fit += 1
            print(f"{name} {tg} {r}: AUC {rec['auc_roc']:.4f} ({rec['seconds']} s)", flush=True)
    log_run({"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "kind": f"fit_{name}",
             "params": params("H0" if name == "H0check" else name), "threads": threads,
             "n_fit": n_fit, "git": git_head(), "seconds": round(time.time() - t_start, 1)})


def per_repo_auc_from(records, oof_dir, y_full):
    """{repo: AUC} from saved float32 predictions (as r3_report.per_repo_auc)."""
    out = {}
    for r in records:
        z = np.load(os.path.join(oof_dir, r["oof_file"]))
        out[r["test"]] = auc(y_full[z["row_id"]], z["prob"].astype(float))
    return out


def summarise():
    import pandas as pd
    t_start = time.time()
    T = pd.read_csv(TABLE, low_memory=False, usecols=["repo"] + [f"y_{t}__main_D" for t in TARGETS.values()])
    stored = stored_main()
    ref = json.load(open(R4_NUMBERS)) if os.path.exists(R4_NUMBERS) else {}
    nums, checks = {}, {"base_equals_main": check_base_equals_main()}
    # reproduction of H0
    rep = [json.loads(x) for x in open(os.path.join(OUT, "folds_H0check.jsonl"))]
    mx = max(r["max_abs_diff_prob"] for r in rep)
    checks["h0_reproduction"] = {"folds": [{k: r[k] for k in ("target", "test", "n_test", "auc_roc",
                                                                "stored_auc_roc", "max_abs_diff_prob")}
                                           for r in rep],
                                 "max_abs_diff_prob": mx, "ok": bool(mx <= 1e-6 and len(rep) == 8)}
    if not checks["h0_reproduction"]["ok"]:
        raise SystemExit(f"H0 reproduction failed: {checks['h0_reproduction']}")
    nums["hp.repro.max_abs_diff_prob"] = mx
    nums["hp.repro.n_folds"] = len(rep)
    alt = {}
    for name in ("H1", "H2", "H3"):
        p = os.path.join(OUT, f"folds_{name}.jsonl")
        alt[name] = [json.loads(x) for x in open(p)] if os.path.exists(p) else []
    for name in SETS:
        nums[f"hp.{name.lower()}.params"] = json.dumps(SETS[name] or {"main": True})
    checks["ref_e3"] = {}
    for k, tg in TARGETS.items():
        y_full = T[f"y_{tg}__main_D"].to_numpy()
        a0 = per_repo_auc_from(stored[tg], os.path.join(RUN, "models", "oof"), y_full)
        repos = sorted(a0)
        v0 = np.array([a0[r] for r in repos])
        m0, lo0, hi0 = S.cluster_boot_mean(v0, B, BOOT_SEED)
        v0_ok = v0[~np.isnan(v0)]
        nums.update({f"hp.h0.{k}.auc": m0, f"hp.h0.{k}.auc_lo": lo0, f"hp.h0.{k}.auc_hi": hi0,
                     f"hp.h0.{k}.verdict": verdict(lo0), f"hp.h0.{k}.n_repos": int(len(v0_ok))})
        r_auc = ref.get("pred", {}).get(k, {}).get("loro", {}).get("auc") if ref else None
        checks["ref_e3"][k] = {"auc_here": m0, "auc_r4_numbers": r_auc,
                               "ok": r_auc is None or abs(m0 - r_auc) < 1e-9}
        robust = True
        complete = True
        for name in ("H1", "H2", "H3"):
            recs = [r for r in alt[name] if r["target"] == tg]
            a = per_repo_auc_from(recs, os.path.join(OUT, "oof"), y_full)
            v = np.array([a.get(r, math.nan) for r in repos])
            n_done = len(recs)
            complete &= n_done == len(stored[tg])
            m, lo, hi = S.cluster_boot_mean(v, B, BOOT_SEED)
            d, dlo, dhi = S.paired_boot_diff(v, v0, B, BOOT_SEED)
            signif = bool((dlo > 0 or dhi < 0) and abs(d) >= SIGNIF)
            vd = verdict(lo)
            robust &= (vd == verdict(lo0)) and not signif
            key = f"hp.{name.lower()}.{k}"
            nums.update({f"{key}.auc": m, f"{key}.auc_lo": lo, f"{key}.auc_hi": hi, f"{key}.verdict": vd,
                         f"{key}.n_folds": n_done, f"{key}.n_repos": int(np.sum(~np.isnan(v))),
                         f"{key}.diff": d, f"{key}.diff_lo": dlo, f"{key}.diff_hi": dhi,
                         f"{key}.diff_signif": int(signif),
                         f"{key}.auc_repo_min": float(np.nanmin(v)) if n_done else None,
                         f"{key}.auc_repo_max": float(np.nanmax(v)) if n_done else None,
                         f"{key}.fit_seconds_mean": float(np.mean([r["seconds"] for r in recs])) if recs else None})
        nums[f"hp.{k}.complete"] = int(complete)
        nums[f"hp.{k}.robust"] = int(robust) if complete else None
    checks["ref_e3_ok"] = all(c["ok"] for c in checks["ref_e3"].values())
    run = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "kind": "summarise", "git": git_head(),
           "seconds": round(time.time() - t_start, 1), "boot": {"B": B, "seed": BOOT_SEED},
           "checks": {"h0_reproduction_ok": checks["h0_reproduction"]["ok"], "ref_e3_ok": checks["ref_e3_ok"]},
           "summary": {k: v for k, v in nums.items() if k.endswith((".auc", ".robust", ".diff"))}}
    log_run(run)
    checks["runs"] = [json.loads(x) for x in open(os.path.join(OUT, "runs.jsonl"), encoding="utf-8")]
    json.dump(checks, open(os.path.join(OUT, "checks.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump({k: nums[k] for k in sorted(nums)}, open(os.path.join(OUT, "numbers.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print(json.dumps(run, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["H0check", "H1", "H2", "H3"])
    ap.add_argument("--threads", type=int, default=THREADS)
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.set:
        run_set(args.set, args.threads)
    if args.summarise:
        summarise()
    return 0


if __name__ == "__main__":
    sys.exit(main())
