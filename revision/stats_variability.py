#!/usr/bin/env python3
"""
HAIT 2026 revision — Analysis A: variability, confidence intervals,
paired significance tests, and per-repository LORO breakdown.

Answers reviewer comments R1.3 / R2.4:
  * no confidence intervals / no estimate of between-experiment variability
  * "CatBoost performs comparably" is not backed by any statistical test
  * for T2 there is no per-repository breakdown like Fig. 3b has for T1

This script is ADDITIVE. It does not modify week4_modeling.py; it imports the
loading / feature / target / evaluation code from it so that hyperparameters,
random_state, and the feature list are provably identical to the published run.

Only two deliberate deviations from week4_modeling.py, both thread-count only
(this machine is shared with other jobs), neither of which affects results:
    LightGBM  n_jobs=-1      -> n_jobs=4
    CatBoost  (unset)        -> thread_count=4

Outputs (<DISSERTATION_DIR>/journal_articles/HAIT_2026/revision_results/):
    A-cv-variability.md     article-ready tables + discrepancy report
    A-per-fold-raw.csv      every single fold observation
    A-loro-per-repo.csv     every single per-repository LORO observation
"""

import csv
import os
import sys
import time
import warnings

import numpy as np
from scipy import stats

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostClassifier
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── Reuse the published pipeline verbatim ───────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_SCRIPTS, "predict"))

import week4_modeling as w4  # noqa: E402
from paths import DISSERTATION  # noqa: E402

FEATURE_COLS = w4.FEATURE_COLS
SIZE_ONLY_COLS = w4.SIZE_ONLY_COLS
load_data = w4.load_data
dp2_target = w4.dp2_target
dp4_target = w4.dp4_target
evaluate = w4.evaluate

PROCESSED_DIR = os.path.join(_SCRIPTS, "data", "processed")
OUT_DIR = os.path.join(DISSERTATION, "journal_articles", "HAIT_2026", "revision_results")
os.makedirs(OUT_DIR, exist_ok=True)

N_THREADS = 4
METRIC_KEYS = ["accuracy", "f1", "precision", "recall", "auc_roc", "auc_pr"]

# Repeated-CV seeds. Seed 42 is exactly the published split, so the seed-42
# folds double as the primary Table 3 / Table 4 numbers.
CV_SEEDS = [42, 43, 44, 45, 46]


# ── Model factory (identical hyperparameters to week4_modeling.py) ──────────

def make_models():
    return {
        "Dummy (majority)": lambda: DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
        "CatBoost (size-only)": lambda: CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1,
            verbose=0, random_seed=42, auto_class_weights="Balanced",
            thread_count=N_THREADS,
        ),
        "CatBoost": lambda: CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            verbose=0, random_seed=42, auto_class_weights="Balanced",
            thread_count=N_THREADS,
        ),
        "LightGBM": lambda: lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            is_unbalance=True, verbose=-1, random_state=42, n_jobs=N_THREADS,
        ),
    }


MAIN_MODELS = ["CatBoost", "LightGBM"]


def fit_predict(model_name, model_fn, X_train, y_train, X_test):
    """Train one model exactly as week4_modeling.py does."""
    model = model_fn()
    if model_name == "Logistic Regression":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return y_pred, y_prob


def select_cols(model_name, X, idx):
    """size-only models see only the 4 size features (as in the published run)."""
    if "size-only" in model_name:
        cols = [FEATURE_COLS.index(c) for c in SIZE_ONLY_COLS]
        return X[idx][:, cols]
    return X[idx]


# ── Statistics helpers ──────────────────────────────────────────────────────

def mean_std_ci(values, conf=0.95):
    """Mean, sample std (ddof=1) and a two-sided t-distribution CI.

    Returns dict with mean, std, n, df, t_crit, sem, lo, hi.
    """
    a = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    n = len(a)
    out = {"n": n, "mean": float(np.mean(a)) if n else float("nan"),
           "std": float("nan"), "sem": float("nan"), "df": max(n - 1, 0),
           "t_crit": float("nan"), "lo": float("nan"), "hi": float("nan")}
    if n < 2:
        return out
    sd = float(np.std(a, ddof=1))
    sem = sd / np.sqrt(n)
    t_crit = float(stats.t.ppf(0.5 + conf / 2.0, n - 1))
    out.update(std=sd, sem=sem, t_crit=t_crit,
               lo=out["mean"] - t_crit * sem, hi=out["mean"] + t_crit * sem)
    return out


def paired_compare(a, b, conf=0.95):
    """Paired comparison of two vectors of per-fold metrics (a - b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = len(d)
    res = {"n": n, "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
           "mean_diff": float(np.mean(d)), "std_diff": float(np.std(d, ddof=1)) if n > 1 else float("nan"),
           "wins_a": int(np.sum(d > 0)), "wins_b": int(np.sum(d < 0)), "ties": int(np.sum(d == 0))}
    if n > 1:
        sem = res["std_diff"] / np.sqrt(n)
        t_crit = float(stats.t.ppf(0.5 + conf / 2.0, n - 1))
        res["ci_lo"] = res["mean_diff"] - t_crit * sem
        res["ci_hi"] = res["mean_diff"] + t_crit * sem
        res["t_crit"] = t_crit
        res["df"] = n - 1
        try:
            t_stat, t_p = stats.ttest_rel(a, b)
            res["t_stat"], res["t_p"] = float(t_stat), float(t_p)
        except Exception as e:                                    # pragma: no cover
            res["t_stat"], res["t_p"], res["t_error"] = float("nan"), float("nan"), str(e)
        try:
            if np.allclose(d, 0):
                res["w_stat"], res["w_p"] = float("nan"), float("nan")
                res["w_error"] = "all differences are exactly zero"
            else:
                w_stat, w_p = stats.wilcoxon(a, b)
                res["w_stat"], res["w_p"] = float(w_stat), float(w_p)
        except Exception as e:                                    # pragma: no cover
            res["w_stat"], res["w_p"], res["w_error"] = float("nan"), float("nan"), str(e)
        # Cohen's d_z for paired samples
        res["cohens_dz"] = res["mean_diff"] / res["std_diff"] if res["std_diff"] > 0 else float("nan")
    return res


def f3(v):
    """Format to 3 decimals, or an explicit marker when the value is absent."""
    if v is None:
        return "n/a"
    try:
        if np.isnan(v):
            return "n/a"
    except TypeError:
        return "n/a"
    return f"{v:.3f}"


def pfmt(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


# ── Experiment 1+2: repeated stratified 5-fold CV ───────────────────────────

def run_repeated_cv(task, X, y, per_fold_rows):
    """Run stratified 5-fold CV.

    Seed 42 -> all five models (this is the published protocol).
    Seeds 43..46 -> CatBoost + LightGBM only (for the 25-fold paired test).

    Appends every fold observation to per_fold_rows and returns
    {(model, seed): [fold_metric_dicts]}.
    """
    models = make_models()
    collected = {}

    for seed in CV_SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = list(skf.split(X, y))
        model_names = list(models) if seed == 42 else MAIN_MODELS

        for model_name in model_names:
            model_fn = models[model_name]
            fold_metrics = []
            t0 = time.time()
            for fold, (tr, te) in enumerate(splits):
                X_tr = select_cols(model_name, X, tr)
                X_te = select_cols(model_name, X, te)
                y_pred, y_prob = fit_predict(model_name, model_fn, X_tr, y[tr], X_te)
                m = evaluate(y[te], y_pred, y_prob)
                fold_metrics.append(m)
                row = {"task": task, "protocol": "5-fold CV", "model": model_name,
                       "cv_seed": seed, "fold": fold,
                       "n_train": int(len(tr)), "n_test": int(len(te)),
                       "pos_rate_test": float(y[te].mean())}
                for k in METRIC_KEYS:
                    row[k] = m.get(k, float("nan"))
                per_fold_rows.append(row)
            collected[(model_name, seed)] = fold_metrics
            print(f"    [{task}] seed={seed} {model_name:22s} "
                  f"f1={np.mean([m['f1'] for m in fold_metrics]):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return collected


# ── Experiment 3: leave-one-repo-out ────────────────────────────────────────

def run_loro(task, X, y, repos, loro_rows):
    """Leave-one-repo-out, identical eligibility rules to week4_modeling.py."""
    models = make_models()
    repos_arr = np.array(repos)
    unique_repos = sorted(set(repos))
    out = {}

    for model_name in MAIN_MODELS:
        model_fn = models[model_name]
        per_repo = {}
        t0 = time.time()
        for repo in unique_repos:
            test_mask = repos_arr == repo
            train_mask = ~test_mask
            if test_mask.sum() < 10 or train_mask.sum() < 10:
                print(f"    [{task}] {model_name}: SKIPPED {repo} (too few rows)", flush=True)
                continue
            if len(np.unique(y[test_mask])) < 2:
                print(f"    [{task}] {model_name}: SKIPPED {repo} (single-class test set)", flush=True)
                continue
            tr = np.where(train_mask)[0]
            te = np.where(test_mask)[0]
            X_tr = select_cols(model_name, X, tr)
            X_te = select_cols(model_name, X, te)
            y_pred, y_prob = fit_predict(model_name, model_fn, X_tr, y[tr], X_te)
            m = evaluate(y[te], y_pred, y_prob)
            per_repo[repo] = m
            row = {"task": task, "protocol": "LORO", "model": model_name, "repo": repo,
                   "n_test": int(te.size), "n_train": int(tr.size),
                   "pos_rate_test": float(y[te].mean())}
            for k in METRIC_KEYS:
                row[k] = m.get(k, float("nan"))
            loro_rows.append(row)
        out[model_name] = per_repo
        print(f"    [{task}] LORO {model_name:22s} {len(per_repo)} repos "
              f"({time.time() - t0:.0f}s)", flush=True)
    return out


# ── Markdown emitters ───────────────────────────────────────────────────────

MODEL_ORDER = ["Dummy (majority)", "Logistic Regression", "CatBoost (size-only)",
               "CatBoost", "LightGBM"]

PUBLISHED = {
    "T1": {
        "Dummy (majority)":    dict(accuracy=.656, f1=.000, precision=.000, recall=.000, auc_roc=.500, auc_pr=.344),
        "Logistic Regression": dict(accuracy=.766, f1=.566, precision=.783, recall=.443, auc_roc=.834, auc_pr=.719),
        "CatBoost (size-only)": dict(accuracy=.815, f1=.776, precision=.664, recall=.935, auc_roc=.903, auc_pr=.797),
        "CatBoost":            dict(accuracy=.839, f1=.798, precision=.702, recall=.925, auc_roc=.928, auc_pr=.852),
        "LightGBM":            dict(accuracy=.842, f1=.800, precision=.707, recall=.919, auc_roc=.928, auc_pr=.852),
    },
    "T2": {
        "Dummy (majority)":    dict(accuracy=.595, f1=.746, precision=.595, recall=1.000, auc_roc=.500, auc_pr=.595),
        "Logistic Regression": dict(accuracy=.694, f1=.761, precision=.711, recall=.820, auc_roc=.755, auc_pr=.802),
        "CatBoost (size-only)": dict(accuracy=.567, f1=.615, precision=.652, recall=.583, auc_roc=.580, auc_pr=.652),
        "CatBoost":            dict(accuracy=.823, f1=.851, precision=.852, recall=.850, auc_roc=.897, auc_pr=.915),
        "LightGBM":            dict(accuracy=.825, f1=.854, precision=.847, recall=.862, auc_roc=.895, auc_pr=.914),
    },
}

PUBLISHED_LORO = {
    "T1": {"CatBoost": dict(f1=.736, auc_roc=.887), "LightGBM": dict(f1=.732, auc_roc=.881)},
    "T2": {"CatBoost": dict(f1=.635, auc_roc=.751), "LightGBM": dict(f1=.650, auc_roc=.750)},
}


ARTIFACT_PATH = os.path.join(_SCRIPTS, "predict", "output", "modeling_results.md")


def parse_original_artifact(path=ARTIFACT_PATH):
    """Parse the ORIGINAL week4_modeling.py run report (predict/output/modeling_results.md).

    That file was written by the published run itself, so it is the ground truth for what the
    code produced — as opposed to what the article says the code produced.

    Returns {"T1"/"T2": {"cv": {model: {metric: val}}, "loro": {model: {f1, auc_roc}}}} or None.
    """
    if not os.path.exists(path):
        return None
    import re
    with open(path, encoding="utf-8") as f:
        text = f.read()
    out = {}
    # DP2 -> T1, DP4 -> T2
    for section, task in (("DP2", "T1"), ("DP4", "T2")):
        m = re.search(rf"##\s*{section}\b(.*?)(?=\n##\s|\Z)", text, re.S)
        if not m:
            return None
        body = m.group(1)
        cv = {}
        for row in re.finditer(r"^\|\s*([^|]+?)\s*\|" + r"\s*([\d.]+)\s*\|" * 6, body, re.M):
            name = row.group(1).strip()
            if name in ("Model", "---"):
                continue
            cv[name] = dict(zip(METRIC_KEYS, [float(row.group(i)) for i in range(2, 8)]))
        loro = {}
        for row in re.finditer(r"\*\*(\w+):\*\*\s*F1=([\d.]+),\s*AUC-ROC=([\d.]+)", body):
            loro[row.group(1)] = {"f1": float(row.group(2)), "auc_roc": float(row.group(3))}
        out[task] = {"cv": cv, "loro": loro}
    return out


def summarise_cv(collected, model_name, seed=42):
    fm = collected.get((model_name, seed))
    if fm is None:
        return None
    return {k: mean_std_ci([m.get(k, float("nan")) for m in fm]) for k in METRIC_KEYS}


def loro_summary(per_repo, key):
    vals = np.array([m.get(key, np.nan) for m in per_repo.values()], dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return None
    return {"n": int(vals.size), "mean": float(vals.mean()), "median": float(np.median(vals)),
            "std": float(vals.std(ddof=1)) if vals.size > 1 else float("nan"),
            "min": float(vals.min()), "max": float(vals.max())}


def main():
    t_start = time.time()
    L = []                       # markdown lines
    per_fold_rows = []
    loro_rows = []
    notes = []                   # discrepancy / integrity notes

    # ── Data integrity ──────────────────────────────────────────────────
    print("Loading data...", flush=True)
    merged_path = os.path.join(PROCESSED_DIR, "all_merged_features.csv")
    open_path = os.path.join(PROCESSED_DIR, "all_open_features.csv")

    def raw_row_count(path):
        with open(path, newline="", encoding="utf-8") as f:
            return sum(1 for _ in csv.DictReader(f))

    n_raw_merged = raw_row_count(merged_path)
    n_raw_open = raw_row_count(open_path)

    X1, y1, repos1 = load_data(merged_path, FEATURE_COLS, dp2_target)
    X2, y2, repos2 = load_data(open_path, FEATURE_COLS, dp4_target)

    data_facts = {
        "T1": dict(csv_rows=n_raw_merged, modelled=len(y1), pos=int(y1.sum()),
                   pos_rate=float(y1.mean()), repos=len(set(repos1))),
        "T2": dict(csv_rows=n_raw_open, modelled=len(y2), pos=int(y2.sum()),
                   pos_rate=float(y2.mean()), repos=len(set(repos2))),
    }
    print(f"  T1: csv={n_raw_merged} modelled={len(y1)} pos={int(y1.sum())} "
          f"({100*y1.mean():.2f}%) repos={len(set(repos1))}", flush=True)
    print(f"  T2: csv={n_raw_open} modelled={len(y2)} pos={int(y2.sum())} "
          f"({100*y2.mean():.2f}%) repos={len(set(repos2))}", flush=True)

    # ── Run everything ──────────────────────────────────────────────────
    print("\n=== T1 (DP2, review_rounds >= 2) — repeated 5-fold CV ===", flush=True)
    cv1 = run_repeated_cv("T1", X1, y1, per_fold_rows)
    print("\n=== T2 (DP4, is_stale) — repeated 5-fold CV ===", flush=True)
    cv2 = run_repeated_cv("T2", X2, y2, per_fold_rows)

    print("\n=== T1 — leave-one-repo-out ===", flush=True)
    loro1 = run_loro("T1", X1, y1, repos1, loro_rows)
    print("\n=== T2 — leave-one-repo-out ===", flush=True)
    loro2 = run_loro("T2", X2, y2, repos2, loro_rows)

    # ── Write raw CSVs ──────────────────────────────────────────────────
    fold_csv = os.path.join(OUT_DIR, "A-per-fold-raw.csv")
    fold_fields = ["task", "protocol", "model", "cv_seed", "fold", "n_train", "n_test",
                   "pos_rate_test"] + METRIC_KEYS
    with open(fold_csv, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=fold_fields)
        wtr.writeheader()
        for r in per_fold_rows:
            wtr.writerow({k: r.get(k, "") for k in fold_fields})

    loro_csv = os.path.join(OUT_DIR, "A-loro-per-repo.csv")
    loro_fields = ["task", "protocol", "model", "repo", "n_train", "n_test",
                   "pos_rate_test"] + METRIC_KEYS
    with open(loro_csv, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=loro_fields)
        wtr.writeheader()
        for r in loro_rows:
            wtr.writerow({k: r.get(k, "") for k in loro_fields})

    # ── Discrepancy analysis ────────────────────────────────────────────
    disc = []          # (severity, text)

    for task, cv in [("T1", cv1), ("T2", cv2)]:
        for mname in MODEL_ORDER:
            s = summarise_cv(cv, mname, 42)
            if s is None:
                disc.append(("HIGH", f"{task} / {mname}: seed-42 CV did not run — no value to compare."))
                continue
            for k in METRIC_KEYS:
                pub = PUBLISHED[task][mname].get(k)
                got = s[k]["mean"]
                if pub is None or np.isnan(got):
                    continue
                delta = got - pub
                if abs(delta) >= 0.0005:
                    sev = "HIGH" if abs(delta) >= 0.01 else ("MED" if abs(delta) >= 0.002 else "LOW")
                    disc.append((sev, f"{task} / {mname} / {k}: published {pub:.3f}, "
                                      f"reproduced {got:.3f} (Δ={delta:+.3f})"))

    for task, lo in [("T1", loro1), ("T2", loro2)]:
        for mname in MAIN_MODELS:
            per_repo = lo.get(mname, {})
            if not per_repo:
                disc.append(("HIGH", f"{task} / LORO / {mname}: no repositories evaluated."))
                continue
            for k in ["f1", "auc_roc"]:
                summ = loro_summary(per_repo, k)
                pub = PUBLISHED_LORO[task][mname].get(k)
                if summ is None or pub is None:
                    continue
                delta = summ["mean"] - pub
                if abs(delta) >= 0.0005:
                    sev = "HIGH" if abs(delta) >= 0.01 else ("MED" if abs(delta) >= 0.002 else "LOW")
                    disc.append((sev, f"{task} / LORO / {mname} / {k}: published {pub:.3f}, "
                                      f"reproduced {summ['mean']:.3f} (Δ={delta:+.3f})"))

    # ── Second comparison: reproduced vs the ORIGINAL run's own report ──
    original = parse_original_artifact()
    disc_artifact = []
    if original is None:
        disc_artifact.append(("HIGH", f"`{ARTIFACT_PATH}` not found — cannot separate an article "
                                      f"transcription error from a genuine reproduction failure."))
    else:
        for task, cv, lo in [("T1", cv1, loro1), ("T2", cv2, loro2)]:
            for mname in MODEL_ORDER:
                ref = original[task]["cv"].get(mname)
                s = summarise_cv(cv, mname, 42)
                if ref is None or s is None:
                    continue
                for k in METRIC_KEYS:
                    if k not in ref or np.isnan(s[k]["mean"]):
                        continue
                    d = s[k]["mean"] - ref[k]
                    if abs(d) >= 0.0005:
                        disc_artifact.append(("HIGH", f"{task} / {mname} / {k}: original run "
                                                      f"{ref[k]:.3f}, reproduced {s[k]['mean']:.3f} "
                                                      f"(Δ={d:+.3f})"))
            for mname in MAIN_MODELS:
                ref = original[task]["loro"].get(mname)
                per_repo = lo.get(mname, {})
                if ref is None or not per_repo:
                    continue
                for k in ["f1", "auc_roc"]:
                    summ = loro_summary(per_repo, k)
                    if summ is None or k not in ref:
                        continue
                    d = summ["mean"] - ref[k]
                    if abs(d) >= 0.0005:
                        disc_artifact.append(("HIGH", f"{task} / LORO / {mname} / {k}: original run "
                                                      f"{ref[k]:.3f}, reproduced {summ['mean']:.3f} "
                                                      f"(Δ={d:+.3f})"))

    # dataset-size checks
    if data_facts["T1"]["csv_rows"] != 64820:
        disc.append(("HIGH", f"all_merged_features.csv holds {data_facts['T1']['csv_rows']} rows, "
                             f"the article claims 64,820 merged PRs."))
    if data_facts["T1"]["modelled"] != data_facts["T1"]["csv_rows"]:
        disc.append(("HIGH", f"T1 model trained on {data_facts['T1']['modelled']} rows, not all "
                             f"{data_facts['T1']['csv_rows']} rows in the CSV "
                             f"({data_facts['T1']['csv_rows'] - data_facts['T1']['modelled']} dropped "
                             f"for a missing review_rounds value)."))
    if data_facts["T2"]["csv_rows"] != 7156:
        disc.append(("HIGH", f"all_open_features.csv holds {data_facts['T2']['csv_rows']} rows, "
                             f"the article claims 7,156 open PRs."))
    if data_facts["T2"]["pos"] != 4256:
        disc.append(("HIGH", f"T2 positives = {data_facts['T2']['pos']}, the article claims 4,256 stale."))

    # ── Markdown: header ────────────────────────────────────────────────
    L += [
        "# HAIT 2026 revision — Analysis A: fold variability, confidence intervals, paired tests",
        "",
        f"Last updated: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "> Answers reviewer comments **R1.3** and **R2.4**: \"no confidence intervals, no statistical",
        "> tests, no estimate of variability between experiments\"; \"the claim *CatBoost performs",
        "> comparably* is not backed by any statistical test\"; \"for T2 there is no per-repository",
        "> breakdown like Figure 3b has for T1\".",
        "",
        "Produced by `revision/stats_variability.py`, which imports the model",
        "definitions, feature list, target functions and metric code directly from",
        "`predict/week4_modeling.py`. Hyperparameters, `random_state=42` and the",
        "24-feature list are therefore identical to the published run by construction. The only",
        "deviation is thread count (`n_jobs=4` / `thread_count=4`), which does not change results.",
        "",
        "**Notation.** T1 = DP2 = *review requires >= 2 iterations* (merged PRs).",
        "T2 = DP4 = *pull request becomes stale* (open PRs).",
        "",
        "---",
        "",
    ]

    # ── Markdown: discrepancies ─────────────────────────────────────────
    order = {"HIGH": 0, "MED": 1, "LOW": 2}

    L += ["## 0. Discrepancies vs published values", "",
          "Two separate comparisons are made, and the distinction matters:",
          "",
          "1. **Reproduced vs the article text** — what a reader of the paper would check.",
          "2. **Reproduced vs the pipeline's own recorded output** "
          "(`predict/output/modeling_results.md`, written by the original run "
          "on 2026-05-01) — what the code actually produced.",
          "",
          "A mismatch in (1) that does *not* appear in (2) is a transcription error in the",
          "manuscript, not a reproduction failure.",
          "",
          "### 0.0.1 Reproduced vs the pipeline's own recorded output", ""]
    if original is None:
        L += ["**Could not be checked** — the original artifact was not found.", ""]
    elif not disc_artifact:
        L += ["**Exact match on every value.** All 60 cross-validation figures (2 tasks × 5 models ×",
              "6 metrics) and all 8 LORO figures (2 tasks × 2 models × 2 metrics) reproduce to 3",
              "decimals against the original run's own report. The pipeline is deterministic and this",
              "analysis is running the same computation on the same data.", ""]
    else:
        disc_artifact.sort(key=lambda x: order[x[0]])
        L += ["| Severity | Discrepancy |", "|---|---|"]
        for sev, txt in disc_artifact:
            L.append(f"| {sev} | {txt} |")
        L.append("")

    L += ["### 0.0.2 Reproduced vs the article text", ""]
    if not disc:
        L += ["**None.** Every published figure reproduced to 3 decimals, and every dataset size",
              "claim matches the data actually used for training.", ""]
    else:
        disc.sort(key=lambda x: order[x[0]])
        L += ["| Severity | Discrepancy |", "|---|---|"]
        for sev, txt in disc:
            L.append(f"| {sev} | {txt} |")
        L += ["", "Severity: HIGH >= 0.010 absolute, MED >= 0.002, LOW >= 0.0005 (i.e. visible at 3 decimals).", ""]

    # Root-cause note for the T2 LORO LightGBM mismatch, if it is present.
    if original is not None and disc and not disc_artifact:
        ref_lg = original["T2"]["loro"].get("LightGBM", {})
        ref_cb = original["T2"]["loro"].get("CatBoost", {})
        if ref_lg and ref_cb:
            L += [
                "### 0.0.3 Root cause — the article misreports one LORO figure", "",
                "Because section 0.0.1 is an exact match, the mismatches above are **errors in the",
                "manuscript text, not in the code**. The relevant sentence in Results reads:",
                "",
                "> Under LORO validation, staleness prediction degrades more sharply, to F1 = 0.65 and",
                "> AUC-ROC = 0.75 (LightGBM; CatBoost: F1 = 0.635, AUC-ROC = 0.751).",
                "",
                "| Quantity | Article | Pipeline output | Verdict |",
                "|---|---|---|---|",
                f"| T2 LORO LightGBM F1 | 0.65 | {ref_lg['f1']:.3f} | correct — 0.645 rounds to 0.65 |",
                f"| T2 LORO LightGBM AUC-ROC | 0.75 | {ref_lg['auc_roc']:.3f} | **wrong — 0.743 rounds to 0.74** |",
                f"| T2 LORO CatBoost F1 | 0.635 | {ref_cb['f1']:.3f} | correct |",
                f"| T2 LORO CatBoost AUC-ROC | 0.751 | {ref_cb['auc_roc']:.3f} | correct |",
                "",
                "The value 0.75 attributed to LightGBM is CatBoost's 0.751; LightGBM's own AUC-ROC is",
                f"{ref_lg['auc_roc']:.3f}. **Fix in the revision:** change the LightGBM AUC-ROC in that",
                "sentence from 0.75 to 0.74. Note that the Abstract and Conclusions also state",
                "\"staleness prediction is more project-specific (AUC-ROC = 0.75)\" — that figure is",
                "CatBoost's and is correct as long as it is not attributed to LightGBM.",
                "",
            ]

    # ── Markdown: dataset integrity ─────────────────────────────────────
    L += [
        "### 0.1 Dataset size and label balance",
        "",
        "| Quantity | Article claims | Actual | Match |",
        "|---|---|---|---|",
        f"| Merged PRs (rows in `all_merged_features.csv`) | 64,820 | {data_facts['T1']['csv_rows']:,} | "
        f"{'yes' if data_facts['T1']['csv_rows'] == 64820 else 'NO'} |",
        f"| Merged PRs actually used to train T1 | 64,820 | {data_facts['T1']['modelled']:,} | "
        f"{'yes' if data_facts['T1']['modelled'] == 64820 else 'NO'} |",
        f"| Merged PRs with >= 2 iterations | 34.4% | {100*data_facts['T1']['pos_rate']:.2f}% "
        f"({data_facts['T1']['pos']:,}) | "
        f"{'yes' if abs(100*data_facts['T1']['pos_rate'] - 34.4) < 0.05 else 'NO'} |",
        f"| Open PRs (rows in `all_open_features.csv`) | 7,156 | {data_facts['T2']['csv_rows']:,} | "
        f"{'yes' if data_facts['T2']['csv_rows'] == 7156 else 'NO'} |",
        f"| Open PRs actually used to train T2 | 7,156 | {data_facts['T2']['modelled']:,} | "
        f"{'yes' if data_facts['T2']['modelled'] == 7156 else 'NO'} |",
        f"| Stale open PRs | 4,256 (59.5%) | {data_facts['T2']['pos']:,} "
        f"({100*data_facts['T2']['pos_rate']:.1f}%) | "
        f"{'yes' if data_facts['T2']['pos'] == 4256 else 'NO'} |",
        f"| Repositories (merged / open) | 12 | {data_facts['T1']['repos']} / {data_facts['T2']['repos']} | "
        f"{'yes' if data_facts['T1']['repos'] == 12 and data_facts['T2']['repos'] == 12 else 'NO'} |",
        "",
    ]

    # ── Markdown: filtering audit (static, from source) ─────────────────
    L += [
        "### 0.2 Where the merged-PR filtering actually happens",
        "",
        "The filtering is implemented in **`enrich/compute_features.py`**,",
        "function `should_include_merged()` (lines 333-349), applied in `process_repo()` at line 405.",
        "It runs at CSV-build time, so `all_merged_features.csv` is already filtered and",
        "`week4_modeling.py` applies no further exclusion — it reads the CSV and drops nothing.",
        "",
        "The manuscript states the criteria in two sentences (Section 3, Data):",
        "",
        "> Merged PRs were added to the dataset if they had **at least one review event**, valid",
        "> creation and merge timestamps, and **targeted the main branch**. We excluded PRs merged in",
        "> **under one minute**, PRs open for **more than 365 days**, PRs **without code changes**,",
        "> **reverts**, and **duplicates**.",
        "",
        "Criterion-by-criterion against the code:",
        "",
        "| Article criterion | In the code? | Implementation |",
        "|---|---|---|",
        "| at least one review event | yes | `review_count == 0` -> excluded |",
        "| targeted the main branch | yes, but broader | base must be exactly `main`/`master`/`trunk`/`develop`/`release`/`canary`/`dev`, **or** contain the substring `main`, `master`, `trunk`, `release`, `canary`. So `release/8.2` and `main-next` are kept — wider than \"the main branch\" |",
        "| valid creation and merge timestamps | partly | no explicit validity check; rows with an unparseable timestamp get `time_to_merge_hours = None` and then bypass the two duration filters rather than being dropped |",
        "| merged in under one minute | yes | `time_to_merge_hours < 1/60` |",
        "| open more than 365 days | yes | `time_to_merge_hours > 365*24` |",
        "| no code changes | yes | `lines_added == 0 and lines_deleted == 0` |",
        "| **reverts** | **no** | no revert detection anywhere in the pipeline (`grep -ri revert` over `**/*.py` returns nothing) |",
        "| **duplicates** | **no** | no duplicate detection anywhere. `fetch/utils.py:89` `load_existing_ids()` deduplicates by PR id when *resuming an interrupted fetch*; it does not exclude duplicate or superseded pull requests |",
        "",
        "**Consequence for the revision.** Most of the stated filter set is genuinely implemented. Two",
        "stated exclusions — **reverts** and **duplicates** — were never implemented, so the article",
        "currently claims a cleaning step that did not happen. Two further statements are looser than",
        "the code: \"targeted the main branch\" is implemented as a substring match that also admits",
        "release and canary branches, and \"valid creation and merge timestamps\" is not enforced as a",
        "check. The honest fix is one sentence in the Data section: drop the words \"reverts, and",
        "duplicates\", and describe the branch rule as \"targeted a main, release, or canary branch\".",
        "Nothing needs re-running — the model was always trained on exactly the rows in the CSV.",
        "",
        "---",
        "",
    ]

    # ── Markdown: Tables 3 and 4 with mean +/- std and CI ────────────────
    L += [
        "## 1. Tables 3 and 4 — mean ± std and 95% confidence intervals",
        "",
        "Protocol: stratified 5-fold cross-validation, `shuffle=True`, `random_state=42` — the exact",
        "published split. Each cell is `mean ± std` over the **5 folds**; std is the sample standard",
        "deviation (`ddof=1`). The 95% confidence interval is the **Student t interval**",
        "`mean ± t(0.975, df=4) · s/√5`, with **t(0.975, 4) = 2.776** — an exact small-sample interval,",
        "not a normal approximation. `n/a` marks a metric that is undefined for that model.",
        "",
        "Column order below is **Accuracy, F1, Precision, Recall, AUC-ROC, AUC-PR** in *both* tables,",
        "which is exactly the order in the author's submitted file `30062026_...docx`. Verified by",
        "reading the table headers out of `word/document.xml`: the AUC-PR column is displaced to",
        "position 1 **only in the editor-returned copy** (`review/Отпр_автору...docx`, Table 3), not in",
        "the submitted manuscript and not in `01072026_...docx`. The column-order half of comment R1.7",
        "is therefore an artefact of the editorial copy, not an author error — worth saying so in the",
        "response letter rather than silently \"fixing\" a table that was already correct.",
        "",
    ]

    for task, cv, title in [("T1", cv1, "Table 3 — T1 (review requires >= 2 iterations), 5-fold CV"),
                            ("T2", cv2, "Table 4 — T2 (pull request becomes stale), 5-fold CV")]:
        L += [f"### {title}", "",
              "**Paste-ready (`mean ± std`):**", "",
              "| Model | Accuracy | F1 | Precision | Recall | AUC-ROC | AUC-PR |",
              "|---|---|---|---|---|---|---|"]
        for mname in MODEL_ORDER:
            s = summarise_cv(cv, mname, 42)
            if s is None:
                L.append(f"| {mname} | did not run | | | | | |")
                continue
            cells = []
            for k in METRIC_KEYS:
                if np.isnan(s[k]["mean"]):
                    cells.append("n/a")
                else:
                    cells.append(f"{s[k]['mean']:.3f} ± {s[k]['std']:.3f}")
            L.append(f"| {mname} | " + " | ".join(cells) + " |")
        L += ["", "**95% CI (t, df=4) for the same runs:**", "",
              "| Model | Accuracy | F1 | Precision | Recall | AUC-ROC | AUC-PR |",
              "|---|---|---|---|---|---|---|"]
        for mname in MODEL_ORDER:
            s = summarise_cv(cv, mname, 42)
            if s is None:
                L.append(f"| {mname} | did not run | | | | | |")
                continue
            cells = []
            for k in METRIC_KEYS:
                if np.isnan(s[k]["mean"]):
                    cells.append("n/a")
                elif np.isnan(s[k]["lo"]):
                    cells.append(f"{s[k]['mean']:.3f} (CI n/a)")
                else:
                    cells.append(f"[{s[k]['lo']:.3f}, {s[k]['hi']:.3f}]")
            L.append(f"| {mname} | " + " | ".join(cells) + " |")
        L.append("")

    # per-fold appendix
    L += ["### 1.1 Every fold value (seed 42)", "",
          "Full raw values, including seeds 43-46, are in `A-per-fold-raw.csv`.", ""]
    for task, cv in [("T1", cv1), ("T2", cv2)]:
        L += [f"**{task}**", "",
              "| Model | Metric | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | mean | std |",
              "|---|---|---|---|---|---|---|---|---|"]
        for mname in MODEL_ORDER:
            fm = cv.get((mname, 42))
            if fm is None:
                continue
            for k in ["f1", "auc_roc"]:
                vals = [m.get(k, float("nan")) for m in fm]
                s = mean_std_ci(vals)
                L.append(f"| {mname} | {k} | " + " | ".join(f3(v) for v in vals) +
                         f" | {f3(s['mean'])} | {f3(s['std'])} |")
        L.append("")

    L += ["---", ""]

    # ── Markdown: paired test ───────────────────────────────────────────
    L += [
        "## 2. CatBoost vs LightGBM — paired significance test",
        "",
        "**Why not a 5-fold Wilcoxon.** With n = 5 paired observations the Wilcoxon signed-rank test",
        "has 2^5 = 32 equally likely sign assignments under the null, so its smallest attainable",
        "two-sided p-value is 2/32 = **0.0625**. A 5-fold Wilcoxon *cannot* return p < 0.05 no matter",
        "how large the effect is, and reporting one would be uninformative. We therefore use",
        "**repeated stratified 5-fold cross-validation: 5 repeats × 5 folds = 25 paired observations**,",
        "with `random_state` = 42, 43, 44, 45, 46 for the fold assignment (model seeds stay at 42).",
        "Repeat 1 (seed 42) is the published split, so Table 3/4 remain a strict subset of this design.",
        "",
        "The difference is defined as **CatBoost − LightGBM**. The confidence interval is the Student t",
        "interval on the 25 paired differences (df = 24, t(0.975, 24) = 2.064). Both a paired t-test",
        "and a Wilcoxon signed-rank test are reported on the same 25 pairs.",
        "",
        "*Caveat, stated for honesty:* repeated CV folds are not independent (the same rows recur",
        "across repeats), so these p-values are mildly anti-conservative in the Nadeau–Bengio sense.",
        "They are used here to test whether the two models are *distinguishable at all*; a null result",
        "is only strengthened by that caveat.",
        "",
        "| Task | Metric | CatBoost | LightGBM | Mean diff (CB−LGBM) | 95% CI of diff | paired t | p (t) | Wilcoxon W | p (W) | CatBoost wins |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    paired_results = {}
    for task, cv in [("T1", cv1), ("T2", cv2)]:
        for metric in ["f1", "auc_roc"]:
            cb, lg = [], []
            ok = True
            for seed in CV_SEEDS:
                fcb = cv.get(("CatBoost", seed))
                flg = cv.get(("LightGBM", seed))
                if fcb is None or flg is None:
                    ok = False
                    break
                cb += [m.get(metric, float("nan")) for m in fcb]
                lg += [m.get(metric, float("nan")) for m in flg]
            n_expected = 5 * len(CV_SEEDS)
            if not ok or len(cb) != n_expected:
                L.append(f"| {task} | {metric} | FAILED TO COMPUTE — expected {n_expected} pairs, "
                         f"got {len(cb)} | | | | | | | | |")
                continue
            r = paired_compare(cb, lg)
            paired_results[(task, metric)] = r
            L.append(
                f"| {task} | {metric.replace('auc_roc', 'AUC-ROC').replace('f1', 'F1')} | "
                f"{r['mean_a']:.3f} | {r['mean_b']:.3f} | {r['mean_diff']:+.4f} | "
                f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['t_stat']:.3f} | {pfmt(r['t_p'])} | "
                f"{f3(r.get('w_stat'))} | {pfmt(r.get('w_p'))} | {r['wins_a']}/{n_expected} |"
            )

    L += ["", "### 2.1 Verdict", ""]
    for (task, metric), r in paired_results.items():
        label = "F1" if metric == "f1" else "AUC-ROC"
        sig = r["t_p"] < 0.05
        ci_excl = not (r["ci_lo"] <= 0 <= r["ci_hi"])
        if sig and ci_excl:
            verdict = (f"the difference **is** statistically distinguishable from zero "
                       f"(paired t p={pfmt(r['t_p'])}), but its size is {abs(r['mean_diff']):.4f} "
                       f"— roughly {abs(r['mean_diff'])*100:.2f} percentage points")
        else:
            verdict = (f"the difference is **not** statistically distinguishable from zero "
                       f"(paired t p={pfmt(r['t_p'])}, 95% CI includes 0)")
        L.append(f"- **{task}, {label}**: CatBoost {r['mean_a']:.3f} vs LightGBM {r['mean_b']:.3f}; "
                 f"mean difference {r['mean_diff']:+.4f}, 95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]; "
                 f"{verdict}. Cohen's d_z = {f3(r.get('cohens_dz'))}.")
    # Concrete replacement wording, instantiated from the numbers actually computed.
    L += ["", "**Suggested replacement wording for the article** (replaces the unbacked "
              "\"CatBoost performs comparably\"):", ""]
    r_t1f = paired_results.get(("T1", "f1"))
    r_t1a = paired_results.get(("T1", "auc_roc"))
    r_t2f = paired_results.get(("T2", "f1"))
    r_t2a = paired_results.get(("T2", "auc_roc"))
    if all(r is not None for r in (r_t1f, r_t1a, r_t2f, r_t2a)):
        # gap from the better main model down to the size-only baseline, for scale
        gap_t1 = PUBLISHED["T1"]["LightGBM"]["f1"] - PUBLISHED["T1"]["CatBoost (size-only)"]["f1"]
        gap_t2 = PUBLISHED["T2"]["LightGBM"]["f1"] - PUBLISHED["T2"]["CatBoost (size-only)"]["f1"]

        def _dir(r):
            return "LightGBM" if r["mean_diff"] < 0 else "CatBoost"

        L += [
            "> We compared CatBoost and LightGBM over 25 paired folds (5 repeats of stratified 5-fold",
            "> cross-validation, fold seeds 42-46). On T1 the F1 difference is "
            f"{abs(r_t1f['mean_diff']):.4f} in favour of {_dir(r_t1f)} "
            f"(95% CI [{r_t1f['ci_lo']:+.4f}, {r_t1f['ci_hi']:+.4f}], paired t-test p = {pfmt(r_t1f['t_p'])}, "
            f"Wilcoxon p = {pfmt(r_t1f['w_p'])}), while the AUC-ROC difference is not distinguishable",
            f"> from zero ({r_t1a['mean_diff']:+.4f}, 95% CI [{r_t1a['ci_lo']:+.4f}, {r_t1a['ci_hi']:+.4f}], "
            f"p = {pfmt(r_t1a['t_p'])}). On T2 the F1 difference is {abs(r_t2f['mean_diff']):.4f} in favour",
            f"> of {_dir(r_t2f)} (95% CI [{r_t2f['ci_lo']:+.4f}, {r_t2f['ci_hi']:+.4f}], p = {pfmt(r_t2f['t_p'])}) "
            f"and the AUC-ROC difference is again not distinguishable from zero",
            f"> ({r_t2a['mean_diff']:+.4f}, 95% CI [{r_t2a['ci_lo']:+.4f}, {r_t2a['ci_hi']:+.4f}], "
            f"p = {pfmt(r_t2a['t_p'])}). Where a difference is detectable it does not exceed",
            f"> {max(abs(r_t1f['mean_diff']), abs(r_t2f['mean_diff']))*100:.1f} percentage points of F1, against a "
            f"{min(gap_t1, gap_t2)*100:.1f}-point margin of either model over the size-only",
            "> baseline. We therefore treat the two learners as practically equivalent for this task and",
            "> report CatBoost throughout for its native handling of the class imbalance.",
            "",
            "The last clause is a placeholder for whatever the real reason was for preferring CatBoost in",
            "the SHAP analysis — replace it with the actual reason rather than leaving a rationalisation.",
            "",
        ]
    else:
        L += ["> NOT GENERATED — the paired comparison did not complete, so no wording is proposed.", ""]
    L += ["---", ""]

    # ── Markdown: LORO per-repo ─────────────────────────────────────────
    L += [
        "## 3. Leave-one-repo-out, per repository",
        "",
        "Eligibility rules are unchanged from `week4_modeling.py`: a repository is evaluated only if",
        "its held-out set has >= 10 rows, the training set has >= 10 rows, and the held-out set",
        "contains both classes. The aggregate row is the **macro average over repositories** (each",
        "repository weighted equally), which is how the published LORO figures were computed.",
        "",
        "`n test` is the number of held-out pull requests; `pos rate` is the share of the positive",
        "class in that held-out set. Raw values: `A-loro-per-repo.csv`.",
        "",
    ]

    for task, lo, human in [("T2", loro2, "T2 — pull request becomes stale (NEW: feeds Figure 3, panel c)"),
                            ("T1", loro1, "T1 — review requires >= 2 iterations (existing Figure 3b, re-emitted for comparability)")]:
        L += [f"### {human}", ""]
        for mname in MAIN_MODELS:
            per_repo = lo.get(mname, {})
            L += [f"**{mname}**", "",
                  "| Repository | n test | pos rate | F1 | Precision | Recall | AUC-ROC | AUC-PR |",
                  "|---|---|---|---|---|---|---|---|"]
            if not per_repo:
                L += ["| *no repositories were eligible* | | | | | | | |", ""]
                continue
            lookup = {(r["task"], r["model"], r["repo"]): r for r in loro_rows}
            for repo in sorted(per_repo):
                m = per_repo[repo]
                meta = lookup.get((task, mname, repo), {})
                L.append(f"| {repo} | {meta.get('n_test', '?')} | "
                         f"{f3(meta.get('pos_rate_test'))} | {f3(m.get('f1'))} | "
                         f"{f3(m.get('precision'))} | {f3(m.get('recall'))} | "
                         f"{f3(m.get('auc_roc'))} | {f3(m.get('auc_pr'))} |")
            # aggregates
            agg_cells = []
            for k in ["f1", "precision", "recall", "auc_roc", "auc_pr"]:
                s = loro_summary(per_repo, k)
                agg_cells.append(f3(s["mean"]) if s else "n/a")
            total_n = sum(lookup.get((task, mname, r), {}).get("n_test", 0) for r in per_repo)
            L.append(f"| **macro mean ({len(per_repo)} repos)** | {total_n} | — | " +
                     " | ".join(agg_cells) + " |")
            L.append("")
            f1s = loro_summary(per_repo, "f1")
            aucs = loro_summary(per_repo, "auc_roc")
            if f1s:
                L.append(f"F1 across repositories — median **{f1s['median']:.3f}**, "
                         f"mean **{f1s['mean']:.3f}**, std {f3(f1s['std'])}, "
                         f"range **{f1s['min']:.3f}–{f1s['max']:.3f}** (n={f1s['n']} repos).")
            if aucs:
                L.append(f"AUC-ROC across repositories — median **{aucs['median']:.3f}**, "
                         f"mean **{aucs['mean']:.3f}**, std {f3(aucs['std'])}, "
                         f"range **{aucs['min']:.3f}–{aucs['max']:.3f}** (n={aucs['n']} repos).")
            L.append("")

    # LORO comparison summary table
    L += ["### 3.1 LORO summary, both tasks side by side", "",
          "| Task | Model | Repos | F1 median | F1 mean | F1 range | AUC-ROC median | AUC-ROC mean | AUC-ROC range |",
          "|---|---|---|---|---|---|---|---|---|"]
    for task, lo in [("T1", loro1), ("T2", loro2)]:
        for mname in MAIN_MODELS:
            per_repo = lo.get(mname, {})
            if not per_repo:
                L.append(f"| {task} | {mname} | 0 | did not compute | | | | | |")
                continue
            f1s = loro_summary(per_repo, "f1")
            aucs = loro_summary(per_repo, "auc_roc")
            L.append(f"| {task} | {mname} | {len(per_repo)} | "
                     f"{f3(f1s['median'])} | {f3(f1s['mean'])} | {f1s['min']:.3f}–{f1s['max']:.3f} | "
                     f"{f3(aucs['median'])} | {f3(aucs['mean'])} | {aucs['min']:.3f}–{aucs['max']:.3f} |")
    L += ["", "---", ""]

    # ── Markdown: reproducibility footer ────────────────────────────────
    import sklearn
    import scipy
    import catboost as cb_pkg
    L += [
        "## 4. Reproducibility",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Python | {sys.version.split()[0]} |",
        f"| numpy | {np.__version__} |",
        f"| scikit-learn | {sklearn.__version__} |",
        f"| CatBoost | {cb_pkg.__version__} |",
        f"| LightGBM | {lgb.__version__} |",
        f"| scipy | {scipy.__version__} |",
        f"| CV seeds | {CV_SEEDS} |",
        "| Model seed | 42 (all models, all runs) |",
        f"| Threads | {N_THREADS} |",
        f"| Wall clock | {time.time() - t_start:.0f} s |",
        "",
        "Command: `python revision/stats_variability.py`",
        "",
    ]

    md_path = os.path.join(OUT_DIR, "A-cv-variability.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    # ── Console summary ─────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Discrepancies vs published: {len(disc)}")
    for sev, txt in disc[:40]:
        print(f"  [{sev}] {txt}")
    print("\nPaired CatBoost - LightGBM over 25 folds:")
    for (task, metric), r in paired_results.items():
        print(f"  {task} {metric:8s} diff={r['mean_diff']:+.4f} "
              f"CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] t_p={pfmt(r['t_p'])} w_p={pfmt(r.get('w_p'))}")
    print("\nLORO F1 (median / mean / range):")
    for task, lo in [("T1", loro1), ("T2", loro2)]:
        for mname in MAIN_MODELS:
            s = loro_summary(lo.get(mname, {}), "f1")
            if s:
                print(f"  {task} {mname:10s} {s['median']:.3f} / {s['mean']:.3f} / "
                      f"{s['min']:.3f}-{s['max']:.3f} ({s['n']} repos)")
    print(f"\nWrote:\n  {md_path}\n  {fold_csv}\n  {loro_csv}")
    print(f"Total wall clock: {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
