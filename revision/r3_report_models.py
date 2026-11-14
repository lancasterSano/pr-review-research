#!/usr/bin/env python3
"""Chapter 3 reporting step: the few new fits (blocks q75, r4, nmin10).

  q75   sensitivity label of rule R4: t_r above the upper quartile of the
        project history (main sample, clock D, history minimum 30);
        CatBoost, t0 features, LORO over 40 repositories.
  r4    model P_tr for the R4 cost: CatBoost, label TFR (main, D), features
        t0_v1compat, trained on all 40 repositories of corpus v2, applied to
        the merged PRs of the 12 corpus v1 repositories (features computed by
        the same r3_v2_prep.prep_repo from data/raw/<repo>/merged_prs.json).

Outputs go to <R3_OUT>/../report/ (never into the night-run folders).

    R3_OUT=revision/output/r3/v2 python revision/r3_report_models.py --block q75|r4|nmin10
"""
import argparse
import bisect
import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import r3_v2_models as M  # noqa: E402
from r3_v2_prep import N_MIN_HISTORY, load_repo_meta, prep_repo, r3_paths  # noqa: E402


def _quantile_sorted(s, q):
    """numpy.quantile(s, q) with the default linear interpolation, s sorted."""
    n = len(s)
    if n == 0:
        return math.nan
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def prefix_quantile_labels(created, merged, tr_pre, tr_own, q, n_min=N_MIN_HISTORY):
    """Labels "t_r above the q-quantile of t_r of the project's sample PRs merged
    strictly before t0" for ONE repository.

    created, merged: comparable timestamps of the sample PRs (same order);
    tr_pre: t_r of each PR as known at its merge (None/NaN = undefined);
    tr_own: t_r of the PR itself (None/NaN = undefined).
    Returns (labels, n_hist): label NaN if the history has < n_min defined
    values or the PR's own t_r is undefined.
    """
    n = len(created)
    order = sorted(range(n), key=lambda i: merged[i])
    m_sorted = [merged[i] for i in order]
    vals_by_k = []          # defined t_r_pre in merge order
    pos_by_k = []           # their index in merge order
    for k, i in enumerate(order):
        v = tr_pre[i]
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals_by_k.append(v)
            pos_by_k.append(k)
    # prefix quantiles over the first j defined values
    s, qs = [], [math.nan]
    for v in vals_by_k:
        bisect.insort(s, v)
        qs.append(_quantile_sorted(s, q))
    labels, nh = [math.nan] * n, [0] * n
    for i in range(n):
        k = bisect.bisect_left(m_sorted, created[i])
        j = bisect.bisect_left(pos_by_k, k)
        nh[i] = j
        own = tr_own[i]
        if j >= n_min and own is not None and not (isinstance(own, float) and math.isnan(own)):
            labels[i] = int(own > qs[j])
    return labels, nh


def add_q75_label(tab, q=0.75):
    """Adds y_TFRQ75__main_D to the table (main sample, clock D)."""
    t = tab.t
    col = "y_TFRQ75__main_D"
    t[col] = np.nan
    s = t[(t.hait_filter == 1) & (t.in_main == 1)]
    for repo, g in s.groupby("repo"):
        lab, _ = prefix_quantile_labels(
            g["created"].tolist(), g["merged"].tolist(),
            g["t_r_pre_h_D"].astype(float).tolist(), g["t_r_h_D"].astype(float).tolist(), q)
        t.loc[g.index, col] = lab
    return col


def block_q75(out_dir, tab):
    add_q75_label(tab)
    tab.t[["row_id", "repo", "pr_number", "y_TFRQ75__main_D"]].to_csv(
        os.path.join(out_dir, "labels_tfr_q75.csv"), index=False)
    store = M.Store(os.path.join(out_dir, "models"))
    M.run_config(store, tab, "sens_tr_q75", "TFR", "CatBoost", ["loro"], label="TFRQ75")
    print(f"q75 done: {store.n_fit} new fits", flush=True)


def add_nmin_labels(tab, n_min):
    """y_TM_N{n}__main_D and y_TFR_N{n}__main_D: median labels with history
    minimum n_min (the night table defines labels only for history >= 30, so
    its n_min_10 variant equalled the main run; fixed 31.10.2026)."""
    t = tab.t
    s = t[(t.hait_filter == 1) & (t.in_main == 1)]
    out = {}
    for lab, pre, own in (("TM", "t_m_h_D", "t_m_h_D"), ("TFR", "t_r_pre_h_D", "t_r_h_D")):
        col = f"y_{lab}_N{n_min}__main_D"
        t[col] = np.nan
        mism = 0
        for repo, g in s.groupby("repo"):
            y, _ = prefix_quantile_labels(g["created"].tolist(), g["merged"].tolist(),
                                          g[pre].astype(float).tolist(), g[own].astype(float).tolist(),
                                          0.5, n_min=n_min)
            t.loc[g.index, col] = y
            if n_min <= N_MIN_HISTORY:   # check: where the night label exists it must agree
                a = np.array(y, float)
                b = g[f"y_{lab}__main_D"].to_numpy(float)
                ok = ~np.isnan(b)
                mism += int((a[ok] != b[ok]).sum())
        out[col] = mism
    return out


def block_nmin10(out_dir, tab):
    chk = add_nmin_labels(tab, 10)
    print(f"nmin10 label check (mismatches with night labels where defined): {chk}", flush=True)
    tab.t[["row_id", "y_TM_N10__main_D", "y_TFR_N10__main_D"]].to_csv(
        os.path.join(out_dir, "labels_nmin10.csv"), index=False)
    store = M.Store(os.path.join(out_dir, "models"))
    for tg in ("TM", "TFR"):
        M.run_config(store, tab, "n_min_10_fixed", tg, "CatBoost", ["loro"], n_min=10, label=f"{tg}_N10")
    print(f"nmin10 done: {store.n_fit} new fits", flush=True)


V1_DIR = {r: r.replace("/", "_", 1) for r in M.V1_REPOS}


def v1_features(raw_v1, meta_v2):
    """Per-PR table of the 12 corpus v1 repositories (merged PRs only)."""
    import pandas as pd
    rows = []
    for repo in M.V1_REPOS:
        path = os.path.join(raw_v1, V1_DIR[repo], "merged_prs.json")
        with open(path, encoding="utf-8") as f:
            prs = json.load(f)
        r, _ = prep_repo(repo, meta_v2[repo], iter(prs))
        rows.extend(r)
        print(f"  v1 {repo}: {len(r)} merged", flush=True)
    return pd.DataFrame(rows)


def block_r4(out_dir, tab, raw_v1, meta_v2):
    import pandas as pd
    from catboost import CatBoostClassifier  # noqa: F401  (make_model imports it)
    feats = M.FEATURE_SETS["t0_v1compat"]
    idx, y = tab.rows("main", "D", "TFR")
    X = tab.X(idx, feats, "D")
    Xtr = X.copy()
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    for j in range(Xtr.shape[1]):
        Xtr[np.isnan(Xtr[:, j]), j] = med[j]
    model = M.make_model("CatBoost")
    t = time.time()
    model.fit(Xtr, y)
    print(f"r4 model fitted on {len(y)} rows ({time.time() - t:.0f}s)", flush=True)
    v1 = v1_features(raw_v1, meta_v2)
    v1.to_csv(os.path.join(out_dir, "v1_pr_table.csv"), index=False)
    M.check_features(feats)
    Xv = v1[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    for j in range(Xv.shape[1]):
        Xv[np.isnan(Xv[:, j]), j] = med[j]
    v1["p_tr"] = model.predict_proba(Xv)[:, 1]
    v1[["repo", "pr_number", "created_at", "p_tr"] + feats].to_csv(
        os.path.join(out_dir, "r4_v1_predictions.csv"), index=False)
    json.dump({"n_train": int(len(y)), "pos_train": float(np.mean(y)), "features": feats,
               "impute_median": dict(zip(feats, map(float, med)))},
              open(os.path.join(out_dir, "r4_model_info.json"), "w"), indent=1)
    print(f"r4 predictions: {len(v1)} v1 PRs", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", choices=["q75", "r4", "nmin10"], required=True)
    args = ap.parse_args()
    P = r3_paths()
    out_dir = os.path.join(os.path.dirname(P["out"]), "report")
    os.makedirs(out_dir, exist_ok=True)
    os.environ.setdefault("R3_THREADS", "6")
    tab = M.Table(os.path.join(P["out"], "pr_table_v2.csv"))
    if args.block == "q75":
        block_q75(out_dir, tab)
    elif args.block == "nmin10":
        block_nmin10(out_dir, tab)
    else:
        raw_v1 = os.environ.get("R3_RAW_V1", os.path.join(os.path.dirname(P["raw_v2"]), "raw"))
        block_r4(out_dir, tab, raw_v1, load_repo_meta(P["raw_v2"]))


if __name__ == "__main__":
    main()
