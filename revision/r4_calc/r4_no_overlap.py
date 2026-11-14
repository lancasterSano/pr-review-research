#!/usr/bin/env python3
"""Chapter 4: rule R4 "availability priority" without overlap between the
training rows of the long-first-review model P_tr and the ranking test PRs.

The chapter 3 model (r3_report_models.block_r4) is trained on all 40
repositories of the main sample; the 12 ranking repositories (corpus v1) are
among them. Here the SAME block is run with the training rows restricted to the
28 other repositories (a Table whose rows() drops the 12), and the SAME
r3_report_r4.main computes the R4 table on its predictions. Thresholds are NOT
recomputed: report/details/thresholds_by_repo.csv (k = tr) is copied as is.

Checks written to checks.json:
  * thresholds_by_repo.csv (k = tr) equals r3_report.rule_thresholds on the raw
    LORO output of CatBoost (auto_class_weights="Balanced", no prior correction,
    no calibration) with r3_report_stats.MIN_FIRES = 20 firings;
  * the share of main-sample PRs with R4 firing (r4.*.trigger_pct) is
    reproduced from that file (it does not depend on the model here);
  * the old R4 numbers are reproduced from the old predictions with the current
    thresholds file (r3_report_r4.main on a copy);
  * --repro-old: block_r4 on all 40 repositories reproduces the old predictions.

Inputs (read only): revision/output/r3_b40d/{v2/pr_table_v2.csv, v2/models,
report/*, rank_shared/results}, raw v1 (R3_RAW_V1), raw v2 meta (R3_RAW_V2).
Output: revision/output/r4/no_overlap/ (numbers.json, per_pr.csv,
model_info.json, checks.json, model/, old_check/).

    python revision/r4_calc/r4_no_overlap.py [--repro-old]
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(_HERE)
CLONE = os.path.dirname(REV)
R3_ROOT = os.environ.get("R3_ROOT") or os.path.join(REV, "output", "r3_b40d")
COLLECTOR = CLONE  # data/ of this repository unless R3_RAW_V2 / R3_RAW_V1 are set
# the environment of the 02.11 rerun (revision/r3_rerun_0211.sh); set before the imports
os.environ["R3_ROOT"] = R3_ROOT
os.environ.setdefault("R3_OUT", os.path.join(R3_ROOT, "v2"))
os.environ.setdefault("R3_RAW_V2", os.path.join(COLLECTOR, "data", "raw_v2"))
os.environ.setdefault("R3_RAW_V1", os.path.join(COLLECTOR, "data", "raw"))
os.environ.setdefault("R3_THREADS", "6")
sys.path.insert(0, CLONE)
sys.path.insert(0, REV)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import r3_report_models as RM  # noqa: E402
import r3_report_r4 as R4  # noqa: E402
import r3_report_stats as S  # noqa: E402
import r3_v2_models as M  # noqa: E402
from r3_v2_prep import N_MIN_HISTORY, load_repo_meta  # noqa: E402

OUT = os.path.join(REV, "output", "r4", "no_overlap")
REP_OLD = os.path.join(R3_ROOT, "report")
THR = os.path.join(REP_OLD, "details", "thresholds_by_repo.csv")
DISS_NUM = os.environ.get("R3_NUMBERS_JSON", "")  # optional: chapter 3 numbers to compare with
CS = list(R4.CR)
QS = ("hit3_relevance", "hit3_latency", "exp_latency_relevance_h", "exp_latency_latency_h")


class TableNoRank(M.Table):
    """The night table whose default training rows leave out the 12 ranking repositories."""
    exclude = set(M.V1_REPOS)

    def rows(self, pop, clock, target, repos=None, *a, **kw):
        if repos is None:
            repos = sorted(set(self.t["repo"]) - self.exclude)
        return super().rows(pop, clock, target, repos, *a, **kw)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_head():
    def run(*a):
        return subprocess.run(["git", "-C", CLONE, *a], capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "script_dirty": bool(run("status", "--porcelain", "--", os.path.relpath(__file__, CLONE)))}


def fit_block(out_dir, tab):
    """r3_report_models.block_r4 unchanged, writing into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    RM.block_r4(out_dir, tab, os.environ["R3_RAW_V1"], load_repo_meta(os.environ["R3_RAW_V2"]))


def r4_table(rep_dir, pred_csv):
    """r3_report_r4.main unchanged, on a copy of its inputs in rep_dir."""
    os.makedirs(os.path.join(rep_dir, "details"), exist_ok=True)
    if os.path.abspath(pred_csv) != os.path.abspath(os.path.join(rep_dir, "r4_v1_predictions.csv")):
        shutil.copyfile(pred_csv, os.path.join(rep_dir, "r4_v1_predictions.csv"))
    shutil.copyfile(THR, os.path.join(rep_dir, "details", "thresholds_by_repo.csv"))
    R4.REP = rep_dir
    R4.main()
    cost = pd.read_csv(os.path.join(rep_dir, "details", "r4_cost.csv")).set_index("c")
    return (json.load(open(os.path.join(rep_dir, "numbers_r4.json"))), cost,
            pd.read_csv(os.path.join(rep_dir, "details", "r4_per_pr.csv")))


# ── checks ──────────────────────────────────────────────────────────────────

def check_threshold_scale():
    """thresholds_by_repo.csv (k = tr) against rule_thresholds on the raw LORO prob."""
    import inspect
    import r3_report as RR   # loads the night table (read only)
    d = RR.preds("main", "TFR", "loro")
    csv = pd.read_csv(THR)
    csv = csv[csv.k == "tr"]
    max_abs, mism = 0.0, 0
    for c, cr in RR.CR.items():
        rec = RR.rule_thresholds(d, cr)
        for _, r in csv[csv.c == c].iterrows():
            a, b = r.theta, rec[r.repo]["theta"]
            if pd.isna(a) != (b is None):
                mism += 1
            elif b is not None:
                max_abs = max(max_abs, abs(a - b))
    src = inspect.getsource(M.make_model)
    mf_default = inspect.signature(S.choose_threshold).parameters["min_fires"].default
    return {"rows_tr": int(len(csv)), "reachability_mismatches": mism, "theta_max_abs_diff": max_abs,
            "prob_source": "models/oof *.npz prob = make_model('CatBoost').predict_proba (fit_predict), no calibration",
            "auto_class_weights_balanced": 'auto_class_weights="Balanced"' in src,
            "rule_thresholds_uses_default_min_fires": "min_fires" not in inspect.getsource(RR.rule_thresholds),
            "choose_threshold_min_fires_default": mf_default, "MIN_FIRES": S.MIN_FIRES,
            "n_oof_rows": int(len(d)), "n_repos": int(d.test.nunique()),
            "ok": bool(mism == 0 and max_abs < 1e-12 and mf_default == 20 and S.MIN_FIRES == 20
                       and "min_fires" not in inspect.getsource(RR.rule_thresholds)
                       and 'auto_class_weights="Balanced"' in src)}


def trigger_pct_from_csv():
    """r4.*.trigger_pct as in r3_report.thresholds_transfer_r4, from the saved file."""
    t = pd.read_csv(THR)
    ok = None
    for k in ("it", "st", "tr"):
        g = t[(t.k == k) & (t.c == "c01") & (t.n_early >= 20) & (t.n_late >= 1)]
        ok = set(g.repo) if ok is None else ok & set(g.repo)
    out = {}
    for c in CS:
        g = t[(t.k == "tr") & (t.c == c) & t.repo.isin(ok)]
        out[c] = 100 * float(np.where(g.theta.notna(), g.fire_share.fillna(0.0), 0.0).mean())
    return out, len(ok)


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    y, p = np.asarray(y), np.asarray(p)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else math.nan


def auc_block(per_new, per_old, v1):
    """Descriptive AUC of P_tr (new and old) on the evaluated ranking test PRs;
    label y_TFR__main_D of the v1 table (history >= N_MIN_HISTORY)."""
    lab = v1[["repo", "pr_number", "y_TFR__main_D", "n_hist_tr__main_D"]].rename(
        columns={"repo": "repo_full", "pr_number": "pr"})
    m = per_new[["repo_full", "pr", "p_tr"]].merge(
        per_old[["repo_full", "pr", "p_tr"]], on=["repo_full", "pr"], suffixes=("_new", "_old"))
    m = m.merge(lab, on=["repo_full", "pr"], how="left")
    m = m[m.p_tr_new.notna() & m.p_tr_old.notna() & m.y_TFR__main_D.notna()
          & (m.n_hist_tr__main_D >= N_MIN_HISTORY)]
    y = m.y_TFR__main_D.astype(int)
    per = {r: (auc(g.y_TFR__main_D.astype(int), g.p_tr_new), auc(g.y_TFR__main_D.astype(int), g.p_tr_old))
           for r, g in m.groupby("repo_full")}
    return {"n": int(len(m)), "pos_pct": 100 * float(y.mean()),
            "pooled_new": auc(y, m.p_tr_new), "pooled_old": auc(y, m.p_tr_old),
            "repo_mean_new": float(np.nanmean([a for a, _ in per.values()])),
            "repo_mean_old": float(np.nanmean([b for _, b in per.values()])),
            "by_repo": {r: {"new": a, "old": b} for r, (a, b) in per.items()}}


def overlap(tab, per):
    """Evaluated ranking test PRs that were training rows of the old (40-repo) model."""
    idx, _ = M.Table.rows(tab, "main", "D", "TFR")
    tr_old = tab.t.iloc[idx][["repo", "pr_number"]].rename(columns={"repo": "repo_full", "pr_number": "pr"})
    idx_new, _ = tab.rows("main", "D", "TFR")
    tr_new = tab.t.iloc[idx_new][["repo", "pr_number"]].rename(columns={"repo": "repo_full", "pr_number": "pr"})
    ev = per[per.p_tr.notna()][["repo_full", "pr"]]
    k = lambda df: set(zip(df.repo_full, df.pr.astype(int)))  # noqa: E731
    e = k(ev)
    by = {r: int(len({x for x in e & k(tr_old) if x[0] == r})) for r in sorted(M.V1_REPOS)}
    return {"n_eval": len(e), "in_old_train": len(e & k(tr_old)),
            "in_old_train_pct": 100 * len(e & k(tr_old)) / len(e), "in_new_train": len(e & k(tr_new)),
            "by_repo_in_old_train": by}


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro-old", action="store_true", help="also refit the 40-repository model")
    ap.add_argument("--skip-fit", action="store_true", help="reuse model/r4_v1_predictions.csv")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    checks = {"git": git_head(), "thresholds_sha256": sha256(THR)}

    checks["threshold_scale"] = thr = check_threshold_scale()
    print("threshold scale:", json.dumps({k: v for k, v in thr.items()}, ensure_ascii=False), flush=True)
    if not thr["ok"]:
        json.dump(checks, open(os.path.join(OUT, "checks.json"), "w"), ensure_ascii=False, indent=1)
        sys.exit("thresholds_by_repo.csv is not on the raw P_tr scale with 20 firings: stop")

    tab = TableNoRank(os.path.join(os.environ["R3_OUT"], "pr_table_v2.csv"))
    model_dir = os.path.join(OUT, "model")
    if not args.skip_fit:
        fit_block(model_dir, tab)
    if args.repro_old:
        fit_block(os.path.join(OUT, "repro_old_fit"), M.Table(os.path.join(os.environ["R3_OUT"], "pr_table_v2.csv")))
        a = pd.read_csv(os.path.join(OUT, "repro_old_fit", "r4_v1_predictions.csv"))
        b = pd.read_csv(os.path.join(REP_OLD, "r4_v1_predictions.csv"))
        checks["repro_old_fit_p_tr_max_abs_diff"] = float(np.max(np.abs(a.p_tr.to_numpy() - b.p_tr.to_numpy())))

    # old numbers with the current thresholds file, then the new model
    num_old, cost_old, per_old = r4_table(os.path.join(OUT, "old_check"), os.path.join(REP_OLD, "r4_v1_predictions.csv"))
    saved_old = json.load(open(os.path.join(REP_OLD, "numbers_r4.json")))
    checks["old_numbers_reproduced"] = all(
        (saved_old[k] is None and num_old[k] is None) or abs(saved_old[k] - num_old[k]) < 1e-12 for k in saved_old)
    num_new, cost_new, per_new = r4_table(model_dir, os.path.join(model_dir, "r4_v1_predictions.csv"))
    checks["e4_consistency_new"] = json.load(open(os.path.join(model_dir, "details", "r4_detail.json")))["check"]

    trig, n_ev_repos = trigger_pct_from_csv()
    diss = json.load(open(DISS_NUM))["r4"] if os.path.exists(DISS_NUM) else {}
    checks["trigger_pct_matches_r3_numbers"] = {c: (diss.get(c, {}).get("trigger_pct"), trig[c]) for c in CS}

    num = {"r4_nox.n_prs_eval": num_new["r4.n_prs_eval"], "r4_nox.old.n_prs_eval": saved_old["r4.n_prs_eval"]}
    for c in CS:
        num[f"r4_nox.{c}.trigger_pct"] = trig[c]
        for tag, src, cost in (("", num_new, cost_new), ("old.", saved_old, cost_old)):
            for q in QS:
                num[f"r4_nox.{tag}{c}.{q}"] = src[f"r4.{c}.{q}"]
            num[f"r4_nox.{tag}{c}.n_fire_test"] = int(cost.loc[c, "n_fire"])
            num[f"r4_nox.{tag}{c}.fire_test_pct"] = float(cost.loc[c, "fire_pct"])
            num[f"r4_nox.{tag}{c}.hit3_change_all"] = -float(cost.loc[c, "cost_all"])
            r, l = src[f"r4.{c}.hit3_relevance"], src[f"r4.{c}.hit3_latency"]
            num[f"r4_nox.{tag}{c}.hit3_drop"] = None if r is None else r - l
            er, el = src[f"r4.{c}.exp_latency_relevance_h"], src[f"r4.{c}.exp_latency_latency_h"]
            num[f"r4_nox.{tag}{c}.exp_latency_ratio"] = None if not er else el / er
        for q in QS + ("n_fire_test", "fire_test_pct", "hit3_change_all", "hit3_drop", "exp_latency_ratio"):
            a, b = num[f"r4_nox.{c}.{q}"], num[f"r4_nox.old.{c}.{q}"]
            num[f"r4_nox.diff.{c}.{q}"] = None if a is None or b is None else a - b

    v1 = pd.read_csv(os.path.join(model_dir, "v1_pr_table.csv"), low_memory=False)
    a = auc_block(per_new, per_old, v1)
    num.update({"r4_nox.auc.n": a["n"], "r4_nox.auc.pos_pct": a["pos_pct"],
                "r4_nox.auc.pooled_new": a["pooled_new"], "r4_nox.auc.pooled_old": a["pooled_old"],
                "r4_nox.auc.repo_mean_new": a["repo_mean_new"], "r4_nox.auc.repo_mean_old": a["repo_mean_old"]})
    ov = overlap(tab, per_new)
    num.update({"r4_nox.overlap.n_eval": ov["n_eval"], "r4_nox.overlap.in_old_train": ov["in_old_train"],
                "r4_nox.overlap.in_old_train_pct": ov["in_old_train_pct"],
                "r4_nox.overlap.in_new_train": ov["in_new_train"]})
    info = json.load(open(os.path.join(model_dir, "r4_model_info.json")))
    idx, _ = tab.rows("main", "D", "TFR")
    train_repos = sorted(set(tab.t["repo"].to_numpy()[idx]))
    num.update({"r4_nox.n_train": info["n_train"], "r4_nox.pos_train_pct": 100 * info["pos_train"],
                "r4_nox.n_train_repos": len(train_repos)})
    old_info = json.load(open(os.path.join(REP_OLD, "r4_model_info.json")))
    num.update({"r4_nox.old.n_train": old_info["n_train"], "r4_nox.old.pos_train_pct": 100 * old_info["pos_train"]})

    json.dump(num, open(os.path.join(OUT, "numbers.json"), "w"), ensure_ascii=False, indent=1)
    json.dump({**info, "train_repos": train_repos, "excluded_repos": sorted(M.V1_REPOS), "seed": M.SEED,
               "model": "CatBoost (r3_v2_models.make_model)", "threads": int(os.environ["R3_THREADS"]),
               "label": "y_TFR__main_D, history >= %d" % N_MIN_HISTORY, "clone": checks["git"],
               "pr_table": os.path.join(os.environ["R3_OUT"], "pr_table_v2.csv"),
               "thresholds": THR, "thresholds_sha256": checks["thresholds_sha256"]},
              open(os.path.join(OUT, "model_info.json"), "w"), ensure_ascii=False, indent=1)
    per = per_new.merge(per_old[["repo_full", "pr", "p_tr"] + [f"fire_{c}" for c in CS]],
                        on=["repo_full", "pr"], suffixes=("", "_old"))
    per.to_csv(os.path.join(OUT, "per_pr.csv"), index=False)
    checks.update({"auc_by_repo": a["by_repo"], "overlap_by_repo": ov["by_repo_in_old_train"],
                   "trigger_eval_repos": n_ev_repos})
    json.dump(checks, open(os.path.join(OUT, "checks.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(num, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in checks.items() if k not in ("auc_by_repo",)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
