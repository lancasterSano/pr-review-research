#!/usr/bin/env python3
"""Chapter 4, R4 without overlap, thresholds: theta_{tr,r} of rule R4 chosen on the predictions
of the SAME 28-repository t0_v1compat model that scores the ranking test PRs.

r4_no_overlap.py applies the thresholds of report/details/thresholds_by_repo.csv,
which were chosen on LORO predictions of the full-feature (t0) model; the test
PRs are scored by the t0_v1compat model trained on the 28 non-ranking
repositories, so the two scales differ. Here:

  1. the model of r4_no_overlap.py is refitted with r3_report_models.block_r4 unchanged
     (r4_no_overlap.TableNoRank, r4_no_overlap.fit_block); the fitted CatBoost
     object is captured and its v1 predictions are checked against
     output/r4/no_overlap/model/r4_v1_predictions.csv;
  2. the captured model (with the same train-median imputation) scores the
     main-sample rows of the 12 ranking repositories that r3_report.rule_thresholds
     uses for k = tr (r3_report.preds("main", "TFR", "loro"), label y_TFR__main_D);
     none of the 12 repositories is in its training rows;
  3. theta is chosen by r3_report.rule_thresholds itself (r3_report_stats.choose_threshold,
     MIN_FIRES = 20, p > theta, earlier half chooses, later half evaluates) for
     c = 0.1 / 0.2 / 0.3;
  4. r3_report_r4.main runs unchanged on a copy of its inputs with the new thresholds.

Checks (checks.json): r4_no_overlap.check_threshold_scale (the old file is
reproduced by rule_thresholds on the LORO predictions), the refit equals r4_no_overlap.py,
the scored rows equal M.Table.rows for the 12 repositories, the later-half
replication here equals rule_thresholds, the old R4 numbers of r4_no_overlap.py are
reproduced with the old thresholds and the refitted predictions; overlap of
the ranking test PRs with the earlier half (labels used to choose theta).

Output: revision/output/r4/no_overlap_thr/ (numbers.json with keys r4_noxt.*,
by_repo.csv, checks.json, model/, r4/, per_pr.csv).

    python revision/r4_calc/r4_no_overlap_thr.py
"""
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

import r4_no_overlap as NO   # sets R3_ROOT, R3_OUT, R3_RAW_*, R3_THREADS and sys.path

import r3_report_models as RM  # noqa: E402
import r3_report_r4 as R4  # noqa: E402
import r3_report_stats as S  # noqa: E402
import r3_v2_models as M  # noqa: E402

OUT = os.path.join(NO.REV, "output", "r4", "no_overlap_thr")
NOX = os.path.join(NO.REV, "output", "r4", "no_overlap")
CS = NO.CS
QS = NO.QS
V1 = sorted(M.V1_REPOS)


def fit_and_capture(model_dir, tab):
    """r4_no_overlap.fit_block (block_r4 unchanged); returns the fitted model."""
    box = []
    orig = RM.M.make_model

    def capture(name):
        m = orig(name)
        box.append(m)
        return m
    RM.M.make_model = capture
    try:
        NO.fit_block(model_dir, tab)
    finally:
        RM.M.make_model = orig
    assert len(box) == 1
    return box[0]


def halves(g):
    """The earlier / later split of r3_report.rule_thresholds for one repository."""
    g = g.sort_values("created")
    h = len(g) // 2
    return g.iloc[:h], g.iloc[h:]


def late_stats(d, theta_by_repo):
    """apply_threshold on the later half of each repository (None theta = no firing)."""
    out = {}
    for repo, g in d.groupby("test"):
        _, late = halves(g)
        th = theta_by_repo.get(repo)
        if th is None or (isinstance(th, float) and math.isnan(th)):
            out[repo] = {"theta": None, "n_late": len(late), "n_fire": 0, "tp": 0, "fire_share": 0.0,
                         "precision": math.nan}
        else:
            out[repo] = {"theta": th, "n_late": len(late), **S.apply_threshold(late.prob.to_numpy(), late.y.to_numpy(), th)}
    return out


def agg(stats, c_r):
    """Mean fire share over the 12 repositories (unreachable = 0, as r4.*.trigger_pct),
    pooled and median precision of the firings, repositories below 1 - c_r."""
    v = list(stats.values())
    reach = [x for x in v if x["theta"] is not None]
    nf = sum(x["n_fire"] for x in v)
    tp = sum(x["tp"] for x in v)
    prec = [x["precision"] for x in reach if x["n_fire"]]
    return {"fire_pct_mean": 100 * float(np.mean([x["fire_share"] if x["theta"] is not None else 0.0 for x in v])),
            "n_fire": int(nf), "n_late": int(sum(x["n_late"] for x in v)),
            "fire_pct_pooled": 100 * nf / sum(x["n_late"] for x in v),
            "precision_pooled": tp / nf if nf else None,
            "precision_median": float(np.median(prec)) if prec else None,
            "n_repos_prec_below": int(sum(1 for p in prec if p < 1 - c_r - 1e-12)),
            "n_unreachable": int(sum(1 for x in v if x["theta"] is None))}


def r4_run(rep_dir, pred_csv, thr_df):
    """r3_report_r4.main unchanged, on a copy of its inputs in rep_dir."""
    os.makedirs(os.path.join(rep_dir, "details"), exist_ok=True)
    shutil.copyfile(pred_csv, os.path.join(rep_dir, "r4_v1_predictions.csv"))
    thr_df.to_csv(os.path.join(rep_dir, "details", "thresholds_by_repo.csv"), index=False)
    R4.REP = rep_dir
    R4.main()
    cost = pd.read_csv(os.path.join(rep_dir, "details", "r4_cost.csv")).set_index("c")
    return (json.load(open(os.path.join(rep_dir, "numbers_r4.json"))), cost,
            pd.read_csv(os.path.join(rep_dir, "details", "r4_per_pr.csv")))


def summary(num_r4, cost, c):
    s = {q: num_r4[f"r4.{c}.{q}"] for q in QS}
    s["n_fire_test"] = int(cost.loc[c, "n_fire"])
    s["fire_test_pct"] = float(cost.loc[c, "fire_pct"])
    s["hit3_change_all"] = -float(cost.loc[c, "cost_all"])
    s["hit3_drop"] = None if s["hit3_relevance"] is None else s["hit3_relevance"] - s["hit3_latency"]
    er, el = s["exp_latency_relevance_h"], s["exp_latency_latency_h"]
    s["exp_latency_ratio"] = None if not er else el / er
    return s


def subset_summary(per, fire_col):
    x = R4.cost_table(per, fire_col)
    s = {"n_eval": int(len(per)), "n_fire_test": x["n_fire"], "fire_test_pct": x["fire_pct"],
         "hit3_change_all": -x["cost_all"]}
    for q in QS:
        s[q] = None if math.isnan(x[q]) else x[q]
    s["hit3_drop"] = None if s["hit3_relevance"] is None else s["hit3_relevance"] - s["hit3_latency"]
    er, el = s["exp_latency_relevance_h"], s["exp_latency_latency_h"]
    s["exp_latency_ratio"] = None if not er else el / er
    return s


def main():
    os.makedirs(OUT, exist_ok=True)
    checks = {"git": NO.git_head(), "thresholds_sha256": NO.sha256(NO.THR)}
    checks["git"]["this_script_dirty"] = bool(subprocess.run(
        ["git", "-C", NO.CLONE, "status", "--porcelain", "--", os.path.relpath(__file__, NO.CLONE)],
        capture_output=True, text=True).stdout.strip())

    # 0. the old thresholds file is rule_thresholds on the LORO predictions
    checks["threshold_scale"] = thr_chk = NO.check_threshold_scale()
    print("threshold scale:", json.dumps(thr_chk, ensure_ascii=False), flush=True)
    if not thr_chk["ok"]:
        json.dump(checks, open(os.path.join(OUT, "checks.json"), "w"), ensure_ascii=False, indent=1)
        sys.exit("old thresholds not reproduced: stop")
    import r3_report as RR   # already loaded by check_threshold_scale

    # 1. refit the model of r4_no_overlap.py and capture it
    tab = NO.TableNoRank(os.path.join(os.environ["R3_OUT"], "pr_table_v2.csv"))
    model_dir = os.path.join(OUT, "model")
    model = fit_and_capture(model_dir, tab)
    a = pd.read_csv(os.path.join(model_dir, "r4_v1_predictions.csv"))
    b = pd.read_csv(os.path.join(NOX, "model", "r4_v1_predictions.csv"))
    same_keys = bool((a.repo.to_numpy() == b.repo.to_numpy()).all()
                     and (a.pr_number.to_numpy() == b.pr_number.to_numpy()).all())
    checks["refit_equals_no20"] = {"same_rows": same_keys,
                                   "p_tr_max_abs_diff": float(np.max(np.abs(a.p_tr.to_numpy() - b.p_tr.to_numpy())))}
    info = json.load(open(os.path.join(model_dir, "r4_model_info.json")))
    feats = info["features"]
    assert feats == list(M.FEATURE_SETS["t0_v1compat"])
    idx_tr, _ = tab.rows("main", "D", "TFR")
    train_repos = set(tab.t["repo"].to_numpy()[idx_tr])
    checks["train_repos_n"] = len(train_repos)
    checks["train_repos_disjoint_from_12"] = not (train_repos & set(V1))
    med_info = np.array([info["impute_median"][f] for f in feats])
    med = np.nanmedian(tab.X(idx_tr, feats, "D"), axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    checks["impute_median_equal"] = bool(np.array_equal(med, med_info))

    # 2. score the main-sample rows of the 12 repositories used by rule_thresholds (k = tr)
    d_all = RR.preds("main", "TFR", "loro")
    d_old = d_all[d_all.test.isin(V1)].copy()
    idx12, y12 = M.Table.rows(tab, "main", "D", "TFR", repos=V1)
    checks["scored_rows"] = {
        "n": int(len(d_old)), "repos": int(d_old.test.nunique()),
        "test_equals_repo": bool((d_old.test == d_old.repo).all()),
        "rows_equal_table_rows": bool(np.array_equal(np.sort(d_old.row_id.to_numpy()), np.sort(tab.t.row_id.to_numpy()[idx12]))),
        "labels_equal": bool(np.array_equal(
            d_old.sort_values("row_id").y.to_numpy().astype(int),
            pd.Series(y12, index=tab.t.row_id.to_numpy()[idx12]).sort_index().to_numpy()))}
    X = tab.X(d_old.row_id.to_numpy(), feats, "D")
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = med[j]
    d_new = d_old.copy()
    d_new["prob"] = model.predict_proba(X)[:, 1]
    d_new["pr_number"] = tab.t["pr_number"].to_numpy()[d_new.row_id.to_numpy()]
    d_new[["row_id", "repo", "pr_number", "created", "y", "prob"]].assign(prob_loro_t0=d_old.prob.to_numpy()).to_csv(
        os.path.join(OUT, "main12_predictions.csv"), index=False)

    # 3. thresholds by rule_thresholds itself, new (t0_v1compat, 28 repos) and old (LORO t0)
    thr_new = {c: RR.rule_thresholds(d_new, cr) for c, cr in RR.CR.items()}
    thr_old = {c: RR.rule_thresholds(d_old, cr) for c, cr in RR.CR.items()}
    old_csv = pd.read_csv(NO.THR)
    old_csv = old_csv[(old_csv.k == "tr") & old_csv.repo.isin(V1)]
    mism = 0
    for _, r in old_csv.iterrows():
        x = thr_old[r.c][r.repo]["theta"]
        mism += int((x is None) != pd.isna(r.theta) or (x is not None and abs(x - r.theta) > 1e-12))
    checks["old_thresholds_12_reproduced"] = {"rows": int(len(old_csv)), "mismatches": mism}
    # later-half replication here == rule_thresholds
    rep_ok = True
    for c in CS:
        ls = late_stats(d_new, {r: v["theta"] for r, v in thr_new[c].items()})
        for r, v in thr_new[c].items():
            if v["theta"] is not None and (ls[r]["n_fire"] != v["n_fire"] or ls[r]["tp"] != v["tp"]):
                rep_ok = False
    checks["late_half_replication_ok"] = rep_ok
    thr_df = pd.DataFrame([{"k": "tr", "c": c, "repo": r, **rec} for c in CS for r, rec in thr_new[c].items()])
    thr_df.to_csv(os.path.join(OUT, "thresholds_by_repo_new.csv"), index=False)

    # 4. R4 with the new thresholds; old thresholds as a check against r4_no_overlap.py
    pred = os.path.join(model_dir, "r4_v1_predictions.csv")
    num_new, cost_new, per_new = r4_run(os.path.join(OUT, "r4"), pred, thr_df)
    checks["e4_consistency_new"] = json.load(open(os.path.join(OUT, "r4", "details", "r4_detail.json")))["check"]
    num_oc, cost_oc, per_oc = r4_run(os.path.join(OUT, "old_thr_check"), pred, pd.read_csv(NO.THR))
    nox = json.load(open(os.path.join(NOX, "numbers.json")))
    diffs = []
    for c in CS:
        s = summary(num_oc, cost_oc, c)
        for q, v in s.items():
            w = nox[f"r4_nox.{c}.{q}"]
            diffs.append(0.0 if v is None and w is None else abs(v - w))
    checks["no20_numbers_reproduced_max_abs_diff"] = float(max(diffs))

    num = {"r4_noxt.n_prs_eval": num_new["r4.n_prs_eval"], "r4_noxt.n_main12_rows": int(len(d_new)),
           "r4_noxt.n_train": info["n_train"], "r4_noxt.n_train_repos": len(train_repos)}
    by_repo = []
    for c, cr in RR.CR.items():
        s_new = summary(num_new, cost_new, c)
        s_old = {q: nox[f"r4_nox.{c}.{q}"] for q in s_new}
        for q in s_new:
            num[f"r4_noxt.{c}.{q}"] = s_new[q]
            num[f"r4_noxt.old.{c}.{q}"] = s_old[q]
            num[f"r4_noxt.diff.{c}.{q}"] = None if s_new[q] is None or s_old[q] is None else s_new[q] - s_old[q]
        # later half of the main sample of the 12 repositories
        th_new = {r: v["theta"] for r, v in thr_new[c].items()}
        th_old = {r: v["theta"] for r, v in thr_old[c].items()}
        variants = {"new": late_stats(d_new, th_new),            # new theta, new model
                    "old": late_stats(d_new, th_old),            # old theta, new model (what r4_no_overlap.py applied)
                    "design": late_stats(d_old, th_old)}         # old theta, LORO t0 (as designed in R3)
        for tag, st in variants.items():
            for q, v in agg(st, cr).items():
                num[f"r4_noxt.late.{tag}.{c}.{q}"] = v
        num[f"r4_noxt.{c}.n_unreachable"] = agg(variants["new"], cr)["n_unreachable"]
        num[f"r4_noxt.old.{c}.n_unreachable"] = agg(variants["design"], cr)["n_unreachable"]
        for r in V1:
            fr = per_new[per_new.repo_full == r]
            fo = per_oc[per_oc.repo_full == r]
            row = {"c": c, "repo": r, "n_early": thr_new[c][r]["n_early"], "n_late": thr_new[c][r]["n_late"],
                   "theta_new": th_new[r], "theta_old": th_old[r],
                   "reachable_new": th_new[r] is not None, "reachable_old": th_old[r] is not None,
                   "n_test": int(fr.p_tr.notna().sum()),
                   "fire_test_pct_new": 100 * float(fr[f"fire_{c}"].mean()) if len(fr) else math.nan,
                   "fire_test_pct_old": 100 * float(fo[f"fire_{c}"].mean()) if len(fo) else math.nan}
            for tag, st in variants.items():
                x = st[r]
                row[f"late_fire_pct_{tag}"] = 100 * x["fire_share"] if x["theta"] is not None else 0.0
                row[f"late_n_fire_{tag}"] = x["n_fire"]
                row[f"late_precision_{tag}"] = x["precision"]
            by_repo.append(row)

    # 5. ranking test PRs that are earlier-half rows (their labels chose theta)
    early = {}
    for repo, g in d_new.groupby("test"):
        e, l = halves(g)
        early[repo] = (set(e.pr_number.astype(int)), set(l.pr_number.astype(int)), e.created.max())
    per = per_new[per_new.p_tr.notna()].copy()
    per["in_early"] = [int(p) in early[r][0] for r, p in zip(per.repo_full, per.pr)]
    per["in_late"] = [int(p) in early[r][1] for r, p in zip(per.repo_full, per.pr)]
    po = per_oc[per_oc.p_tr.notna()][["repo_full", "pr"] + [f"fire_{c}" for c in CS]]
    per = per.merge(po, on=["repo_full", "pr"], suffixes=("", "_oldthr"))
    per.to_csv(os.path.join(OUT, "per_pr.csv"), index=False)
    num["r4_noxt.early.n_test_in_early"] = int(per.in_early.sum())
    num["r4_noxt.early.pct_test_in_early"] = 100 * float(per.in_early.mean())
    num["r4_noxt.early.n_test_in_late"] = int(per.in_late.sum())
    num["r4_noxt.early.n_test_not_in_main"] = int((~per.in_early & ~per.in_late).sum())
    checks["early_by_repo"] = {r: {"n_test": int((per.repo_full == r).sum()),
                                   "in_early": int(per[per.repo_full == r].in_early.sum()),
                                   "in_late": int(per[per.repo_full == r].in_late.sum()),
                                   "early_last_created": str(early[r][2])} for r in V1}
    sub = per[~per.in_early]
    for c in CS:
        for tag, col in (("", f"fire_{c}"), ("old.", f"fire_{c}_oldthr")):
            for q, v in subset_summary(sub, col).items():
                num[f"r4_noxt.excl_early.{tag}{c}.{q}"] = v

    # 6. verdict of R4: ~half the expected time at a Hit@3 drop < 0.01 for c = 0.2, 0.3
    for tag in ("", "excl_early."):
        ok = all(num[f"r4_noxt.{tag}{c}.hit3_drop"] is not None and num[f"r4_noxt.{tag}{c}.hit3_drop"] < 0.01
                 and 0.35 <= num[f"r4_noxt.{tag}{c}.exp_latency_ratio"] <= 0.65 for c in ("c02", "c03"))
        checks[f"verdict_holds{'_' + tag.rstrip('.') if tag else ''}"] = ok   # ratio in [0.35, 0.65]

    pd.DataFrame(by_repo).to_csv(os.path.join(OUT, "by_repo.csv"), index=False)
    json.dump(num, open(os.path.join(OUT, "numbers.json"), "w"), ensure_ascii=False, indent=1)
    json.dump(checks, open(os.path.join(OUT, "checks.json"), "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(num, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in checks.items() if k != "early_by_repo"}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
