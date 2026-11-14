#!/usr/bin/env python3
"""Chapter 3 on corpus v2: models over the per-PR table of r3_v2_prep.py.

Every fit is appended to <out>/metrics.jsonl and its test predictions to
<out>/oof/<key>.npz, so a restarted run continues where it stopped. The report
(cluster bootstrap by repository, calibration, thresholds for c_r, language
effect) is a separate stage that runs AFTER the evaluation thresholds are
fixed in advance; this module only fits and stores predictions.

    python revision/r3_v2_models.py --block main|controls|ladder|sensitivity|temporal|heuristics
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from r3_v2_prep import FEATURES_T0, FORBIDDEN, N_MIN_HISTORY, SIZE_T0_FEATURES, TEXT_FEATURES, r3_paths  # noqa: E402

N_THREADS = int(os.environ.get("R3_THREADS", "8"))
SEED = 42
TARGETS = ["IT", "ST", "TM", "TFR"]
SHUFFLE_REPEATS = 3
RANDLANG_REPEATS = 5
TEMPORAL_QUANTILE = 0.7
V1_REPOS = ["angular/angular", "discourse/discourse", "django/django", "facebook/react",
            "grafana/grafana", "mattermost/mattermost", "microsoft/vscode", "nodejs/node",
            "strapi/strapi", "supabase/supabase", "vercel/next.js", "woocommerce/woocommerce"]
# Features also available in corpus v1 (no commit sizes; no closed PRs there).
FEATURES_V1_COMPAT = [f for f in FEATURES_T0 if f not in SIZE_T0_FEATURES + ["author_merge_rate"]]
FEATURE_SETS = {
    "t0": FEATURES_T0,
    "t0_no_text": [f for f in FEATURES_T0 if f not in TEXT_FEATURES],
    "t0_v1compat": FEATURES_V1_COMPAT,
    # partial-feature baselines (rerun of 02.11.2026)
    "size_only": SIZE_T0_FEATURES + ["num_commits_t0"],
    "author_only": [f for f in FEATURES_T0 if f.startswith("author_")],
}


def check_features(cols):
    """Leak guard: only allow-listed features may reach a model."""
    bad = [c for c in cols if c not in FEATURES_T0 or c in FORBIDDEN]
    if bad:
        raise ValueError(f"features outside the t0 allow-list: {bad}")
    return cols


def make_model(name):
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, verbose=0,
                                  random_seed=SEED, auto_class_weights="Balanced",
                                  thread_count=N_THREADS, allow_writing_files=False)
    if name == "LogReg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0))
    raise KeyError(name)


def _log1p_counts(X):
    """Logistic regression gets log1p of non-negative heavy-tailed columns."""
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.nanmin(col) >= 0 and np.nanmax(col) > 1:
            X[:, j] = np.log1p(col)
    return X


def impute_train_median(Xtr, Xte):
    """Median of the TRAINING rows fills gaps in both (no test information)."""
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for j in range(Xtr.shape[1]):
        Xtr[np.isnan(Xtr[:, j]), j] = med[j]
        Xte[np.isnan(Xte[:, j]), j] = med[j]
    return Xtr, Xte


def metrics(y, prob):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    out = {"n_test": int(len(y)), "pos_test": float(np.mean(y)) if len(y) else math.nan,
           "brier": float(brier_score_loss(y, prob)) if len(y) else math.nan}
    if len(np.unique(y)) == 2:
        out["auc_roc"] = float(roc_auc_score(y, prob))
        out["auc_pr"] = float(average_precision_score(y, prob))
    return out


class Store:
    def __init__(self, out_dir):
        self.out = out_dir
        os.makedirs(os.path.join(out_dir, "oof"), exist_ok=True)
        self.metrics_path = os.path.join(out_dir, "metrics.jsonl")
        self.done = set()
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path) as f:
                for line in f:
                    try:
                        self.done.add(self.key(json.loads(line)))
                    except json.JSONDecodeError:
                        pass   # torn last line after a crash: refit that fold
        self.n_fit = 0
        self.t0 = time.time()

    @staticmethod
    def key(r):
        return (r["variant"], r["target"], r["protocol"], r["model"], str(r["test"]), int(r.get("rep", 0)))

    def has(self, rec):
        return self.key(rec) in self.done

    def put(self, rec, row_ids, prob):
        k = "|".join(map(str, self.key(rec)))
        fn = hashlib.sha1(k.encode()).hexdigest()[:16] + ".npz"
        np.savez_compressed(os.path.join(self.out, "oof", fn), row_id=row_ids.astype(np.int64),
                            prob=prob.astype(np.float32), key=np.array(k))
        rec["oof_file"] = fn
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.done.add(self.key(rec))
        self.n_fit += 1
        if self.n_fit % 20 == 0:
            print(f"    fits={self.n_fit} elapsed={time.time() - self.t0:.0f}s", flush=True)


class Table:
    def __init__(self, path):
        import pandas as pd
        t = pd.read_csv(path, low_memory=False)
        t["row_id"] = np.arange(len(t))
        t["created"] = pd.to_datetime(t["created_at"], utc=True)
        t["merged"] = pd.to_datetime(t["merged_at"], utc=True)
        self.t = t

    def rows(self, pop, clock, target, repos=None, n_min=N_MIN_HISTORY, label=None):
        """Evaluated rows and labels of one configuration."""
        t = self.t
        suf = f"__{pop}_{clock}"
        lab = label or target
        m = (t["hait_filter"] == 1)
        if pop == "main":
            m &= t["in_main"] == 1
        if repos is not None:
            m &= t["repo"].isin(repos)
        y = t[f"y_{lab}{suf}"]
        m &= y.notna()
        if target == "TM":
            m &= t[f"n_hist_tm{suf}"] >= n_min
        if target == "TFR":
            m &= t[f"n_hist_tr{suf}"] >= n_min
        idx = np.where(m.to_numpy())[0]
        return idx, t[f"y_{lab}{suf}"].to_numpy()[idx].astype(int)

    def X(self, idx, cols, clock):
        cols = check_features(list(cols))
        src = [c if (clock == "D" or c not in HIST_CLOCKED) else c + "__A" for c in cols]
        return self.t[src].to_numpy(dtype=float)[idx]


# history features whose value depends on the clock (t_r of earlier PRs)
HIST_CLOCKED = {"author_avg_tfr_human"}


def fit_predict(model, Xtr, ytr, Xte):
    Xtr, Xte = impute_train_median(Xtr, Xte)
    if model == "LogReg":
        Xtr, Xte = _log1p_counts(Xtr), _log1p_counts(Xte)
    m = make_model(model)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def run_config(store, tab, variant, target, model, protocols, pop="main", clock="D",
               feats="t0", repos=None, n_min=N_MIN_HISTORY, label=None, reps=1):
    idx, y = tab.rows(pop, clock, target, repos, n_min, label)
    if len(idx) == 0:
        return
    X = tab.X(idx, FEATURE_SETS[feats], clock)
    repo = tab.t["repo"].to_numpy()[idx]
    lang = tab.t["language"].to_numpy()[idx]
    group = tab.t["group"].to_numpy()[idx]
    row_id = tab.t["row_id"].to_numpy()[idx]
    base = {"variant": variant, "target": target, "label": label or target, "model": model,
            "features": feats, "pop": pop, "clock": clock, "n_min": n_min}
    test_repos = sorted(set(repo))

    def one(protocol, test, tr, te, rep=0, y_tr=None, extra=None):
        rec = dict(base, protocol=protocol, test=test, rep=rep)
        if store.has(rec) or len(te) == 0 or len(tr) == 0:
            return
        ytr = y[tr] if y_tr is None else y_tr
        if len(np.unique(ytr)) < 2:
            return
        prob = fit_predict(model, X[tr], ytr, X[te])
        rec.update(metrics(y[te], prob))
        rec.update({"n_train": int(len(tr)), "pos_train": float(np.mean(ytr))})
        if extra:
            rec.update(extra)
        store.put(rec, row_id[te], prob)

    for protocol in protocols:
        if protocol == "loro":
            for r in test_repos:
                one("loro", r, np.where(repo != r)[0], np.where(repo == r)[0])
        elif protocol == "lolo":
            for r in test_repos:
                lr = lang[repo == r][0]
                one("lolo", r, np.where(lang != lr)[0], np.where(repo == r)[0], extra={"lang": lr})
        elif protocol == "logo":
            for r in test_repos:
                g = group[repo == r][0]
                one("logo", r, np.where(group != g)[0], np.where(repo == r)[0], extra={"group": g})
        elif protocol == "loro_shuffled":
            for rep in range(reps):
                for r in test_repos:
                    tr = np.where(repo != r)[0]
                    rng = np.random.default_rng(SEED + 1000 * rep + test_repos.index(r))
                    one("loro_shuffled", r, tr, np.where(repo == r)[0], rep=rep, y_tr=rng.permutation(y[tr]))
        elif protocol == "lolo_randlang":
            repo_lang = {r: lang[repo == r][0] for r in test_repos}
            for rep in range(reps):
                rng = np.random.default_rng(SEED + 5000 + rep)
                perm = dict(zip(test_repos, rng.permutation([repo_lang[r] for r in test_repos])))
                plang = np.array([perm[r] for r in repo])
                for r in test_repos:
                    one("lolo_randlang", r, np.where(plang != perm[r])[0], np.where(repo == r)[0], rep=rep,
                        extra={"pseudo_lang": perm[r], "true_lang": repo_lang[r]})
        elif protocol == "temporal":
            created = tab.t["created"].iloc[idx].reset_index(drop=True)
            merged = tab.t["merged"].iloc[idx].reset_index(drop=True)
            cut = temporal_cut(tab)
            created, merged = created.to_numpy(), merged.to_numpy()
            tr = np.where((created < cut) & (merged < cut))[0]
            te = np.where(created >= cut)[0]
            rec = dict(base, protocol="temporal", test="pooled", rep=0)
            if not store.has(rec) and len(np.unique(y[tr])) == 2:
                prob = fit_predict(model, X[tr], y[tr], X[te])
                rec.update(metrics(y[te], prob))
                rec.update({"n_train": int(len(tr)), "pos_train": float(np.mean(y[tr])),
                            "cut": str(cut)})
                store.put(rec, row_id[te], prob)
        elif protocol == "loro_late":
            # LORO on the test PRs of the temporal split: train on the other
            # projects' PRs created and merged before the cut, so the only
            # difference from "temporal" is the project's own earlier history
            created = tab.t["created"].to_numpy()[idx]
            merged = tab.t["merged"].to_numpy()[idx]
            cut = temporal_cut(tab)
            early = (created < cut) & (merged < cut)
            late = created >= cut
            for r in test_repos:
                one("loro_late", r, np.where(early & (repo != r))[0], np.where(late & (repo == r))[0],
                    extra={"cut": str(cut)})
        else:
            raise KeyError(protocol)


def temporal_cut(tab):
    """First day of the month that holds the 70 % quantile of creation times of
    the main sample (fixed before the run)."""
    import pandas as pd
    t = tab.t[(tab.t.hait_filter == 1) & (tab.t.in_main == 1)]
    q = t["created"].quantile(TEMPORAL_QUANTILE)
    return pd.Timestamp(year=q.year, month=q.month, day=1, tz="UTC")


def prior_baseline(store, tab, variant="main", pop="main", clock="D"):
    """Feature-free baseline: expanding share of positives of the same project
    (merged before t0). No fit; stored like a model for the report."""
    for target in TARGETS:
        idx, y = tab.rows(pop, clock, target)
        p = tab.t[f"prior_{target}__{pop}_{clock}"].to_numpy(dtype=float)[idx]
        repo = tab.t["repo"].to_numpy()[idx]
        row_id = tab.t["row_id"].to_numpy()[idx]
        for r in sorted(set(repo)):
            te = np.where(repo == r)[0]
            rec = {"variant": variant, "target": target, "label": target, "model": "ProjectPrior",
                   "features": "none", "pop": pop, "clock": clock, "n_min": N_MIN_HISTORY,
                   "protocol": "in_project", "test": r, "rep": 0}
            if store.has(rec):
                continue
            pr = p[te]
            pr = np.where(np.isnan(pr), np.nanmean(pr) if np.isfinite(pr).any() else 0.5, pr)
            rec.update(metrics(y[te], pr))
            store.put(rec, row_id[te], pr)


BLOCKS = ["main", "controls", "ladder", "sensitivity", "temporal", "heuristics"]


def block_main(store, tab):
    for target in TARGETS:
        print(f"== main {target}", flush=True)
        run_config(store, tab, "main", target, "CatBoost", ["loro", "lolo"])
    for target in TARGETS:
        run_config(store, tab, "main", target, "LogReg", ["loro", "lolo"])
    prior_baseline(store, tab)
    for target in TARGETS:
        run_config(store, tab, "main", target, "CatBoost", ["logo"])
        run_config(store, tab, "main", target, "LogReg", ["logo"])


def block_controls(store, tab):
    for target in TARGETS:
        print(f"== controls {target}", flush=True)
        run_config(store, tab, "ctrl", target, "CatBoost", ["loro_shuffled"], reps=SHUFFLE_REPEATS)
    for target in TARGETS:
        run_config(store, tab, "ctrl", target, "CatBoost", ["lolo_randlang"], reps=RANDLANG_REPEATS)


def block_ladder(store, tab):
    """Corpus v2 steps of the contribution table (v1 steps: r3_ladder_v1.py)."""
    for target in TARGETS:
        print(f"== ladder {target}", flush=True)
        run_config(store, tab, "S4a_v2_12repos_A_v1features", target, "CatBoost", ["loro"],
                   clock="A", feats="t0_v1compat", repos=V1_REPOS)
        run_config(store, tab, "S4b_v2_12repos_A", target, "CatBoost", ["loro"], clock="A", repos=V1_REPOS)
        run_config(store, tab, "S4c_v2_40repos_A", target, "CatBoost", ["loro", "lolo"], clock="A")
        run_config(store, tab, "S5_v2_12repos_D", target, "CatBoost", ["loro"], repos=V1_REPOS)


def block_sensitivity(store, tab):
    for target in TARGETS:
        print(f"== sensitivity {target}", flush=True)
        run_config(store, tab, "with_bot_authors", target, "CatBoost", ["loro"], pop="all")
        run_config(store, tab, "no_text_features", target, "CatBoost", ["loro"], feats="t0_no_text")
    for target in ["TM", "TFR"]:
        for n_min in (10, 50):
            run_config(store, tab, f"n_min_{n_min}", target, "CatBoost", ["loro"], n_min=n_min)
        run_config(store, tab, "fixed24", target, "CatBoost", ["loro"], label=f"{target}24")
        run_config(store, tab, "fixed24_A", target, "CatBoost", ["loro"], label=f"{target}24", clock="A")


def block_temporal(store, tab):
    for target in TARGETS:
        for model in ("CatBoost", "LogReg"):
            run_config(store, tab, "main", target, model, ["temporal", "loro_late"])


def block_heuristics(store, tab):
    for target in TARGETS:
        print(f"== heuristics {target}", flush=True)
        for feats in ("size_only", "author_only"):
            run_config(store, tab, feats, target, "CatBoost", ["loro"], feats=feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", choices=BLOCKS, required=True)
    ap.add_argument("--table", default="")
    args = ap.parse_args()
    P = r3_paths()
    table = args.table or os.path.join(P["out"], "pr_table_v2.csv")
    store = Store(os.path.join(P["out"], "models"))
    tab = Table(table)
    t = time.time()
    {"main": block_main, "controls": block_controls, "ladder": block_ladder,
     "sensitivity": block_sensitivity, "temporal": block_temporal,
     "heuristics": block_heuristics}[args.block](store, tab)
    print(f"block {args.block} done: {store.n_fit} new fits, {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
