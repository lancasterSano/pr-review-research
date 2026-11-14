#!/usr/bin/env python3
"""Chapter 3 reporting step on corpus v2 (dissertation, chapter 3: evaluation
thresholds 1–6 and the reporting step).

Reads the night-run outputs (pr_table_v2.csv, models/metrics.jsonl, models/oof)
and the reporting fits of r3_report_models.py; writes ONLY into
<R3_ROOT>/report/ (default revision/output/r3/report/):
  numbers_v2.json      contract keys computed here (source v2 and r4.*)
  status_v2.json       reason for every null
  tables/*.csv         tables of the contract (v2 part)
  details/*.csv        per-repository details behind the numbers

    R3_OUT=revision/output/r3/v2 python revision/r3_report.py
"""
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import r3_report_stats as S  # noqa: E402
from r3_v2_prep import FEATURES_T0, FORBIDDEN, N_MIN_HISTORY, r3_paths  # noqa: E402

B = 10000
TGT = {"it": "IT", "st": "ST", "tr": "TFR", "tm": "TM"}
KS = ["it", "st", "tr"]
CR = {"c01": 0.1, "c02": 0.2, "c03": 0.3}
LANG_ORDER = ["TypeScript", "JavaScript", "Python", "PHP", "Ruby", "Go", "Rust", "Java"]

P = r3_paths()
V2 = P["out"]
R3 = os.path.dirname(V2)
OUT = os.path.join(R3, "report")
TAB_DIR = os.path.join(OUT, "tables")
DET = os.path.join(OUT, "details")
for d in (OUT, TAB_DIR, DET):
    os.makedirs(d, exist_ok=True)

NUM, STATUS = {}, {}


def put(key, value, reason=None):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        NUM[key] = None
        STATUS[key] = reason or "не обчислено"
    else:
        if isinstance(value, (np.floating,)):
            value = float(value)
        if isinstance(value, (np.integer,)):
            value = int(value)
        NUM[key] = value
        if reason:
            STATUS[key] = reason


def short(repo):
    return repo.split("/", 1)[1]


# ── data ────────────────────────────────────────────────────────────────────

def load_table():
    t = pd.read_csv(os.path.join(V2, "pr_table_v2.csv"), low_memory=False)
    t["row_id"] = np.arange(len(t))
    t["created"] = pd.to_datetime(t["created_at"], utc=True)
    return t


T = load_table()
_N10 = os.path.join(OUT, "labels_nmin10.csv")
if os.path.exists(_N10):
    _l = pd.read_csv(_N10).set_index("row_id")
    for _c in ("y_TM_N10__main_D", "y_TFR_N10__main_D"):
        T[_c] = _l[_c].reindex(T.row_id).to_numpy()
_Q75 = os.path.join(OUT, "labels_tfr_q75.csv")
if os.path.exists(_Q75):
    T["y_TFRQ75__main_D"] = pd.read_csv(_Q75).set_index("row_id")["y_TFRQ75__main_D"].reindex(T.row_id).to_numpy()
META = pd.read_csv(os.path.join(os.path.dirname(P["raw_v2"]), "raw_v2", "repos.csv")) \
    if os.path.exists(os.path.join(os.path.dirname(P["raw_v2"]), "raw_v2", "repos.csv")) \
    else pd.read_csv(os.path.join(P["raw_v2"], "repos.csv"))
REPO_LANG = dict(zip(META.repo, META.language))
REPO_GROUP = dict(zip(META.repo, META.group))
SAMPLE = T[(T.hait_filter == 1) & (T.in_main == 1)]


def ycol(label, pop="main", clock="D"):
    return f"y_{label}__{pop}_{clock}"


def load_metrics(path):
    m = pd.read_json(path, lines=True)
    m["oof_dir"] = os.path.join(os.path.dirname(path), "oof")
    return m


MET = load_metrics(os.path.join(V2, "models", "metrics.jsonl"))
REP_MET_PATH = os.path.join(OUT, "models", "metrics.jsonl")
if os.path.exists(REP_MET_PATH):
    MET = pd.concat([MET, load_metrics(REP_MET_PATH)], ignore_index=True)


def preds(variant="main", target="IT", protocol="loro", model="CatBoost", label=None,
          clock="D", pop="main", features=None):
    """Concatenated stored test predictions with labels and repository."""
    q = MET[(MET.variant == variant) & (MET.target == target) & (MET.protocol == protocol)
            & (MET.model == model) & (MET.clock == clock) & (MET["pop"] == pop)]
    q = q[q.label == (label or target)]
    if features:
        q = q[q.features == features]
    parts = []
    for _, r in q.iterrows():
        z = np.load(os.path.join(r.oof_dir, r.oof_file))
        parts.append(pd.DataFrame({"row_id": z["row_id"], "prob": z["prob"].astype(float),
                                   "test": r.test, "rep": int(r.rep)}))
    if not parts:
        return pd.DataFrame(columns=["row_id", "prob", "test", "rep", "repo", "y", "created"])
    d = pd.concat(parts, ignore_index=True)
    lab = T[ycol(label or target, pop, clock)].to_numpy()
    d["y"] = lab[d.row_id.to_numpy()]
    d["repo"] = T["repo"].to_numpy()[d.row_id.to_numpy()]
    d["created"] = T["created"].to_numpy()[d.row_id.to_numpy()]
    return d


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return math.nan
    return float(roc_auc_score(y, p))


def per_repo_auc(d, by="repo"):
    """{repo: AUC} over a prediction frame (rep 0)."""
    d = d[d.rep == 0] if "rep" in d else d
    return {r: auc(g.y, g.prob) for r, g in d.groupby(by)}


def vec(dct, repos):
    return np.array([dct.get(r, math.nan) for r in repos], dtype=float)


# ── 1. corpus and labels ────────────────────────────────────────────────────

def corpus_and_labels():
    hf = T[T.hait_filter == 1]
    put("corpus.n_repos", int(META.repo.nunique()))
    put("corpus.n_languages", int(META.language.nunique()))
    put("corpus.n_merged", int(len(hf)),
        f"після фільтрів HAIT (до них {len(T)} злитих); n_merged − n_bot_author_excluded = n_sample")
    put("corpus.n_bot_author_excluded", int((hf.author_group == "other_bot").sum()),
        f"після фільтрів HAIT (до них {int((T.author_group == 'other_bot').sum())})")
    put("corpus.n_ai_agent_author", int((SAMPLE.author_group == "ai_agent").sum()),
        f"в основній вибірці (після фільтрів HAIT); до фільтрів {int((T.author_group == 'ai_agent').sum())}")
    put("corpus.n_sample", int(len(SAMPLE)))
    for g in ("company", "community"):
        put(f"corpus.groups.{g}.n_repos", int((META.group == g).sum()))
        put(f"corpus.groups.{g}.n_sample", int((SAMPLE.group == g).sum()))
    put("corpus.draft_share_pct", 100 * SAMPLE.created_as_draft.mean())
    dr = SAMPLE[(SAMPLE.created_as_draft == 1) & (SAMPLE.t_rd_fallback == 0)]
    put("corpus.draft_wait_median_h", float(dr.draft_h.median()),
        "серед чернеток з подією виходу з чернетки (без t_rd_fallback)")
    put("labels.n_min_hist", N_MIN_HISTORY)

    rows, repo_rows = [], []
    for k, tg in TGT.items():
        y = SAMPLE[ycol(tg)]
        if tg in ("TM", "TFR"):
            hist = SAMPLE[f"n_hist_{'tm' if tg == 'TM' else 'tr'}__main_D"]
            y = y[hist >= N_MIN_HISTORY]
        y = y.dropna()
        put(f"labels.{k}.n", int(len(y)))
        put(f"labels.{k}.pos_pct", 100 * y.mean())
        big = SAMPLE.repo.value_counts()
        big = set(big[big >= N_MIN_HISTORY].index)
        yy = SAMPLE.loc[y.index]
        by = yy[yy.repo.isin(big)].groupby("repo")[ycol(tg)].mean() * 100
        put(f"labels.{k}.pos_pct_repo_min", float(by.min()),
            "серед репо з ≥ 30 запитами основної вибірки")
        put(f"labels.{k}.pos_pct_repo_max", float(by.max()))
        put(f"labels.{k}.repo_min", short(by.idxmin()))
        put(f"labels.{k}.repo_max", short(by.idxmax()))
        rows.append({"k": k, "n": len(y), "pos_pct": 100 * y.mean(), "min": by.min(), "max": by.max(),
                     "repo_min": short(by.idxmin()), "repo_max": short(by.idxmax())})
    undefined = SAMPLE.t_r_h_D.isna()
    put("labels.tr.n_undefined", int(undefined.sum()))
    put("labels.tr.n_excl_min_hist", int(((~undefined) & (SAMPLE["n_hist_tr__main_D"] < N_MIN_HISTORY)).sum()))
    put("labels.tm.n_excl_min_hist", int((SAMPLE["n_hist_tm__main_D"] < N_MIN_HISTORY).sum()))
    names = {"it": "n_it ≥ 2", "st": "stale = 1", "tr": "t_r довше медіани історії",
             "tm": "t_m довше медіани історії"}
    defs = {"it": "n_it ≥ 2, ітерації за правилом HAIT (формула Р2)",
            "st": "найдовша пауза активності людей від t_rd до злиття > 30 днів (формула Р2)",
            "tr": "t_r від t_rd > медіани t_r запитів проєкту, злитих до t_0 (3.2.1)",
            "tm": "t_m від t_rd > медіани t_m запитів проєкту, злитих до t_0 (3.2.1)"}
    pd.DataFrame([{"мітка": names[r["k"]], "означення (формула Р2 або 3.2.1)": defs[r["k"]],
                   "запитів з міткою": r["n"], "частка міток 1 (%)": round(r["pos_pct"], 1),
                   "найменша і найбільша частка за репозиторіями (%)":
                       f"{r['min']:.1f} ({r['repo_min']}) – {r['max']:.1f} ({r['repo_max']})"}
                  for r in rows]).to_csv(os.path.join(TAB_DIR, "r3_labels.csv"), index=False)

    # per language and per repository
    def lab_share(df, tg):
        y = df[ycol(tg)]
        if tg in ("TM", "TFR"):
            y = y[df[f"n_hist_{'tm' if tg == 'TM' else 'tr'}__main_D"] >= N_MIN_HISTORY]
        y = y.dropna()
        return 100 * y.mean() if len(y) else math.nan

    lang_rows = []
    for lang in LANG_ORDER + ["Разом"]:
        df = SAMPLE if lang == "Разом" else SAMPLE[SAMPLE.language == lang]
        m = META if lang == "Разом" else META[META.language == lang]
        lang_rows.append({"мова": lang,
                          "репозиторіїв (компанія / спільнота)":
                              f"{(m.group == 'company').sum()} / {(m.group == 'community').sum()}",
                          "запитів": len(df),
                          **{f"{c} (%)": round(lab_share(df, tg), 1) for c, tg in
                             (("n_it ≥ 2", "IT"), ("stale", "ST"), ("t_r довгий", "TFR"), ("t_m довгий", "TM"))}})
    pd.DataFrame(lang_rows).to_csv(os.path.join(TAB_DIR, "r3_lang_summary.csv"), index=False)

    rr = []
    for repo in sorted(META.repo, key=lambda r: (LANG_ORDER.index(REPO_LANG[r]), r.lower())):
        df = SAMPLE[SAMPLE.repo == repo]
        rr.append({"репозиторій": repo, "мова": REPO_LANG[repo],
                   "група": "компанія" if REPO_GROUP[repo] == "company" else "спільнота",
                   "запитів": len(df),
                   **{f"{c} (%)": (round(lab_share(df, tg), 1) if len(df) else "")
                      for c, tg in (("n_it ≥ 2", "IT"), ("stale", "ST"), ("t_r довгий", "TFR"),
                                    ("t_m довгий", "TM"))},
                   "чернетки (%)": round(100 * df.created_as_draft.mean(), 1) if len(df) else ""})
    pd.DataFrame(rr).to_csv(os.path.join(TAB_DIR, "r3_repo_labels.csv"), index=False)


# ── 2. features ─────────────────────────────────────────────────────────────

FEATURE_GROUP = {
    "lines_added_t0": "ознаки зміни на t_0", "lines_deleted_t0": "ознаки зміни на t_0",
    "commit_files_t0": "ознаки зміни на t_0", "num_commits_t0": "ознаки зміни на t_0",
    "desc_length": "ознаки зміни на t_0", "title_length": "ознаки зміни на t_0",
    "has_screenshots": "ознаки зміни на t_0", "has_code_blocks": "ознаки зміни на t_0",
    "references_issues": "ознаки зміни на t_0", "is_draft_t0": "ознаки зміни на t_0",
    "day_of_week": "календарні", "hour_of_day": "календарні",
    "author_pr_count": "історія автора до t_0", "author_tenure_days": "історія автора до t_0",
    "author_merge_rate": "історія автора до t_0", "author_avg_review_rounds_h": "історія автора до t_0",
    "author_avg_tfr_human": "історія автора до t_0", "author_commit_frequency_s": "історія автора до t_0",
    "open_pr_count_t0": "стан проєкту до t_0", "merge_velocity_7d_t0": "стан проєкту до t_0",
    "merge_velocity_30d_t0": "стан проєкту до t_0",
}
FEATURE_HOW = {
    "lines_added_t0": "сума доданих рядків комітів з authoredDate ≤ t_0",
    "lines_deleted_t0": "сума вилучених рядків тих самих комітів",
    "commit_files_t0": "сума змінених файлів тих самих комітів (файл у двох комітах двічі)",
    "num_commits_t0": "max(1, кількість комітів з authoredDate ≤ t_0)",
    "desc_length": "довжина опису (кінцевий текст)", "title_length": "довжина назви (кінцевий текст)",
    "has_screenshots": "зображення в описі", "has_code_blocks": "блок коду в описі",
    "references_issues": "closes/fixes/resolves #N в описі",
    "is_draft_t0": "запит створено чернеткою", "day_of_week": "день тижня створення (UTC)",
    "hour_of_day": "година створення (UTC)",
    "author_pr_count": "запити автора в репо, створені до t_0",
    "author_tenure_days": "дні від першого запиту автора в репо до t_0",
    "author_merge_rate": "злиті / (злиті + закриті без злиття) до t_0",
    "author_avg_review_rounds_h": "середнє n_it запитів автора, злитих до t_0 (на момент злиття)",
    "author_avg_tfr_human": "медіана t_r запитів автора, злитих до t_0 (на момент злиття)",
    "author_commit_frequency_s": "коміти запитів автора, злитих до t_0 і створених за 90 днів до t_0",
    "open_pr_count_t0": "запити, створені до t_0 і не закриті до t_0",
    "merge_velocity_7d_t0": "злиття за [t_0 − 7 д, t_0)", "merge_velocity_30d_t0": "злиття за [t_0 − 30 д, t_0)",
}
EXCL_REASON = {
    "lines_added": "кінцевий розмір (відомий лише після t_0)", "lines_deleted": "кінцевий розмір",
    "files_changed": "кінцевий розмір", "num_commits": "кінцева кількість комітів",
    "has_tests": "з кінцевого переліку файлів", "has_docs": "з кінцевого переліку файлів",
    "touches_ci": "з кінцевого переліку файлів", "label_count": "мітки ставлять після t_0",
    "num_file_types": "з кінцевого переліку файлів", "is_draft": "кінцевий стан чернетки",
    "author_avg_review_rounds": "версія HAIT з витоком майбутнього",
    "author_commit_frequency": "версія HAIT з витоком майбутнього",
    "merge_velocity_7d": "версія HAIT з витоком майбутнього",
    "merge_velocity_30d": "версія HAIT з витоком майбутнього",
    "open_pr_count": "версія HAIT з витоком майбутнього",
    "review_count": "події рецензування після t_0", "review_rounds": "події рецензування після t_0",
}


def features():
    put("features.n_allowed", len(FEATURES_T0))
    put("features.n_excluded", len(FORBIDDEN))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", os.path.join(_HERE, "tests", "test_r3_v2.py")],
                       capture_output=True, text=True, cwd=os.path.dirname(_HERE))
    ok = r.returncode == 0
    put("features.leak_test", "пройдено" if ok else "не пройдено")
    open(os.path.join(DET, "leak_test_pytest.txt"), "w").write(r.stdout[-3000:])
    put("features.commits_zero_at_t0_pct", 100 * (SAMPLE.num_commits_t0_raw == 0).mean())
    groups = defaultdict(list)
    for f in FEATURES_T0:
        groups[FEATURE_GROUP[f]].append(f)
    src = {"ознаки зміни на t_0": "коміти з authoredDate ≤ t_0, опис і назва, стан на створенні",
           "історія автора до t_0": "запити автора в репо, створені / злиті / закриті до t_0",
           "стан проєкту до t_0": "запити й злиття репо до t_0",
           "календарні": "момент створення t_0"}
    pd.DataFrame([{"група": g, "кількість ознак": len(groups[g]), "приклади": ", ".join(groups[g][:3]),
                   "з яких подій обчислено": src[g]}
                  for g in ("ознаки зміни на t_0", "історія автора до t_0", "стан проєкту до t_0", "календарні")]
                 ).to_csv(os.path.join(TAB_DIR, "r3_features_groups.csv"), index=False)
    full = [{"ознака": f, "група": FEATURE_GROUP[f], "як обчислено до t_0": FEATURE_HOW[f],
             "дозволена / вилучена": "дозволена", "причина вилучення": ""} for f in FEATURES_T0]
    full += [{"ознака": f, "група": "", "як обчислено до t_0": "", "дозволена / вилучена": "вилучена",
              "причина вилучення": EXCL_REASON.get(f, "")} for f in sorted(FORBIDDEN)]
    pd.DataFrame(full).to_csv(os.path.join(TAB_DIR, "r3_features_full.csv"), index=False)
    return ok


# ── 3. prediction quality ───────────────────────────────────────────────────

RES = {}   # intermediate results shared by later sections


def prediction():
    pred_rows, repo_auc_rows = [], defaultdict(dict)
    for k, tg in TGT.items():
        d_loro = preds("main", tg, "loro")
        d_lolo = preds("main", tg, "lolo")
        d_temp = preds("main", tg, "temporal")
        d_lr = preds("main", tg, "loro", "LogReg")
        d_lr_t = preds("main", tg, "temporal", "LogReg")
        d_prior = preds("main", tg, "in_project", "ProjectPrior")
        a_loro = per_repo_auc(d_loro, "test")
        repos = sorted(a_loro)
        a_lolo = per_repo_auc(d_lolo, "test")
        a_temp = per_repo_auc(d_temp, "repo")
        a_lr = per_repo_auc(d_lr, "test")
        a_lr_t = per_repo_auc(d_lr_t, "repo")
        a_prior = per_repo_auc(d_prior, "test")
        temp_rows = set(d_temp.row_id)
        a_prior_t = per_repo_auc(d_prior[d_prior.row_id.isin(temp_rows)], "repo")
        RES[k] = dict(d_loro=d_loro, a_loro=a_loro, a_lolo=a_lolo, a_temp=a_temp, a_lr=a_lr,
                      a_prior=a_prior, repos=repos, d_prior=d_prior, d_temp=d_temp)
        for proto, a in (("loro", a_loro), ("lolo", a_lolo), ("temporal", a_temp)):
            v = np.array([x for x in a.values() if not math.isnan(x)])
            m, lo, hi = S.cluster_boot_mean(v, B)
            put(f"pred.{k}.{proto}.auc", m)
            put(f"pred.{k}.{proto}.auc_lo", lo)
            put(f"pred.{k}.{proto}.auc_hi", hi)
            RES[k][proto] = (m, lo, hi, len(v))
        v = vec(a_loro, repos)
        put(f"pred.{k}.loro.auc_repo_min", np.nanmin(v))
        put(f"pred.{k}.loro.auc_repo_max", np.nanmax(v))
        RES[k]["repo_min"] = repos[int(np.nanargmin(v))]
        RES[k]["repo_max"] = repos[int(np.nanargmax(v))]
        put(f"pred.{k}.loro.auc_logreg", float(np.nanmean(vec(a_lr, repos))))
        trep = sorted(a_temp)
        put(f"pred.{k}.temporal.auc_logreg", float(np.nanmean(vec(a_lr_t, trep))))
        put(f"pred.{k}.temporal.auc_prior", float(np.nanmean(vec(a_prior_t, trep))))
        g, glo, ghi = S.paired_boot_diff(vec(a_loro, repos), vec(a_lr, repos), B)
        put(f"pred.{k}.loro.gain_logreg", g)
        put(f"pred.{k}.loro.gain_logreg_lo", glo)
        put(f"pred.{k}.loro.gain_logreg_hi", ghi)
        gp = S.paired_boot_diff(vec(a_loro, repos), vec(a_prior, repos), B)
        gpt = S.paired_boot_diff(vec(a_temp, trep), vec(a_prior_t, trep), B)
        glt = S.paired_boot_diff(vec(a_temp, trep), vec(a_lr_t, trep), B)
        RES[k].update(gain_lr=(g, glo, ghi), gain_prior=gp, gain_prior_t=gpt, gain_lr_t=glt)
        # top 10 %
        tops = []
        for r, gdf in d_loro.groupby("test"):
            n = max(1, int(round(0.1 * len(gdf))))
            tops.append(gdf.nlargest(n, "prob").y.mean())
        put(f"pred.{k}.loro.top10_prec_pct", 100 * float(np.mean(tops)))
        RES[k]["top10"] = 100 * float(np.mean(tops))
        RES[k]["base_rate"] = 100 * float(d_loro.groupby("test").y.mean().mean())
        # descriptive: AI-agent authors subgroup
        ai = d_loro[T.author_group.to_numpy()[d_loro.row_id.to_numpy()] == "ai_agent"]
        RES[k]["ai_auc_pooled"] = auc(ai.y, ai.prob) if len(ai) else math.nan
        RES[k]["ai_n"] = len(ai)
        RES[k]["ai_auc_vscode"] = auc(ai[ai.test == "microsoft/vscode"].y, ai[ai.test == "microsoft/vscode"].prob)
        for r in repos:
            repo_auc_rows[r][f"{k}_loro"] = a_loro.get(r)
            repo_auc_rows[r][f"{k}_lolo"] = a_lolo.get(r)
            repo_auc_rows[r][f"{k}_temporal"] = a_temp.get(r)
        # LOGO descriptive
        a_logo = per_repo_auc(preds("main", tg, "logo"), "test")
        RES[k]["logo_minus_loro"] = S.paired_boot_diff(vec(a_logo, repos), vec(a_loro, repos), B)
    # tables
    names = {"it": "n_it ≥ 2", "st": "stale", "tr": "t_r", "tm": "t_m"}
    for k in TGT:
        r = RES[k]
        pred_rows.append({"мітка": names[k], "частка міток 1 (%)": round(NUM[f"labels.{k}.pos_pct"], 1),
                          "LORO": round(r["loro"][0], 3), "LORO_lo": round(r["loro"][1], 3),
                          "LORO_hi": round(r["loro"][2], 3),
                          "LOLO": round(r["lolo"][0], 3), "часове розбиття": round(r["temporal"][0], 3),
                          "логістична регресія (LORO)": round(NUM[f"pred.{k}.loro.auc_logreg"], 3),
                          "апріорна частка проєкту (часове)": round(NUM[f"pred.{k}.temporal.auc_prior"], 3),
                          "точність серед верхніх 10 % (%)": round(r["top10"], 1)})
    pd.DataFrame(pred_rows).to_csv(os.path.join(TAB_DIR, "r3_pred.csv"), index=False)
    out = []
    for repo in sorted(META.repo, key=lambda x: (LANG_ORDER.index(REPO_LANG[x]), x.lower())):
        row = {"репозиторій": repo, "мова": REPO_LANG[repo],
               "група": "компанія" if REPO_GROUP[repo] == "company" else "спільнота"}
        for k in TGT:
            for proto in ("loro", "lolo", "temporal"):
                v = repo_auc_rows.get(repo, {}).get(f"{k}_{proto}")
                row[f"{names[k]} {proto}"] = "" if v is None or (isinstance(v, float) and math.isnan(v)) else round(v, 3)
        out.append(row)
    pd.DataFrame(out).to_csv(os.path.join(TAB_DIR, "r3_repo_auc.csv"), index=False)


# ── 4. transfer: language, groups, D vs A, temporal ─────────────────────────

def language_and_groups():
    lang_tab = {}
    for k, tg in TGT.items():
        r = RES[k]
        repos = r["repos"]
        langs = np.array([REPO_LANG[x] for x in repos])
        loro, lolo = vec(r["a_loro"], repos), vec(r["a_lolo"], repos)
        d = lolo - loro
        ok = ~np.isnan(d)
        md, dlo, dhi = S.grouped_boot_mean(d[ok], langs[ok], B)
        sd = float(np.std(d[ok], ddof=1))
        mde = S.mde_language(sd)
        put(f"transfer.{k}.lolo_minus_loro", md)
        put(f"lang.{k}.mde", mde)
        detected = S.diff_exists(md, dlo, dhi, max(mde, 0.02))
        r["lang_d"] = (md, dlo, dhi, sd, mde, detected)
        tau = S.between_group_sd(loro, langs)
        tlo, thi = S.between_group_sd_boot(loro, langs, 2000)
        put(f"lang.{k}.effect", tau)
        put(f"lang.{k}.effect_lo", tlo)
        put(f"lang.{k}.effect_hi", thi)
        r["lang_tau"] = (tau, tlo, thi)
        by = pd.Series(loro, index=langs).groupby(level=0).mean()
        put(f"lang.{k}.auc_min", float(by.min()))
        put(f"lang.{k}.auc_max", float(by.max()))
        put(f"lang.{k}.lang_min", by.idxmin())
        put(f"lang.{k}.lang_max", by.idxmax())
        lang_tab[k] = by
        # groups
        grp = np.array([REPO_GROUP[x] for x in repos])
        for g in ("company", "community"):
            put(f"group.{g}.{k}.auc_loro", float(np.nanmean(loro[grp == g])))
        dg = S.strat_boot_diff(loro[grp == "company"], loro[grp == "community"], B)
        put(f"group.{k}.diff", dg[0])
        put(f"group.{k}.diff_lo", dg[1])
        put(f"group.{k}.diff_hi", dg[2])
        r["group_diff"] = dg
    with_auc = [x for x in RES["it"]["repos"] if not math.isnan(RES["it"]["a_loro"].get(x, math.nan))]
    cnt = pd.Series([REPO_LANG[x] for x in with_auc]).value_counts()
    put("lang.n_repos_min", int(cnt.min()),
        f"серед {len(with_auc)} репо з AUC LORO")
    put("lang.n_repos_max", int(cnt.max()), f"серед {len(with_auc)} репо з AUC LORO")
    RES["lang_counts_models"] = cnt.to_dict()
    names = {"it": "n_it ≥ 2", "st": "stale", "tr": "t_r", "tm": "t_m"}
    rows = []
    for lang in LANG_ORDER:
        rows.append({"рядок": lang, **{names[k]: round(lang_tab[k].get(lang, math.nan), 3) for k in TGT}})
    rows.append({"рядок": "Середнє", **{names[k]: round(RES[k]["loro"][0], 3) for k in TGT}})
    rows.append({"рядок": "Ефект мови [інтервал]",
                 **{names[k]: f"{RES[k]['lang_tau'][0]:.3f} [{RES[k]['lang_tau'][1]:.3f}; {RES[k]['lang_tau'][2]:.3f}]"
                    for k in TGT}})
    rows.append({"рядок": "Мінімальний виявний ефект", **{names[k]: round(RES[k]["lang_d"][4], 3) for k in TGT}})
    rows.append({"рядок": "LOLO − LORO [інтервал] (довідково)",
                 **{names[k]: f"{RES[k]['lang_d'][0]:.3f} [{RES[k]['lang_d'][1]:.3f}; {RES[k]['lang_d'][2]:.3f}]"
                    for k in TGT}})
    pd.DataFrame(rows).to_csv(os.path.join(TAB_DIR, "r3_lang_auc.csv"), index=False)
    grows = []
    for g, gname in (("company", "веде компанія"), ("community", "спільнотний")):
        reps = [x for x in RES["it"]["repos"] if REPO_GROUP[x] == g
                and not math.isnan(RES["it"]["a_loro"].get(x, math.nan))]
        grows.append({"група": gname, "репозиторіїв": len(reps),
                      "запитів": int((SAMPLE.group == g).sum()),
                      **{names[k]: round(NUM[f"group.{g}.{k}.auc_loro"], 3) for k in TGT}})
    grows.append({"група": "різниця [інтервал]", "репозиторіїв": "", "запитів": "",
                  **{names[k]: f"{RES[k]['group_diff'][0]:.3f} [{RES[k]['group_diff'][1]:.3f}; {RES[k]['group_diff'][2]:.3f}]"
                     for k in TGT}})
    pd.DataFrame(grows).to_csv(os.path.join(TAB_DIR, "r3_groups.csv"), index=False)


def clock_and_temporal():
    for k, tg in TGT.items():
        r = RES[k]
        trep = sorted(r["a_temp"])
        put(f"transfer.{k}.temporal_minus_loro", r["temporal"][0] - r["loro"][0])
        r["temp_minus_loro_same"] = S.paired_boot_diff(vec(r["a_temp"], trep), vec(r["a_loro"], trep), B)
        if k in ("tr", "tm", "st"):
            dA = preds("S4c_v2_40repos_A", tg, "loro", clock="A")
            aA = per_repo_auc(dA, "test")
            mA = float(np.nanmean(vec(aA, r["repos"])))
            put(f"sens.{k}.auc_loro_clock_created", mA)
            yA = SAMPLE[ycol(tg, clock="A")]
            if tg in ("TM", "TFR"):
                yA = yA[SAMPLE[f"n_hist_{'tm' if tg == 'TM' else 'tr'}__main_A"] >= N_MIN_HISTORY]
            pA = 100 * yA.dropna().mean()
            put(f"sens.{k}.pos_pct_clock_created", pA)
            dd = S.paired_boot_diff(vec(r["a_loro"], r["repos"]), vec(aA, r["repos"]), B)
            r["clock"] = {"auc_A": mA, "pos_A": pA, "dAUC_D_minus_A": dd,
                          "dpos_D_minus_A": NUM[f"labels.{k}.pos_pct"] - pA}


# ── 5. calibration ──────────────────────────────────────────────────────────

def calibration():
    rows = []
    for k in TGT:
        d = RES[k]["d_loro"]
        br, bc, er, ec, slopes, ints, early_slopes = [], [], [], [], [], [], []
        for repo, g in d.groupby("test"):
            g = g.sort_values("created")
            h = len(g) // 2
            early, late = g.iloc[:h], g.iloc[h:]
            ab = S.platt_fit(early.prob.to_numpy(), early.y.to_numpy()) if h >= 20 else None
            if ab is None or len(late) == 0:
                continue
            early_slopes.append(ab[1])
            pc = S.platt_apply(late.prob.to_numpy(), ab)
            y = late.y.to_numpy()
            br.append(np.mean((late.prob.to_numpy() - y) ** 2))
            bc.append(np.mean((pc - y) ** 2))
            er.append(S.ece(y, late.prob.to_numpy()))
            ec.append(S.ece(y, pc))
            ab_all = S.platt_fit(g.prob.to_numpy(), g.y.to_numpy())
            if ab_all:
                ints.append(ab_all[0]); slopes.append(ab_all[1])
        vals = dict(brier_raw=np.mean(br), brier_cal=np.mean(bc), ece_raw=np.mean(er), ece_cal=np.mean(ec))
        for key, v in vals.items():
            put(f"pred.{k}.loro.{key}", float(v))
        # the calibration keeps the order of PRs within a project only if every slope is positive
        put(f"pred.{k}.loro.calib_n_slope_nonpos", int(sum(b <= 0 for b in early_slopes)),
            "проєктів, де нахил калібрування на ранішій половині ≤ 0")
        RES[k]["calib"] = dict(vals, n_repos=len(br), slope_median=float(np.median(slopes)),
                               intercept_median=float(np.median(ints)))
        rows.append({"мітка": {"it": "n_it ≥ 2", "st": "stale", "tr": "t_r", "tm": "t_m"}[k],
                     "Брієр до": round(vals["brier_raw"], 3), "Брієр після": round(vals["brier_cal"], 3),
                     "ECE до": round(vals["ece_raw"], 3), "ECE після": round(vals["ece_cal"], 3)})
    pd.DataFrame(rows).to_csv(os.path.join(TAB_DIR, "r3_calib.csv"), index=False)


# ── 6. thresholds (5), transfer (6), R4 share on v2 ─────────────────────────

def rule_thresholds(d, c_r):
    """Per repository: theta on the earlier half, evaluation on the later half."""
    out = {}
    for repo, g in d.groupby("test"):
        g = g.sort_values("created")
        h = len(g) // 2
        early, late = g.iloc[:h], g.iloc[h:]
        th = S.choose_threshold(early.prob.to_numpy(), early.y.to_numpy(), c_r)
        rec = {"n_early": len(early), "n_late": len(late), "theta": th}
        if th is not None and len(late):
            rec.update(S.apply_threshold(late.prob.to_numpy(), late.y.to_numpy(), th))
        out[repo] = rec
    return out


def thresholds_transfer_r4():
    thr = {}
    evalable = None
    for k in KS:
        d = RES[k]["d_loro"]
        thr[k] = {c: rule_thresholds(d, cr) for c, cr in CR.items()}
        ok = {r for r, rec in thr[k]["c01"].items() if rec["n_early"] >= 20 and rec["n_late"] >= 1}
        evalable = ok if evalable is None else evalable & ok
    RES["thr"] = thr
    pd.DataFrame([{"k": k, "c": c, "repo": r, **{kk: vv for kk, vv in rec.items()}}
                  for k in KS for c in CR for r, rec in thr[k][c].items()]).to_csv(
        os.path.join(DET, "thresholds_by_repo.csv"), index=False)
    RES["thr_eval_repos"] = sorted(evalable)
    put("thr.n_projects_eval", len(evalable))
    rows, repo_rows = [], defaultdict(dict)
    label = {"it": "P̂_it (Р1, Р2)", "st": "P̂_st (Р3)", "tr": "P̂_tr (Р4)"}
    for k in KS:
        for c in CR:
            recs = {r: thr[k][c][r] for r in evalable}
            unreach = [r for r, x in recs.items() if x["theta"] is None]
            reach = [x for x in recs.values() if x["theta"] is not None]
            put(f"thr.{k}.{c}.n_unreachable", len(unreach))
            if reach:
                med = (round(float(np.median([x["theta"] for x in reach])), 3),
                       round(float(np.nanmedian([x["precision"] for x in reach])), 3),
                       round(100 * float(np.median([x["fire_share"] for x in reach])), 1))
            else:
                med = ("недосяжно",) * 3
            rows.append({"прогноз і c_r": f"{label[k]}: c_r = {CR[c]:.1f}".replace(".", ","),
                         "медіана порогу θ за репозиторіями": med[0],
                         "медіана досягнутої точності на пізніших запитах": med[1],
                         "медіана частки спрацювань (%)": med[2],
                         "репозиторіїв «недосяжно» з thr.n_projects_eval": f"{len(unreach)} з {len(evalable)}",
                         "репозиторіїв, де точність на пізніших < 1 − c_r":
                             sum(1 for x in reach if x["precision"] == x["precision"] and x["precision"] < 1 - CR[c] - 1e-12),
                         "медіана нижньої межі Вілсона точності":
                             round(float(np.nanmedian([x["prec_lo"] for x in reach])), 3) if reach else "недосяжно"})
            for r in evalable:
                x = recs[r]
                key = f"{k} c_r={CR[c]}"
                if x["theta"] is None:
                    repo_rows[r][f"{key} θ"] = "недосяжно"
                    repo_rows[r][f"{key} точність"] = ""
                    repo_rows[r][f"{key} спрацювань (%)"] = ""
                else:
                    repo_rows[r][f"{key} θ"] = round(x["theta"], 3)
                    repo_rows[r][f"{key} точність"] = (f"{x['precision']:.3f} [{x['prec_lo']:.3f}; {x['prec_hi']:.3f}]"
                                                      if x["n_fire"] else "0 спрацювань")
                    repo_rows[r][f"{key} спрацювань (%)"] = round(100 * x["fire_share"], 1)
    pd.DataFrame(rows).to_csv(os.path.join(TAB_DIR, "r3_thresholds.csv"), index=False)
    pd.DataFrame([{"репозиторій": r, **repo_rows[r]} for r in sorted(repo_rows)]).to_csv(
        os.path.join(TAB_DIR, "r3_repo_thresholds.csv"), index=False)
    # R4 share on v2 (later half, mean over evaluable repos, unreachable = 0)
    for c in CR:
        shares = [(thr["tr"][c][r].get("fire_share", 0.0) if thr["tr"][c][r]["theta"] is not None else 0.0)
                  for r in evalable]
        put(f"r4.{c}.trigger_pct", 100 * float(np.mean(shares)))
    # transfer (6)
    tr_rows = []
    for k, tg in TGT.items():
        d = RES[k]["d_loro"]
        res = {}
        for c, cr in CR.items():
            n_ev = n_suf = 0
            for repo in RES[k]["repos"]:
                other = d[d.test != repo]
                mine = d[d.test == repo]
                x = S.transfer_check(other.prob.to_numpy(), other.y.to_numpy(),
                                     mine.prob.to_numpy(), mine.y.to_numpy(), cr)
                n_ev += x["evaluable"]
                n_suf += x["sufficient"]
                tr_rows.append({"k": k, "c_r": cr, "repo": repo, **x})
            res[c] = (n_suf, n_ev)
            put(f"transfer.{k}.{c}.n_sufficient", n_suf)
            put(f"transfer.{k}.{c}.n_evaluable", n_ev)
        RES[k]["transfer"] = res
        s, e = res["c02"]
        put(f"transfer.{k}.verdict", S.verdict_rule_transfer(s, e),
            "правило вердикту погоджено автором 01.11.2026 (поріг оцінювання 6)")
    pd.DataFrame(tr_rows).to_csv(os.path.join(DET, "transfer_by_repo.csv"), index=False)


# ── 7. controls and sensitivity ─────────────────────────────────────────────

def controls_and_sensitivity():
    put("control.n_shuffle", 3)
    put("sens.signif_diff", 0.02)
    devs, rl = {}, {}
    for k, tg in TGT.items():
        d = preds("ctrl", tg, "loro_shuffled")
        reps = []
        for rep, g in d.groupby("rep"):
            a = per_repo_auc(g.assign(rep=0), "test")
            reps.append(np.nanmean(list(a.values())))
        devs[k] = (float(np.mean(reps)), reps)
        d = preds("ctrl", tg, "lolo_randlang")
        drand = []
        for rep, g in d.groupby("rep"):
            a = per_repo_auc(g.assign(rep=0), "test")
            drand.append(np.nanmean([a[r] - RES[k]["a_loro"][r] for r in a if r in RES[k]["a_loro"]]))
        rl[k] = (float(np.mean(drand)), drand)
    put("control.shuffled_auc_max_dev", max(abs(v[0] - 0.5) for v in devs.values()))
    put("control.random_lang_effect_max", max(abs(v[0]) for v in rl.values()))
    RES["ctrl"] = {"shuffled": devs, "randlang": rl}
    man = os.path.join(R3, "data_manifest_sha256.txt")
    put("control.manifest_sha256", hashlib.sha256(open(man, "rb").read()).hexdigest()[:12])

    # robustness controls: strata of the same LORO predictions
    checks, n_hold = [], 0
    for k, tg in TGT.items():
        d = RES[k]["d_loro"].copy()
        rid = d.row_id.to_numpy()
        size = (T.lines_added_t0.to_numpy(dtype=float) + T.lines_deleted_t0.to_numpy(dtype=float))[rid]
        d["size"] = size
        d["size_t"] = d.groupby("test")["size"].transform(
            lambda s: pd.qcut(s.rank(method="first"), 3, labels=False) if s.notna().sum() >= 3 else np.nan)
        d["half"] = d.groupby("test")["created"].transform(lambda s: (s.rank(method="first") > len(s) / 2).astype(int))
        d["newauth"] = (T.author_pr_count.to_numpy(dtype=float)[rid] == 0).astype(int)
        base_v = S.verdict_transfer(RES[k]["loro"][1])
        for name, col in (("розмір", "size_t"), ("період", "half"), ("автор", "newauth")):
            strata = []
            for s_val, g in d.dropna(subset=[col]).groupby(col):
                a = per_repo_auc(g.assign(rep=0), "test")
                strata.append(np.array([v for v in a.values() if not math.isnan(v)]))
            # mean over strata of the per-repo mean; bootstrap over repos within strata
            means = [x.mean() for x in strata]
            rng = np.random.default_rng(42)
            boots = [np.mean([x[rng.integers(0, len(x), len(x))].mean() for x in strata]) for _ in range(2000)]
            m, lo = float(np.mean(means)), float(np.percentile(boots, 2.5))
            v = S.verdict_transfer(lo)
            hold = v == base_v
            n_hold += hold
            checks.append({"k": k, "контроль": name, "AUC": m, "lo": lo, "вердикт": v, "основний": base_v,
                           "зберігся": int(hold), "страти": [round(float(x), 3) for x in means]})
        gp = RES[k]["gain_prior"]
        base_prior = S.diff_exists(*gp)
        hold = True   # the project control IS the comparison with the project prior (point 2)
        checks.append({"k": k, "контроль": "проєкт", "AUC": gp[0], "lo": gp[1],
                       "вердикт": f"перевага над апріорною часткою: {'є' if base_prior else 'немає'}",
                       "основний": "п. 2", "зберігся": int(base_prior), "страти": []})
        n_hold += int(base_prior)
    put("control.robust_n_checks", len(checks))
    put("control.robust_n_hold", int(n_hold),
        "для «проєкт» висновок «зберігся» = CatBoost має суттєву перевагу (п. 2) над апріорною часткою проєкту")
    pd.DataFrame(checks).to_csv(os.path.join(DET, "robust_controls.csv"), index=False)
    RES["robust"] = checks
    names = {"it": "n_it ≥ 2", "st": "stale", "tr": "t_r", "tm": "t_m"}
    crow = [{"контроль": "перемішані мітки", **{names[k]: round(devs[k][0], 3) for k in TGT},
             "висновок": "у межах [0,48; 0,52]" if all(0.48 <= devs[k][0] <= 0.52 for k in TGT) else "поза [0,48; 0,52]"},
            {"контроль": "випадкова «мова»", **{names[k]: round(rl[k][0], 3) for k in TGT},
             "висновок": "середнє AUC_randlang − AUC_LORO"},
            {"контроль": "тест на витік", **{names[k]: "" for k in TGT}, "висновок": NUM["features.leak_test"]}]
    for name in ("розмір", "період", "автор", "проєкт"):
        sub = [c for c in checks if c["контроль"] == name]
        crow.append({"контроль": f"контроль {name}" if name != "проєкт" else "контроль проєкту",
                     **{names[c["k"]]: round(c["AUC"], 3) for c in sub},
                     "висновок": f"зберігся {sum(c['зберігся'] for c in sub)} з {len(sub)}"})
    pd.DataFrame(crow).to_csv(os.path.join(TAB_DIR, "r3_controls.csv"), index=False)

    # sensitivity table
    def mean_auc(variant, tg, **kw):
        a = per_repo_auc(preds(variant, tg, "loro", **kw), "test")
        return float(np.nanmean(list(a.values()))) if a else math.nan

    srows = []
    have10 = "y_TFR_N10__main_D" in T and len(preds("n_min_10_fixed", "TFR", "loro", label="TFR_N10"))
    for nm in (10, 50):
        if nm == 10 and have10:
            tr_, tm_ = (mean_auc("n_min_10_fixed", "TFR", label="TFR_N10"),
                        mean_auc("n_min_10_fixed", "TM", label="TM_N10"))
            lab = "N_min = 10 (перераховано 31.10: мітки з історією ≥ 10)"
        else:
            tr_, tm_ = mean_auc(f"n_min_{nm}", "TFR"), mean_auc(f"n_min_{nm}", "TM")
            lab = f"N_min = {nm}" + (" (нічний прогін: збігається з основним, мітки лише для історії ≥ 30)" if nm == 10 else "")
        srows.append({"перевірка": lab, "n_it ≥ 2": "", "stale": "", "t_r": round(tr_, 3), "t_m": round(tm_, 3)})
    srows.insert(1, {"перевірка": "N_min = 30 (основний)", "n_it ≥ 2": "", "stale": "",
                     "t_r": round(RES["tr"]["loro"][0], 3), "t_m": round(RES["tm"]["loro"][0], 3)})
    for frac in (0.05, 0.10, 0.20):
        row = {"перевірка": f"точність серед верхніх {int(frac * 100)} % (%)"}
        for k in TGT:
            d = RES[k]["d_loro"]
            row[{"it": "n_it ≥ 2", "st": "stale", "tr": "t_r", "tm": "t_m"}[k]] = round(100 * float(np.mean(
                [g.nlargest(max(1, int(round(frac * len(g)))), "prob").y.mean() for _, g in d.groupby("test")])), 1)
        srows.append(row)
    row = {"перевірка": "відлік від створення (A), AUC LORO", "n_it ≥ 2": "(не залежить)"}
    for k in ("st", "tr", "tm"):
        row[{"st": "stale", "tr": "t_r", "tm": "t_m"}[k]] = round(NUM[f"sens.{k}.auc_loro_clock_created"], 3)
    srows.append(row)
    row = {"перевірка": "відлік від створення (A), частка міток 1 (%)", "n_it ≥ 2": ""}
    for k in ("st", "tr", "tm"):
        row[{"st": "stale", "tr": "t_r", "tm": "t_m"}[k]] = round(NUM[f"sens.{k}.pos_pct_clock_created"], 1)
    srows.append(row)
    srows.append({"перевірка": "коментарі в розмові запиту для застарілості (C)", "n_it ≥ 2": "",
                  "stale": "не рахувалося (потрібен аудит авторів коментарів)", "t_r": "", "t_m": ""})
    extra = []
    for v, kw in (("with_bot_authors", {"pop": "all"}), ("no_text_features", {})):
        extra.append({"перевірка": {"with_bot_authors": "з ботами-авторами", "no_text_features": "без текстових ознак"}[v],
                      **{{"it": "n_it ≥ 2", "st": "stale", "tr": "t_r", "tm": "t_m"}[k]: round(mean_auc(v, tg, **kw), 3)
                         for k, tg in TGT.items()}})
    extra.append({"перевірка": "фіксований поріг 24 год (D)", "n_it ≥ 2": "", "stale": "",
                  "t_r": round(mean_auc("fixed24", "TFR", label="TFR24"), 3),
                  "t_m": round(mean_auc("fixed24", "TM", label="TM24"), 3)})
    extra.append({"перевірка": "фіксований поріг 24 год (A)", "n_it ≥ 2": "", "stale": "",
                  "t_r": round(mean_auc("fixed24_A", "TFR", label="TFR24", clock="A"), 3),
                  "t_m": round(mean_auc("fixed24_A", "TM", label="TM24", clock="A"), 3)})
    # fixed 24 h labels as keys: AUC LORO and the spread of the
    # label share between projects next to the project-relative labels
    big = SAMPLE.repo.value_counts()
    big = set(big[big >= N_MIN_HISTORY].index)
    for k, tg in (("tr", "TFR"), ("tm", "TM")):
        put(f"sens.fixed24.{k}.auc_loro", mean_auc("fixed24", tg, label=f"{tg}24"), "мітка «довше 24 год», відлік D")
        put(f"sens.fixed24_a.{k}.auc_loro", mean_auc("fixed24_A", tg, label=f"{tg}24", clock="A"))
        y = SAMPLE[ycol(f"{tg}24")].dropna()
        put(f"sens.fixed24.{k}.pos_pct", 100 * y.mean())
        by = SAMPLE.loc[y.index].loc[lambda d: d.repo.isin(big)].groupby("repo")[ycol(f"{tg}24")].mean() * 100
        put(f"sens.fixed24.{k}.pos_pct_repo_min", float(by.min()), "серед репо з ≥ 30 запитами основної вибірки")
        put(f"sens.fixed24.{k}.pos_pct_repo_max", float(by.max()))
    # q75 label
    dq = preds("sens_tr_q75", "TFR", "loro", label="TFRQ75")
    if len(dq):
        yq = T["y_TFRQ75__main_D"]
        aq = per_repo_auc(dq, "test")
        mq = S.cluster_boot_mean(np.array([v for v in aq.values() if not math.isnan(v)]), B)
        put("sens.tr_q75.auc_loro", mq[0])
        sub = T.loc[SAMPLE.index]
        yv = sub.loc[sub["n_hist_tr__main_D"] >= N_MIN_HISTORY, "y_TFRQ75__main_D"].dropna()
        put("sens.tr_q75.pos_pct", 100 * float(yv.mean()))
        th = rule_thresholds(dq, CR["c02"])
        shares = [(th[r].get("fire_share", 0.0) if th[r]["theta"] is not None else 0.0)
                  for r in RES["thr_eval_repos"] if r in th]
        put("sens.tr_q75.r4_fire_pct", 100 * float(np.mean(shares)))
        RES["q75"] = {"auc": mq, "n_unreach_c02": sum(1 for r in RES["thr_eval_repos"] if r in th and th[r]["theta"] is None),
                      "n_repos": len(aq)}
        extra.append({"перевірка": "t_r вище верхнього квартиля (Q75)", "t_r": round(mq[0], 3)})
    else:
        for key in ("auc_loro", "pos_pct", "r4_fire_pct"):
            put(f"sens.tr_q75.{key}", None, "моделі Q75 ще не завершено")
    for key, why in (("sens.st.pos_pct_issuecomment", "варіант C не запускався: потрібен ручний аудит авторів ≈ 80 % коментарів у розмові запиту (план, «Не запускається цієї ночі»)"),
                     ("sens.st.auc_loro_issuecomment", "варіант C не запускався (те саме)"),
                     ("sens.issuecomment_audit_share_pct", "аудит авторів коментарів у розмові запиту не проведено"),
                     ("sens.n_mixed_accounts_excluded", "аудит авторів коментарів у розмові запиту не проведено")):
        put(key, None, why)
    pd.DataFrame(srows + extra).to_csv(os.path.join(TAB_DIR, "r3_sens.csv"), index=False)


# ── 8. contribution ladder (v2 part) ────────────────────────────────────────

def contribution():
    itia = os.path.join(R3, "itia_v120_out", "conf_papers", "october_2026", "ITIA2026", "data")
    lad = pd.read_json(os.path.join(R3, "ladder_v1", "ladder-v1-metrics.jsonl"), lines=True)
    steps = []
    per = {}
    for k, tg in TGT.items():
        s0 = pd.read_csv(os.path.join(itia, f"F-per-repo-{tg}.csv")).set_index("repo")["loro"].to_dict()
        per[(k, "S0")] = s0
        for v in ("S1_lists", "S2_no_bot_auth", "S3_t0_features"):
            q = lad[(lad.variant == v) & (lad.target == tg) & (lad.protocol == "loro")]
            per[(k, v)] = dict(zip(q.test, q.auc_roc))
        for v, clock, feats in (("S4a_v2_12repos_A_v1features", "A", None), ("S4b_v2_12repos_A", "A", None),
                                ("S5_v2_12repos_D", "D", None), ("S4c_v2_40repos_A", "A", None)):
            per[(k, v)] = per_repo_auc(preds(v, tg, "loro", clock=clock), "test")
        per[(k, "main_D_40")] = RES[k]["a_loro"]
    order = ["S0", "S1_lists", "S2_no_bot_auth", "S3_t0_features", "S4a_v2_12repos_A_v1features",
             "S4b_v2_12repos_A", "S5_v2_12repos_D", "S4c_v2_40repos_A", "main_D_40"]
    label = {"S0": "відтворення на v1 з налаштуваннями v1.2.0 (ІТІА)",
             "S1_lists": "нові списки облікових записів", "S2_no_bot_auth": "вибірка без ботів-авторів",
             "S3_t0_features": "ознаки на момент подання (наскільки дозволяє v1)",
             "S4a_v2_12repos_A_v1features": "корпус v2, 12 репо, відлік A, ознаки S3",
             "S4b_v2_12repos_A": "корпус v2, 12 репо, відлік A, повний перелік ознак",
             "S5_v2_12repos_D": "відлік від t_rd (12 репо)",
             "S4c_v2_40repos_A": "40 репо, відлік A",
             "main_D_40": "40 репо, відлік від t_rd (основний розрахунок)"}
    prev = {"S1_lists": "S0", "S2_no_bot_auth": "S1_lists", "S3_t0_features": "S2_no_bot_auth",
            "S4a_v2_12repos_A_v1features": "S3_t0_features", "S4b_v2_12repos_A": "S4a_v2_12repos_A_v1features",
            "S5_v2_12repos_D": "S4b_v2_12repos_A", "S4c_v2_40repos_A": "S4b_v2_12repos_A",
            "main_D_40": "S4c_v2_40repos_A"}
    names = {"it": "n_it ≥ 2", "st": "stale", "tr": "t_r", "tm": "t_m"}
    rows, det = [], []
    for st in order:
        row = {"крок": label[st]}
        for k in TGT:
            a = per[(k, st)]
            vals = [v for v in a.values() if v == v]
            m = float(np.mean(vals)) if vals else math.nan
            txt = f"{m:.3f}"
            if st in prev:
                pa = per[(k, prev[st])]
                common = sorted(set(a) & set(pa))
                dd = S.paired_boot_diff([a[r] for r in common], [pa[r] for r in common], B)
                txt += f" (Δ {dd[0]:+.3f} [{dd[1]:+.3f}; {dd[2]:+.3f}], {len(common)} репо)"
                det.append({"k": k, "step": st, "prev": prev[st], "n_common": len(common), "d": dd[0],
                            "lo": dd[1], "hi": dd[2], "contrib": S.diff_exists(*dd)})
            row[f"{names[k]} AUC LORO"] = txt
        row["Hit@3 / MRR ранжування"] = ""
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(TAB_DIR, "r3_contrib.csv"), index=False)
    pd.DataFrame(det).to_csv(os.path.join(DET, "contrib_steps.csv"), index=False)
    RES["contrib"] = det


# ── 9. reverts (descriptive, for chapter 2) ─────────────────────────────────

def reverts():
    ev = T.revert_evidence.dropna().astype(str)
    cnt = defaultdict(int)
    for s in ev:
        for e in s.split("|"):
            if e:
                cnt[e] += 1
    by_repo = []
    for repo, g in T.groupby("repo"):
        row = {"repo": repo, "merged": len(g), "reverted_30d": int(g.reverted_30d.fillna(0).sum())}
        for e in ("body_reverts", "title_revert", "commit_hash"):
            row[e] = int(g.revert_evidence.fillna("").astype(str).str.contains(e).sum())
        by_repo.append(row)
    pd.DataFrame(by_repo).to_csv(os.path.join(DET, "revert_evidence_by_repo.csv"), index=False)
    links = pd.read_csv(os.path.join(V2, "revert_links_v2.csv"))
    RES["reverts"] = {"evidence_counts_targets": dict(cnt), "n_links": len(links),
                      "links_by_evidence": links.evidence.value_counts().to_dict()}


def extra_keys():
    """Keys for numbers the chapter 3 text needs that the other blocks do not produce."""
    put("corpus.n_merged_all", int(len(T)), "усі злиті запити корпусу до фільтрів HAIT")
    put("corpus.n_repos_main_base_fallback", int(T.loc[T.base_rule == "most_frequent", "repo"].nunique())
        if "base_rule" in T else 0, "репо, де основна гілка = найчастіша базова (виправлення 01.11.2026)")
    for k in TGT:
        r = RES[k]
        put(f"pred.{k}.loro.n_repos", int(r["loro"][3]))
        put(f"pred.{k}.loro.verdict", S.verdict_transfer(r["loro"][1]), "п. 1 порогів")
        for name, key in (("gain_lr", "logreg"), ("gain_prior", "prior")):
            g = r[name]
            if name == "gain_prior":
                put(f"pred.{k}.loro.gain_prior", g[0]); put(f"pred.{k}.loro.gain_prior_lo", g[1])
                put(f"pred.{k}.loro.gain_prior_hi", g[2])
            put(f"pred.{k}.loro.gain_{key}_signif", "так" if S.diff_exists(*g, 0.02) else "ні", "п. 2 порогів")
        for name, key in (("gain_lr_t", "logreg"), ("gain_prior_t", "prior")):
            g = r[name]
            put(f"pred.{k}.temporal.gain_{key}", g[0]); put(f"pred.{k}.temporal.gain_{key}_lo", g[1])
            put(f"pred.{k}.temporal.gain_{key}_hi", g[2])
            put(f"pred.{k}.temporal.gain_{key}_signif", "так" if S.diff_exists(*g, 0.02) else "ні", "п. 2 порогів")
        g = r["logo_minus_loro"]
        put(f"transfer.{k}.logo_minus_loro", g[0]); put(f"transfer.{k}.logo_minus_loro_lo", g[1])
        put(f"transfer.{k}.logo_minus_loro_hi", g[2])
        md, dlo, dhi, sd, mde, det = r["lang_d"]
        put(f"transfer.{k}.lolo_minus_loro_lo", dlo); put(f"transfer.{k}.lolo_minus_loro_hi", dhi)
        put(f"lang.{k}.sd_d", sd); put(f"lang.{k}.detected", "так" if det else "ні", "п. 3 порогів")
        g = r["temp_minus_loro_same"]
        put(f"transfer.{k}.temporal_minus_loro_lo", g[1], "парний інтервал на репо з часовим розбиттям")
        put(f"transfer.{k}.temporal_minus_loro_hi", g[2], "парний інтервал на репо з часовим розбиттям")
        c = r["calib"]
        put(f"pred.{k}.loro.calib_slope_median", c["slope_median"], "медіана нахилу Платта за репо (усі запити репо)")
        put(f"pred.{k}.loro.calib_intercept_median", c["intercept_median"])
        put(f"ai.{k}.n", int(r["ai_n"]))
        put(f"ai.{k}.auc", r["ai_auc_pooled"], "об'єднано за репо, описово")
        put(f"ai.{k}.auc_vscode", r["ai_auc_vscode"], "лише microsoft/vscode, описово")
        if "clock" in r:
            dd = r["clock"]["dAUC_D_minus_A"]
            put(f"sens.{k}.dauc_clock_d_minus_a", dd[0]); put(f"sens.{k}.dauc_clock_d_minus_a_lo", dd[1])
            put(f"sens.{k}.dauc_clock_d_minus_a_hi", dd[2])
            put(f"sens.{k}.dpos_clock_d_minus_a", r["clock"]["dpos_D_minus_A"], "п. п.")
            put(f"sens.{k}.clock_signif", "так" if abs(dd[0]) >= 0.02
                or abs(r["clock"]["dpos_D_minus_A"]) >= 5 else "ні", "п. 4 порогів")
    for k in KS:
        for c, cr in CR.items():
            reach = [RES["thr"][k][c][x] for x in RES["thr_eval_repos"] if RES["thr"][k][c][x]["theta"] is not None]
            put(f"thr.{k}.{c}.n_below_target", sum(1 for x in reach if x["precision"] == x["precision"]
                                                    and x["precision"] < 1 - cr - 1e-12))
            put(f"thr.{k}.{c}.prec_lo_median", float(np.nanmedian([x["prec_lo"] for x in reach])) if reach else None,
                None if reach else "недосяжно на всіх оцінюваних репо")
    q = RES.get("q75")
    if q:
        put("sens.tr_q75.auc_loro_lo", q["auc"][1]); put("sens.tr_q75.auc_loro_hi", q["auc"][2])
        put("sens.tr_q75.n_unreachable_c02", int(q["n_unreach_c02"])); put("sens.tr_q75.n_repos", int(q["n_repos"]))
    rv = RES["reverts"]
    put("rev.n_links", int(rv["n_links"]))
    for e in ("body_reverts", "title_revert", "commit_hash"):
        put(f"rev.links_{e}", int(sum(v for ks, v in rv["links_by_evidence"].items() if e in str(ks).split("|"))),
            "зв'язки з цією ознакою (один зв'язок може мати кілька ознак)")
    put("rev.n_reverted_30d_sample", int(SAMPLE.reverted_30d.fillna(0).sum()), "в основній вибірці, описово")


def rerun_0211_keys():
    """Keys of the rerun of 02.11.2026 (revision/r3_rerun_0211.sh):
    reviews of a draft, LORO on the temporal test PRs, partial-feature baselines.
    Skipped quietly when the predictions are absent (results of an earlier run)."""
    if "first_review_in_draft" in SAMPLE:
        put("corpus.n_first_review_in_draft", int(SAMPLE.first_review_in_draft.fillna(0).sum()),
            "перша рецензія людини подана в чернетці (до t_rd); не входить у характеристики")
    for k, tg in TGT.items():
        r = RES[k]
        a_late = per_repo_auc(preds("main", tg, "loro_late"), "test")
        if a_late:
            trep = sorted(set(a_late) & set(r["a_temp"]))
            v = vec(a_late, trep)
            put(f"transfer.{k}.loro_late.auc", float(np.nanmean(v)), "LORO на тестових запитах часового розбиття")
            g = S.paired_boot_diff(vec(r["a_temp"], trep), v, B)
            put(f"transfer.{k}.temporal_minus_loro_late", g[0])
            put(f"transfer.{k}.temporal_minus_loro_late_lo", g[1])
            put(f"transfer.{k}.temporal_minus_loro_late_hi", g[2])
            put(f"transfer.{k}.temporal_minus_loro_late_signif", "так" if S.diff_exists(*g, 0.02) else "ні",
                "п. 2 порогів, ті самі репо і запити")
        for feats in ("size_only", "author_only"):
            a = per_repo_auc(preds(feats, tg, "loro"), "test")
            if not a:
                continue
            repos = r["repos"]
            m = S.cluster_boot_mean(vec(a, repos), B)
            put(f"pred.{k}.{feats}.auc", m[0]); put(f"pred.{k}.{feats}.auc_lo", m[1])
            put(f"pred.{k}.{feats}.auc_hi", m[2])
            g = S.paired_boot_diff(vec(r["a_loro"], repos), vec(a, repos), B)
            put(f"pred.{k}.gain_{feats}", g[0]); put(f"pred.{k}.gain_{feats}_lo", g[1])
            put(f"pred.{k}.gain_{feats}_hi", g[2])
            put(f"pred.{k}.gain_{feats}_signif", "так" if S.diff_exists(*g, 0.02) else "ні", "п. 2 порогів")


def main():
    corpus_and_labels()
    features()
    prediction()
    language_and_groups()
    clock_and_temporal()
    calibration()
    thresholds_transfer_r4()
    controls_and_sensitivity()
    contribution()
    reverts()
    extra_keys()
    rerun_0211_keys()
    json.dump(NUM, open(os.path.join(OUT, "numbers_v2.json"), "w"), ensure_ascii=False, indent=1)
    json.dump(STATUS, open(os.path.join(OUT, "status_v2.json"), "w"), ensure_ascii=False, indent=1)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items() if not str(k).startswith(("d_", "a_"))}
        if isinstance(o, (list, tuple)):
            return [clean(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (pd.DataFrame, pd.Series)):
            return None
        return o
    json.dump(clean(RES), open(os.path.join(OUT, "details", "results_detail.json"), "w"),
              ensure_ascii=False, indent=1, default=str)
    print(f"numbers: {len(NUM)} keys, {sum(v is None for v in NUM.values())} null")


if __name__ == "__main__":
    main()
