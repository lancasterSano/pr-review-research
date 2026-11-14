#!/usr/bin/env python3
"""Chapter 2 numbers and tables on corpus v2 (40 repositories) -> r2_numbers.json.

Inputs (read only):
  - $R3_ROOT/v2/pr_table_v2.csv (the PR table shared by chapters 2 and 3,
    revision/r3_v2_prep.py; default revision/output/r3_b40d): characteristics
    over human events, clock D from t_rd, author groups, language and project
    group, reverts within 30 days;
  - $R3_ROOT/v2/revert_links_v2.csv: "PR - reverting PR" links (for the 7 and
    90 day windows, by the same rule as reverted_30d in
    r3_v2_prep.match_reverts);
  - $R3_RAW_V2/repos.csv: corpus v2 repository list (default data/raw_v2);
  - $R2_DIR/v2/*.csv: output of r2_raw_v2.py (quantities the table lacks);
  - $R2_DIR/v2/r2_alignment_v2.json: output of r2_alignment_v2.py;
  - $R2_LOGIN_AUDIT/logins.csv: output of revision/taskV2_login_audit.py
    (default revision/output/corpus_v2/login_audit);
  - the account lists of revision/taskC_common.py before the login audit, read
    with `git show v1.2.0:revision/taskC_common.py` (needs a git clone with the
    v1.2.0 tag).

Main sample: in_main = 1 and hait_filter = 1 (as in chapter 3). The "total"
row of a table equals the sum of the rows shown; a repository with fewer than
N_MIN = 30 main-sample PRs is not shown in the characteristics table (the
criterion is stated in the text).

Output: $R2_DIR/r2_numbers.json (the chapter 2 numbers). R2_DIR defaults to
revision/output/r2 (git-ignored).

Run from the repository root after r2_raw_v2.py and r2_alignment_v2.py:
  PYTHONDONTWRITEBYTECODE=1 nice .venv/bin/python revision/r2/make_tables_r2.py

Importing this module has no side effects beyond path constants (and adding
this folder to sys.path for r2_ai_compare_v2).
"""
import ast
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)                      # revision/
ROOT = os.path.dirname(REV)                      # repository root
# chapter 3 run r3_b40d: the PR table without human reviews submitted in draft (before t_rd)
R3_ROOT = os.environ.get("R3_ROOT") or os.path.join(REV, "output", "r3_b40d")
R3_RUN = os.path.basename(os.path.normpath(R3_ROOT))
V2 = os.path.join(R3_ROOT, "v2")
RAW_V2 = os.environ.get("R3_RAW_V2") or os.path.join(ROOT, "data", "raw_v2")
R2_DIR = os.environ.get("R2_DIR") or os.path.join(REV, "output", "r2")
EXTRA = os.path.join(R2_DIR, "v2")
sys.path.insert(0, HERE)
from r2_ai_compare_v2 import compare  # noqa: E402

N_MIN = 30
WINDOW_END = pd.Timestamp("2026-08-31T23:59:59Z")
DELTAS = [15, 30, 45, 60, 90]
LANG_ORDER = ["JavaScript", "TypeScript", "Python", "Go", "Ruby", "PHP", "Java", "Rust"]
LANG_UA = {"JavaScript": "JavaScript", "TypeScript": "TypeScript", "Python": "Python", "Go": "Go",
           "Ruby": "Ruby", "PHP": "PHP", "Java": "Java", "Rust": "Rust"}
GROUP_UA = {"company": "компанія", "community": "спільнота"}
REACT_MIN = 100
# Corpus v2 login audit (29.10.2026, revision/taskV2_login_audit.py). audit_numbers()
# counts the logins and how much the audit added to each account list, from the audit's
# logins.csv and the lists as they were before the audit (taskC_common.py at tag v1.2.0).
# The number of logins flagged for checking (289) is a constant of the run with the
# pre-audit lists: logins.csv was written with the extended lists already, and the
# pattern flag of logins that went into the lists cannot be recovered from it.
AUDIT_DIR = os.environ.get("R2_LOGIN_AUDIT") or os.path.join(REV, "output", "corpus_v2", "login_audit")
AUDIT_REF = "v1.2.0"   # taskC_common.py before the audit (same file as the audit commit's parent)
AUDIT_N_FLAGGED = 289


def lists_before_audit():
    """taskC_common account lists before the audit: class -> set of account keys."""
    src = subprocess.run(
        ["git", "-C", ROOT, "show", f"{AUDIT_REF}:revision/taskC_common.py"],
        check=True, capture_output=True, text=True).stdout
    names = {"AI_ASSISTANTS": "ai_assistant", "NON_AI_BOTS": "other_bot", "UNCERTAIN": "uncertain"}
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in names:
            out[names[node.targets[0].id]] = set(ast.literal_eval(node.value))
    return out


def audit_numbers():
    """Corpus logins and how much the audit added to each list (corpus logins only)."""
    logins = pd.read_csv(os.path.join(AUDIT_DIR, "logins.csv"), usecols=["login", "class"],
                         keep_default_na=False)
    before = lists_before_audit()
    key = logins["login"].str.lower().str.replace(r"\[bot\]$", "", regex=True)

    def added(cls):
        return int(((logins["class"] == cls) & ~key.isin(before[cls])).sum())
    return {"n_logins": int(len(logins)), "n_flagged": AUDIT_N_FLAGGED,
            "added_ai": added("ai_assistant"), "added_bots": added("other_bot"),
            "uncertain": added("uncertain")}


def pct(x):
    x = pd.Series(x)
    return float(100.0 * x.mean()) if len(x) else None


def med(x):
    x = pd.Series(x).dropna()
    return float(x.median()) if len(x) else None


def short(repo):
    return repo.split("/", 1)[1]


def load():
    cols = ["repo", "pr_number", "author_login", "author_group", "in_main", "hait_filter",
            "created_at", "merged_at", "created_as_draft", "t_rd_fallback", "t_m_h_D", "t_r_h_D",
            "t_r_h_D_raw", "n_it_human", "gap_human_days_D", "stale30_D", "final_additions",
            "final_changed_files", "language", "group", "is_revert_pr", "reverted_30d",
            "revert_censored", "base_ref"]
    t = pd.read_csv(os.path.join(V2, "pr_table_v2.csv"), usecols=cols, low_memory=False)
    t["created"] = pd.to_datetime(t.created_at, utc=True)
    t["merged"] = pd.to_datetime(t.merged_at, utc=True)
    x = pd.read_csv(os.path.join(EXTRA, "r2_pr_extra_v2.csv"), low_memory=False)
    main = t[(t.in_main == 1) & (t.hait_filter == 1)].merge(x, on=["repo", "pr_number"], how="inner")
    assert len(main) == int(((t.in_main == 1) & (t.hait_filter == 1)).sum())
    return t, main


def corpus_numbers(t, main):
    allp = pd.read_csv(os.path.join(EXTRA, "r2_all_prs_v2.csv"))
    h = t[t.hait_filter == 1]
    meta = pd.read_csv(os.path.join(RAW_V2, "repos.csv"))
    per_repo = main.groupby("repo").size()
    small = sorted(r for r in meta.repo if per_repo.get(r, 0) < N_MIN)
    return {
        "n_repos": int(len(meta)),
        "n_prs_all": int(len(allp)),
        "n_merged_all": int((allp.state == "MERGED").sum()),
        "n_closed_all": int((allp.state == "CLOSED").sum()),
        "n_open_all": int((allp.state == "OPEN").sum()),
        "n_merged_table": int(len(t)),
        "n_merged_hait": int(len(h)),
        "hait_pct": 100.0 * len(h) / len(t),
        "n_main": int(len(main)),
        "n_main_human": int((main.author_group == "human").sum()),
        "n_main_agent": int((main.author_group == "ai_agent").sum()),
        "n_bot_author": int((h.author_group == "other_bot").sum()),
        "bot_author_pct": 100.0 * (h.author_group == "other_bot").mean(),
        "agent_author_pct": 100.0 * (h.author_group == "ai_agent").mean(),
        "n_lang": int(meta.language.nunique()),
        "n_company": int((meta.group == "company").sum()),
        "n_community": int((meta.group == "community").sum()),
        "n_repos_table": int(len(meta) - len(small)),
        "small_repos": ", ".join(short(r) for r in small),
        "small_repos_n_main": ", ".join(str(int(per_repo.get(r, 0))) for r in small),
        "commits_no_user_pct": 100.0 * main.n_commits_no_user.sum() / main.n_commits.sum(),
        "ci_window_pct": pct(main.ci_window),
        "main_repo_min": int(per_repo[per_repo >= N_MIN].min()),
        "main_repo_min_name": short(per_repo[per_repo >= N_MIN].idxmin()),
        "main_repo_max": int(per_repo.max()),
        "main_repo_max_name": short(per_repo.idxmax()),
    }


def draft_numbers(main):
    by = main.groupby("repo").created_as_draft.mean().mul(100).sort_values(ascending=False)
    return {"exit_pct": pct(main.created_as_draft == 1),
            "fallback_n": int(main.t_rd_fallback.sum()),
            "top": [{"repo": short(r), "pct": float(v)} for r, v in by.head(3).items()],
            "tr_raw_neg_n": int((main.t_r_h_D_raw < 0).sum())}


def replay_numbers(main):
    rep = main.groupby("repo").accepted.agg(["mean", "size", "sum"])
    rep = rep[rep["size"] >= N_MIN]
    rep["pct"] = 100 * rep["mean"]
    worst = rep.sort_values("pct").head(2)
    stale = main[main.gap_h_days_D > 30]
    cats = main.reject_category.fillna("").value_counts()
    table = [{"repo": r, "short": short(r), "n": int(v["size"]), "pct": float(v.pct),
              "rej": int(v["size"] - v["sum"])} for r, v in rep.sort_values("pct").iterrows()]
    m30 = main[main.repo.isin(rep.index)]
    total = {"n": int(len(m30)), "pct": pct(m30.accepted), "rej": int((m30.accepted == 0).sum())}
    return {
        "table": table, "total": total,
        "n": int(len(main)), "accepted_n": int(main.accepted.sum()),
        "accepted_pct": pct(main.accepted), "rejected_n": int((main.accepted == 0).sum()),
        "rejected_pct": 100 - pct(main.accepted),
        "worst": [{"repo": short(r), "pct": float(v)} for r, v in worst.pct.items()],
        "n_ge999": int((rep.pct > 99.9).sum()),
        "cat": {k: int(cats.get(k, 0)) for k in ("comment_or_changes_only", "no_human_review",
                                                 "bot_approval", "approval_dismissed",
                                                 "approval_after_merge")},
        "stale_traces_n": int(len(stale)), "stale_accepted_pct": pct(stale.accepted),
        "first_verdict_approve_n": int((main.first_h_verdict == "APPROVED").sum()),
        "first_verdict_approve_pct": pct(main.first_h_verdict == "APPROVED"),
        "bot_approval_top": short(main[main.reject_category == "bot_approval"].repo.value_counts().idxmax()),
        "bot_approval_top_n": int(main[main.reject_category == "bot_approval"].repo.value_counts().max()),
    }


def duration_numbers(main):
    vs = main[main.repo == "microsoft/vscode"]
    after = main[main.created >= pd.Timestamp("2025-02-01", tz="UTC")]
    first_auto = main.first_reviewer_class.isin(["ai_assistant", "other_bot", "uncertain"])
    tr_def = main[main.t_r_h_D.notna()]
    return {
        "tr_undef_n": int(main.t_r_h_D.isna().sum()),
        "tr_undef_pct": pct(main.t_r_h_D.isna()),
        "first_auto_pct": pct(first_auto),
        "first_ai_pct": pct(main.first_reviewer_class == "ai_assistant"),
        "first_ai_pct_2502": pct(after.first_reviewer_class == "ai_assistant"),
        "first_ai_pct_vscode": pct(vs.first_reviewer_class == "ai_assistant"),
        "tr_med_h": med(tr_def.t_r_h_D), "tr_med_all": med(tr_def.t_r_all_h_D),
        "tr_med_h_vscode": med(vs.t_r_h_D), "tr_med_all_vscode": med(vs.loc[vs.t_r_h_D.notna(), "t_r_all_h_D"]),
        "t_ai_med": med(main.loc[main.u_ai == 1, "t_ai_h_D"]),
        "tm_med": med(main.t_m_h_D),
    }


def iteration_numbers(main):
    it2 = main.n_it_human >= 2
    bnd = main.n_it_h_code > main.n_it_human
    lab_all = main.n_it_all_lit >= 2
    chg = lab_all != it2
    after = main.created >= pd.Timestamp("2025-02-01", tz="UTC")
    vs = main[main.repo == "microsoft/vscode"]
    return {
        "it2_n": int(it2.sum()), "it2_pct": pct(it2),
        "it2_cr_n": int((it2 & (main.has_h_changes_requested == 1)).sum()),
        "it2_cr_pct": 100.0 * (it2 & (main.has_h_changes_requested == 1)).sum() / it2.sum(),
        "it2_ret_n": int((it2 & (main.visits_returned == 1)).sum()),
        "it2_ret_pct": 100.0 * (it2 & (main.visits_returned == 1)).sum() / it2.sum(),
        "cr_n": int((main.has_h_changes_requested == 1).sum()),
        "cr_pct": pct(main.has_h_changes_requested == 1),
        "boundary_n": int(bnd.sum()), "boundary_pct": pct(bnd),
        "boundary_after_appr_n": int((bnd & (main.last_h_verdict == "APPROVED")).sum()),
        "it2_code_pct": pct(main.n_it_h_code >= 2),
        "it2_all_pct": pct(lab_all),
        "label_changed_n": int(chg.sum()), "label_changed_pct": pct(chg),
        "label_changed_after_2502_n": int((chg & after).sum()),
        "vscode_it2_all_pct": pct(vs.n_it_all_lit >= 2), "vscode_it2_h_pct": pct(vs.n_it_human >= 2),
    }


def stale_numbers(main):
    g_h, g_a = main.gap_h_days_D, main.gap_all_days_D
    longer = g_h > g_a + 1e-6
    st = main[main.stale30_D == 1]
    return {
        "stale_n": int((main.stale30_D == 1).sum()), "stale_pct": pct(main.stale30_D == 1),
        "after_appr_n": int(st.gap_h_start_after_appr.sum()),
        "after_appr_pct": pct(st.gap_h_start_after_appr),
        "longer_n": int(longer.sum()), "longer_pct": pct(longer),
        "longer_med_min": float(((g_h - g_a)[longer]).median() * 24 * 60),
        "label_changed_n": int(((g_h > 30) != (g_a > 30)).sum()),
        "stale_all_pct": pct(g_a > 30),
    }


def delta_numbers(main):
    rows = []
    per = main.groupby("repo")
    for d in DELTAS:
        s = main.gap_human_days_D > d
        cnt = (main.gap_human_days_D > d).groupby(main.repo).sum()
        cnt = cnt[per.size() >= N_MIN]
        rows.append({"delta": d, "n": int(s.sum()), "pct": pct(s),
                     "min_repo_n": int(cnt.min()), "min_repo": short(cnt.idxmin()),
                     "repos_lt10": int((cnt < 10).sum()), "repos_zero": int((cnt == 0).sum())})
    return rows


def char_row(g, name):
    tr = g.t_r_h_D.dropna()
    return {"n": int(len(g)), "it2_pct": pct(g.n_it_human >= 2),
            "it2_all_pct": pct(g.n_it_all_lit >= 2),
            "tr_med": float(tr.median()) if len(tr) else None,
            "tr_all_med": med(g.loc[g.t_r_h_D.notna(), "t_r_all_h_D"]),
            "tm_med": med(g.t_m_h_D), "stale_pct": pct(g.stale30_D == 1),
            "ai_pct": pct(g.u_ai), "draft_pct": pct(g.created_as_draft == 1), "name": name}


def repo_tables(main):
    per = main.groupby("repo")
    keep = [r for r, g in per if len(g) >= N_MIN]
    m = main[main.repo.isin(keep)]
    rt = []
    for r in sorted(keep, key=lambda r: (LANG_ORDER.index(m[m.repo == r].language.iloc[0]), r.lower())):
        g = m[m.repo == r]
        row = char_row(g, short(r))
        tools = g.ai_tools.dropna().str.split("|").explode()
        row.update({"repo": r, "short": short(r), "lang": g.language.iloc[0],
                    "group": GROUP_UA[g.group.iloc[0]],
                    "ai_n": int(g.u_ai.sum()),
                    "ai_top": tools.value_counts().idxmax() if len(tools) else "–"})
        rt.append(row)
    tot = char_row(m, "Разом")
    tot.update({"repo": "Разом", "ai_n": int(m.u_ai.sum())})
    lt = []
    for lang in LANG_ORDER:
        g = m[m.language == lang]
        row = char_row(g, LANG_UA[lang])
        row.update({"lang": lang, "n_repos": int(g.repo.nunique())})
        lt.append(row)
    gt = []
    for grp in ("company", "community"):
        g = m[m.group == grp]
        row = char_row(g, GROUP_UA[grp])
        row.update({"group": grp, "n_repos": int(g.repo.nunique())})
        gt.append(row)
    tot["n_repos"] = int(m.repo.nunique())
    ext = {}
    for key, col, lo_better in (("it2", "it2_pct", True), ("tr", "tr_med", True),
                                ("tm", "tm_med", True), ("stale", "stale_pct", True)):
        s = pd.Series({r["short"]: r[col] for r in rt})
        ext[key] = {"min": float(s.min()), "min_repo": s.idxmin(), "max": float(s.max()),
                    "max_repo": s.idxmax()}
    spread = {}
    for lang in LANG_ORDER:
        v = [r["it2_pct"] for r in rt if r["lang"] == lang]
        spread[lang] = max(v) - min(v)
    ext["it2_within_lang_max_spread"] = float(max(spread.values()))
    ext["it2_within_lang_max_spread_lang"] = max(spread, key=spread.get)
    ext["it2_between_lang_spread"] = float(max(r["it2_pct"] for r in lt) - min(r["it2_pct"] for r in lt))
    for key, rows_ in (("lang", lt), ("group", gt)):
        for col in ("it2_pct", "tr_med", "tm_med", "stale_pct"):
            s = pd.Series({r["name"]: r[col] for r in rows_})
            ext[f"{key}_{col}"] = {"min": float(s.min()), "min_name": s.idxmin(),
                                   "max": float(s.max()), "max_name": s.idxmax()}
    return rt, tot, lt, gt, ext


def revert_numbers(t, main):
    L = pd.read_csv(os.path.join(V2, "revert_links_v2.csv"))
    for c in ("target_merged_at", "reverter_merged_at"):
        L[c] = pd.to_datetime(L[c], utc=True)
    rb = defaultdict(list)
    for r in L.itertuples():
        if r.reverter_merged_at >= r.target_merged_at:
            rb[(r.repo, r.target)].append((r.reverter_merged_at, r.reverter))

    def reverted(row, d):
        c = [x for x in rb.get((row.repo, row.pr_number), [])
             if x[0] - row.merged <= pd.Timedelta(days=d)]
        if not c:
            return 0
        best = min(c)
        return 0 if rb.get((row.repo, best[1])) else 1

    base = main[main.is_revert_pr == 0]
    den30 = base[base.revert_censored == 0]
    # check that the reverted_30d column is reproduced
    chk = np.array([reverted(r, 30) for r in den30.itertuples()])
    assert (chk == den30.reverted_30d.to_numpy()).all()
    per = den30.groupby("repo").reverted_30d.agg(["mean", "size", "sum"])
    per = per[per["size"] >= N_MIN]
    per["pct"] = 100 * per["mean"]
    common = base[base.merged <= WINDOW_END - pd.Timedelta(days=90)]
    w = {d: np.array([reverted(r, d) for r in common.itertuples()]) for d in (7, 30, 90)}
    return {
        "n_revert_prs": int(main.is_revert_pr.sum()),
        "n_censored": int((base.revert_censored == 1).sum()),
        "den": int(len(den30)), "n": int(den30.reverted_30d.sum()), "pct": pct(den30.reverted_30d),
        "repo_min": float(per.pct.min()), "repo_min_name": short(per.pct.idxmin()),
        "repo_max": float(per.pct.max()), "repo_max_name": short(per.pct.idxmax()),
        "repo_median": float(per.pct.median()),
        "common_den": int(len(common)),
        "pct7": float(100 * w[7].mean()), "pct30c": float(100 * w[30].mean()),
        "pct90": float(100 * w[90].mean()),
        "share7_of90": float(100 * w[7].sum() / w[90].sum()),
        "share30_of90": float(100 * w[30].sum() / w[90].sum()),
        "per_repo": [{"repo": r, "short": short(r), "pct": float(v.pct), "n": int(v["sum"]),
                      "den": int(v["size"])} for r, v in per.iterrows()],
    }


def fr_numbers(main):
    """Share of PRs for which fewer than N_MIN main-sample PRs were merged in the project before t0."""
    small = 0
    for repo, g in main.groupby("repo"):
        mt = np.sort(g.merged.to_numpy())
        before = np.searchsorted(mt, g.created.to_numpy(), side="left")
        small += int((before < N_MIN).sum())
    return {"n_short_hist": small, "pct_short_hist": 100.0 * small / len(main)}


def ai_numbers(main):
    allp = pd.read_csv(os.path.join(EXTRA, "r2_all_prs_v2.csv"))
    allp["year"] = pd.to_datetime(allp.created_at, utc=True).dt.year
    by_year = allp.groupby("year").u_ai.mean().mul(100)
    tools = main.ai_tools.dropna().str.split("|").explode()
    tool_n = tools.value_counts()
    tool_repos = main.assign(t=main.ai_tools.str.split("|")).explode("t").dropna(subset=["t"]) \
        .groupby("t").repo.nunique()
    ag = main[main.author_group == "ai_agent"]
    per = main.groupby("repo").u_ai.agg(["mean", "size"])
    per = per[per["size"] >= N_MIN]
    per["pct"] = 100 * per["mean"]
    first = pd.to_datetime(main.t_ai_first, utc=True).dropna()
    # monthly shares after enabling (first month with a share >= 10 %), large repositories
    mm = main.assign(month=main.created.dt.to_period("M"))
    monthly = {}
    for repo in ("woocommerce/woocommerce", "vercel/next.js", "supabase/supabase", "microsoft/vscode"):
        sh = mm[mm.repo == repo].groupby("month").u_ai.mean().mul(100)
        on = sh[sh >= 10]
        if len(on):
            s = sh[sh.index >= on.index[0]]
            monthly[short(repo).replace(".", "")] = {"min": float(s.min()), "max": float(s.max()), "from": str(on.index[0])}
    return {
        "n_tools": int(tool_n.size),
        "tools": [{"tool": k, "n": int(v), "repos": int(tool_repos.get(k, 0))} for k, v in tool_n.items()],
        "u_ai_n": int(main.u_ai.sum()), "u_ai_pct": pct(main.u_ai),
        "no_ai_pct": 100 - pct(main.u_ai),
        "year": {str(int(y)): float(v) for y, v in by_year.items()},
        "agent_n": int(len(ag)), "agent_pct": pct(main.author_group == "ai_agent"),
        "agent_top": short(ag.repo.value_counts().idxmax()), "agent_top_n": int(ag.repo.value_counts().max()),
        "agent_repos": int(ag.repo.nunique()),
        "repo_min": float(per.pct.min()), "repo_min_name": short(per.pct.idxmin()),
        "repo_max": float(per.pct.max()), "repo_max_name": short(per.pct.idxmax()),
        "repos_gt10": int((per.pct > 10).sum()), "repos_lt1": int((per.pct < 1).sum()),
        "repos_n": int(len(per)),
        "first_event": first.min().strftime("%d.%m.%Y") if len(first) else "",
        "monthly": monthly,
    }


def reactions(main):
    th = pd.read_csv(os.path.join(EXTRA, "r2_threads_v2.csv"))
    ar = pd.read_csv(os.path.join(EXTRA, "r2_ai_reviews_v2.csv"))
    ai = th[th.starter_class == "ai_assistant"].copy()
    ai["trig"] = ai.hours_since_trigger <= 1
    ar["verdict"] = ar.state.isin(["APPROVED", "CHANGES_REQUESTED"])
    cnt = ai.groupby("repo").size()
    shown = sorted(cnt[cnt >= REACT_MIN].index, key=lambda r: -cnt[r])
    rows = []

    def one(a, r_, name):
        sug = a[a.has_suggestion == 1]
        sug_h = sug[sug.author_group == "human"]
        return {"repo": name, "threads": int(len(a)), "reviews": int(len(r_)),
                "trig_pct": pct(a.trig), "reply_pct": pct(a.reacted), "resolved_pct": pct(a.is_resolved),
                "sugg_pct": pct(a.has_suggestion), "applied_pct": pct(sug_h.apply_commit_after),
                "verdict_pct": pct(r_.verdict) if len(r_) else 0.0}
    for r in shown:
        rows.append(one(ai[ai.repo == r], ar[ar.repo == r], short(r)))
    a_s, r_s = ai[ai.repo.isin(shown)], ar[ar.repo.isin(shown)]
    tot = one(a_s, r_s, "Разом")
    hidden = sorted(set(ai.repo) - set(shown))
    # reaction categories (repositories shown)
    res, rep, com = a_s.is_resolved == 1, a_s.reacted == 1, a_s.commits_after > 0
    by_tool = a_s.groupby("tool").reacted.agg(["mean", "size"])
    by_tool = by_tool[by_tool["size"] >= 200].sort_values("mean")
    vt = ar.groupby(["repo", "tool"]).verdict.agg(["mean", "size"])
    vt = vt[vt["size"] >= 100].sort_values("mean", ascending=False)
    hum = th[th.starter_class == "human"]
    bot = th[th.starter_class == "other_bot"]
    return {
        "rows": rows, "total": tot, "n_shown": len(shown), "n_hidden": len(hidden),
        "hidden": ", ".join(short(r) for r in hidden),
        "hidden_threads": int(ai[ai.repo.isin(hidden)].shape[0]),
        "all_threads": int(len(ai)), "all_reviews": int(len(ar)),
        "all_prs": int(ai[["repo", "pr_number"]].drop_duplicates().shape[0]),
        "res_commit_noreply_pct": pct(res & com & ~rep), "res_reply_pct": pct(res & rep),
        "nothing_pct": pct(~res & ~rep & ~com), "no_signal_pct": pct(~res & ~rep),
        "tool_reply_min": float(100 * by_tool["mean"].iloc[0]), "tool_reply_min_name": tool_short(by_tool.index[0]),
        "tool_reply_max": float(100 * by_tool["mean"].iloc[-1]), "tool_reply_max_name": tool_short(by_tool.index[-1]),
        "tool_reply_copilot": float(100 * by_tool["mean"].get("GitHub Copilot code review", float("nan"))),
        "signal_pct": 100 - pct(~res & ~rep),
        "verdict_2_pct": float(100 * vt["mean"].iloc[1]), "verdict_2_repo": short(vt.index[1][0]),
        "verdict_2_tool": tool_short(vt.index[1][1]),
        "verdict_top_pct": float(100 * vt["mean"].iloc[0]),
        "verdict_top_repo": short(vt.index[0][0]), "verdict_top_tool": tool_short(vt.index[0][1]),
        "trig_h_pct": pct(hum.hours_since_trigger <= 1), "trig_bot_pct": pct(bot.hours_since_trigger <= 1),
        "trig_all_pct": pct(ai.trig),
    }


TOOL_SHORT = {"GitHub Copilot code review": "GitHub Copilot", "OpenAI Codex code review": "OpenAI Codex",
              "Vercel Agent (VADE) code review": "Vercel Agent", "Claude Code (GitHub app)": "Claude Code",
              "Sentry AI code review (Seer)": "Sentry Seer",
              "Discourse Patch Triage (LLM reviewer)": "Discourse Patch Triage"}


def tool_short(name):
    return TOOL_SHORT.get(name, name)


def alignment_numbers():
    """Alignment of the process projection with the log (B.2): output of r2_alignment_v2.py."""
    with open(os.path.join(EXTRA, "r2_alignment_v2.json"), encoding="utf-8") as f:
        a = json.load(f)["align"]
    main_, rel = a["main"], a["relaxed"]
    worst = sorted(main_["by_repo"].items(), key=lambda kv: kv[1]["fit_pct"])
    rel_fit = [v["fit_pct"] for v in rel["by_repo"].values()]
    lm = main_["log_moves"]
    after_appr = sum(v for k, v in lm.items() if k.startswith("rev_approve|after_approval"))
    # log without repeated approvals, net main (experiment E1 of chapter 4)
    nr, nf, ne = a["main_noreapprove"], a["noreapprove_filter"], a["noreapprove_effect"]
    cat = nr["categories"]
    nr_fit = [v["fit_pct"] for v in nr["by_repo"].values()]
    nr_worst = sorted(nr["by_repo"].items(), key=lambda kv: kv[1]["fit_pct"])
    nr_keys = {
        "nr_fitness": nr["fitness"], "nr_fit_pct": nr["fit_pct"], "nr_log_fitness": nr["log_fitness"],
        "nr_precision": nr["precision"],
        "nr_repo_min_pct": min(nr_fit), "nr_n_ge90": sum(x >= 90 for x in nr_fit),
        "nr_worst_1": short(nr_worst[0][0]), "nr_worst_1_pct": nr_worst[0][1]["fit_pct"],
        "nr_dropped_n": nf["dropped_n"], "nr_dropped_events_pct": nf["dropped_pct"],
        "nr_dropped_traces_pct": nf["traces_pct"],
        "nr_fixed_pct": ne["fixed_pct"], "nr_worse_n": ne["worse_n"],
        "nr_nonfit_n": cat["nonfit_n"],
        "nr_explained_traces_pct": cat["explained_pct"],
        "nr_explained_abce_traces_pct": cat["explained_abce_pct"],
    }
    for c in "abcde":
        nr_keys[f"nr_cat_{c}_pct"] = cat["moves_pct"][c]
        nr_keys[f"nr_cat_{c}_traces_pct"] = cat["traces_with_pct"][c]
    return {
        "n_repos": a["n_repos"], "n_per_repo": a["n_per_repo"], "n_traces": a["n_traces"],
        "fitness": main_["fitness"], "fit_pct": main_["fit_pct"], "precision": main_["precision"],
        "rel_fitness": rel["fitness"], "rel_fit_pct": rel["fit_pct"], "rel_precision": rel["precision"],
        "repeat_approval_pct": 100.0 * after_appr / main_["log_moves_total"],
        "worst_1": short(worst[0][0]), "worst_1_pct": worst[0][1]["fit_pct"],
        "worst_2": short(worst[1][0]), "worst_2_pct": worst[1][1]["fit_pct"],
        "rel_repo_min_pct": min(rel_fit), "rel_n_ge90": sum(x >= 90 for x in rel_fit),
        **nr_keys,
    }


def main():
    t, m = load()
    rt, tot, lt, gt, ext = repo_tables(m)
    numbers = {
        "_note": "Корпус v2 (40 репо), основна вибірка in_main = 1 і hait_filter = 1; "
                 "make_tables_r2.py, 31.10.2026.",
        "corpus": corpus_numbers(t, m),
        "draft": draft_numbers(m),
        "replay": replay_numbers(m),
        "dur": duration_numbers(m),
        "it": iteration_numbers(m),
        "stale": stale_numbers(m),
        "delta": delta_numbers(m),
        "repo_table": rt, "repo_total": tot, "lang_table": lt, "group_table": gt, "ext": ext,
        "revert": revert_numbers(t, m),
        "fr": fr_numbers(m),
        "ai": ai_numbers(m),
        "react": reactions(m),
        "ai_compare": compare(m),
        "audit": audit_numbers(),
        "align": alignment_numbers(),
    }
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if o is None or (isinstance(o, float) and np.isnan(o)):
            return "–"
        return o
    numbers["replay_table"] = numbers["replay"].pop("table")
    numbers["replay_total"] = numbers["replay"].pop("total")
    for x in numbers["ai"]["tools"]:
        x["short"] = tool_short(x["tool"])
    numbers = clean(numbers)
    with open(os.path.join(R2_DIR, "r2_numbers.json"), "w", encoding="utf-8") as f:
        json.dump(numbers, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in numbers.items()
                      if k not in ("repo_table", "lang_table", "group_table", "ai_compare", "react")},
                     ensure_ascii=False, indent=1)[:12000])


if __name__ == "__main__":
    main()
