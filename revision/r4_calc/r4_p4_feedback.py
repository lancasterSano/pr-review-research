#!/usr/bin/env python3
"""Chapter 4, P4: the feedback-loop rollback rule replayed on the corpus history.

Question: would the feedback loop give the rollback rule of chapter 2 enough labelled
examples to reach a decision by the formal condition (ybar < y_min), if the rule had
run on the corpus history? A feasibility check on historical data, not a pilot: the
assistants in the corpus run with the projects' own settings, there is no entry
policy and no control group.

Chapter 2 of the dissertation:
  - label y(c) in {1, 0, null}: 2.4.4;
  - rollback criterion (second condition ybar < y_min): 2.4.5;
  - check rule: appendix B.4 (Wilson interval, three states, level 1 - 0.05 * 2^-i
    for the i-th check, design effect by PR, check gate below).

Procedure:
  - ybar = share of y = 1 among labelled examples (y in {0, 1}); null is left out;
  - check after every release cycle once labelled examples are in at least n_f = 20 PRs
    and n / max(1, deff) >= n_f (appendix B.4; deff is
    the one used in the interval of that check); quarters stand in for release cycles (the corpus has
    no releases); an example joins the quarter in which its PR was merged, because the
    label (reactions until merge) is final only then;
  - cumulative examples from the first quarter of the repository (B.4: "examples
    accumulate until the next check"; 2.4.5: "accumulated in the pilot");
  - a quarter without new labelled examples or failing the gate is not a check: i counts
    only the checks performed (it would only spend alpha);
  - states by the Wilson interval at level 1 - 0.05 * 2^-i: rollback (upper < y_min),
    keep (lower >= y_min), undecided (interval contains y_min); the procedure stops at
    the first decision;
  - Rao-Scott correction: n_eff = n / deff, deff from the between-PR variance of the
    ratio estimator (cluster = repo + pr_number), floored at 1;
  - no Holm correction; no required-n formula.

Label (chapter 2, 2.4.4, events only, no text):
  y = 1: the suggestion was applied (sugg_applied), or there was a revision after the
         comment and a human resolved the thread;
  y = 0: a human resolved the thread without a revision;
  null:  everything else (no reaction, a reply without resolution, a revision only,
         a resolution by the assistant itself or another bot).
"Human resolved" = isResolved and resolvedBy is human by participant_class (raw).
What is observed as "revision after the comment" (key prefix says what is observed):
  - obs_line (main):  the commented line changed (isOutdated in raw);
  - obs_any:          a later commit (commits_after > 0, by authoredDate) or isOutdated;
  - obs_commit:       a later commit only (an amend with force-push keeps authoredDate,
                      so this misses such revisions);
  sens_bots_line: obs_line with resolutions by any account.
Baseline: the same labels for threads started by human reviewers and deterministic bots
(starter_class in r2_threads_v2.csv; the same raw fields, second raw pass, cached in
raw_thread_flags_base.csv), keys e2.p4.base.<obs>.*.

Input (read only):
  $R2_DIR/v2/r2_threads_v2.csv (revision/r2/r2_raw_v2.py),
  $R2_DIR/r2_numbers.json (count check),
  $R3_ROOT/v2/pr_table_v2.csv (merged_at; default revision/output/r3_b40d),
  $R3_RAW_V2/<owner_repo>/prs.jsonl (isOutdated, resolvedBy; default data/raw_v2).
Output: revision/output/r4/p4_feedback/
  numbers.json, by_repo_quarter.csv, by_repo.csv, base_by_repo.csv,
  raw_thread_flags.csv and raw_thread_flags_base.csv (caches of the raw passes).

Run from the repository root (the raw pass takes a few minutes; --refresh redoes it):
  python revision/r4_calc/r4_p4_feedback.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.abspath(os.path.join(HERE, ".."))
CLONE = os.path.abspath(os.path.join(REV, ".."))
sys.path.insert(0, REV)

import r3_v2_prep as prep  # noqa: E402
import taskE1_common as e1  # noqa: E402
from review_iterations import login_of, parse_dt  # noqa: E402
from taskF1_duration_language import participant_class  # noqa: E402

RAW_V2 = os.environ.get("R3_RAW_V2") or os.path.join(CLONE, "data", "raw_v2")
R2_TABLES = os.environ.get("R2_DIR") or os.path.join(REV, "output", "r2")
THREADS = os.path.join(R2_TABLES, "v2", "r2_threads_v2.csv")
R2_NUMBERS = os.path.join(R2_TABLES, "r2_numbers.json")
PR_TABLE = os.path.join(os.environ.get("R3_ROOT") or os.path.join(REV, "output", "r3_b40d"), "v2", "pr_table_v2.csv")
OUT = os.path.join(REV, "output", "r4", "p4_feedback")

SEED = 20261114  # no stochastic step; fixed for the record
N_F = 20
Y_MIN = 0.5
ALPHA = 0.05
# variant -> (revision column, resolution column); y_min sensitivity for the three obs_*
VARIANTS = {"obs_line": ("rev_outdated", "res_h"), "obs_any": ("rev_commit_or_outdated", "res_h"),
            "obs_commit": ("rev_commit", "res_h"), "sens_bots_line": ("rev_outdated", "res_all")}
BASE_OBS = ("obs_line", "obs_any")
BASE_MIN = 50  # labelled threads of a group in a repository for repo-level comparisons
Y_MIN_SENS = {"ymin06": 0.6, "ymin07": 0.7}
SMALL_N, SMALL_EFF = 50, 20  # decisions taken on few examples
PFX = "e2.p4."


# ── raw pass: isOutdated and resolvedBy per assistant thread ─────────────────
def raw_flags(ai):
    """Per assistant thread, in the order of r2_raw_v2.py: isOutdated, isResolved,
    resolvedBy login and its class. Rows are matched to r2_threads_v2.csv by position
    within the PR and checked by createdAt."""
    want = {}
    for (repo, num), g in ai.groupby(["repo", "pr_number"], sort=False):
        want.setdefault(repo, {})[int(num)] = list(g.created_at)
    rows = []
    for repo in sorted(want):
        need, seen = want[repo], set()
        for pr in prep.iter_v2(RAW_V2, repo):
            num = int(pr["number"])
            if num in seen:
                continue
            seen.add(num)
            if num not in need:
                continue
            tm = prep.typename_map(pr)
            a_login = login_of(pr)
            ths = []
            for th in (pr.get("reviewThreads") or {}).get("nodes") or []:
                cs = (th.get("comments") or {}).get("nodes") or []
                if not cs:
                    continue
                w0 = login_of(cs[0])
                if w0 == a_login:
                    continue
                if participant_class(w0, tm.get(w0)) == "ai_assistant":
                    ths.append(th)
            ths.sort(key=lambda t: (((t.get("comments") or {}).get("nodes") or [{}])[0].get("createdAt") or ""))
            exp = need[num]
            if len(ths) != len(exp):
                raise SystemExit(f"thread count mismatch {repo}#{num}: raw {len(ths)} vs csv {len(exp)}")
            for i, th in enumerate(ths):
                c0 = th["comments"]["nodes"][0].get("createdAt") or ""
                if c0 != exp[i]:
                    raise SystemExit(f"createdAt mismatch {repo}#{num} [{i}]: {c0} vs {exp[i]}")
                rb = (th.get("resolvedBy") or {}).get("login") or ""
                res = bool(th.get("isResolved"))
                rows.append({"repo": repo, "pr_number": num, "pos": i, "created_at": c0,
                             "is_outdated": int(bool(th.get("isOutdated"))), "raw_resolved": int(res),
                             "resolved_by": rb,
                             "resolver_class": (participant_class(rb, tm.get(rb)) if res else "")})
        missing = set(need) - seen
        if missing:
            raise SystemExit(f"{repo}: {len(missing)} PRs not found in raw")
        print(repo, len(need), flush=True)
    return pd.DataFrame(rows)


def raw_flags_base(hb):
    """Threads started by human reviewers and deterministic bots, in the order of
    r2_raw_v2.py (node order): the raw fields the label needs, computed as in
    taskE1_common.thread_record. Matched by position within the PR, checked by createdAt."""
    want = {}
    for (repo, num), g in hb.groupby(["repo", "pr_number"], sort=False):
        want.setdefault(repo, {})[int(num)] = list(zip(g.created_at, g.starter_class))
    rows = []
    for repo in sorted(want):
        need, seen = want[repo], set()
        for pr in prep.iter_v2(RAW_V2, repo):
            num = int(pr["number"])
            if num in seen:
                continue
            seen.add(num)
            if num not in need:
                continue
            tm = prep.typename_map(pr)
            a_login = login_of(pr)
            commits = e1.pr_commits(pr)
            ths = []
            for th in (pr.get("reviewThreads") or {}).get("nodes") or []:
                cs = (th.get("comments") or {}).get("nodes") or []
                if not cs:
                    continue
                w0 = login_of(cs[0])
                if w0 == a_login:
                    continue
                c0 = participant_class(w0, tm.get(w0))
                if c0 in ("human", "other_bot") and parse_dt(cs[0].get("createdAt")):
                    ths.append((th, c0))
            exp = need[num]
            if len(ths) != len(exp):
                raise SystemExit(f"base thread count mismatch {repo}#{num}: raw {len(ths)} vs csv {len(exp)}")
            for i, (th, c0) in enumerate(ths):
                first = th["comments"]["nodes"][0]
                ca = first.get("createdAt") or ""
                if (ca, c0) != exp[i]:
                    raise SystemExit(f"base mismatch {repo}#{num} [{i}]")
                t0 = parse_dt(ca)
                after = [m for dd, m in commits if dd > t0]
                rb = (th.get("resolvedBy") or {}).get("login") or ""
                res = bool(th.get("isResolved"))
                rows.append({"repo": repo, "pr_number": num, "pos": i, "is_outdated": int(bool(th.get("isOutdated"))),
                             "is_resolved": int(res), "resolver_class": participant_class(rb, tm.get(rb)) if res else "",
                             "commits_after": len(after),
                             "has_suggestion": int("```suggestion" in (first.get("body") or "")),
                             "apply_commit_after": int(any(e1.APPLY_RX.search(m) for m in after))})
        if set(need) - seen:
            raise SystemExit(f"{repo}: base PRs not found in raw")
        print("base", repo, len(need), flush=True)
    return pd.DataFrame(rows)


# ── labels ───────────────────────────────────────────────────────────────────
NULL = -1


def label(d, rev, res):
    """2.4.4: y = 1 if the suggestion was applied or (revision and resolved); y = 0 if
    resolved without revision; null otherwise."""
    y1 = (d.sugg_applied == 1) | (rev & res)
    y0 = ~y1 & res & ~rev
    return np.where(y1, 1, np.where(y0, 0, NULL))


def null_kind(d, y, rev, res):
    rep = d.reacted == 1
    k = np.full(len(d), "", dtype=object)
    isn = y == NULL
    k[isn & ~res & ~rep & ~rev & d.res_all] = "bot_resolved_only"
    k[isn & ~res & ~rep & ~rev & ~d.res_all] = "no_reaction"
    k[isn & ~res & ~rep & rev] = "revision_only"
    k[isn & ~res & rep] = "reply_unresolved"
    k[isn & (k == "")] = "other"
    return k


# ── sequential check ─────────────────────────────────────────────────────────
def design_effect(n_i, x_i):
    """Rao-Scott (1992) design effect of the ratio p = sum x / sum n over clusters."""
    n_i, x_i = np.asarray(n_i, float), np.asarray(x_i, float)
    m, N = len(n_i), n_i.sum()
    if m < 2 or N == 0:
        return 1.0, 1.0
    p = x_i.sum() / N
    if p <= 0 or p >= 1:
        return 1.0, 1.0
    v_cl = m / (m - 1) * np.sum((x_i - p * n_i) ** 2) / N ** 2
    v_bin = p * (1 - p) / N
    raw = v_cl / v_bin
    return raw, max(1.0, raw)


def wilson(p, n, z):
    if n <= 0:
        return np.nan, np.nan
    z2 = z * z
    den = 1 + z2 / n
    c = (p + z2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def state_of(lo, hi, y_min=Y_MIN):
    if hi < y_min:
        return "rollback"
    if lo >= y_min:
        return "keep"
    return "undecided"


def replay(d, col, use_deff=True, y_min=Y_MIN):
    """Rows repo x quarter for one approximation; each repository from its first quarter."""
    out = []
    lab = d[d[col] != NULL]
    for repo, g in d.groupby("repo"):
        qs = pd.period_range(g.quarter.min(), d.quarter.max(), freq="Q")
        gl = lab[lab.repo == repo]
        i = 0
        decided = None
        prev_n = 0
        for k, q in enumerate(qs, start=1):
            cum = gl[gl.quarter <= q]
            n = len(cum)
            rec = {"repo": repo, "approx": col.replace("y_", ""), "quarter": str(q), "quarter_idx": k,
                   "n_new_threads": int((g.quarter == q).sum()), "n_new_labelled": n - prev_n,
                   "n_labelled": n, "ybar": cum[col].mean() if n else np.nan,
                   "check_i": "", "level": "", "deff_raw": "", "deff": "", "n_eff": "", "m_prs": "",
                   "wilson_lo": "", "wilson_hi": "", "state": "", "after_decision": int(decided is not None)}
            gate = False
            if n > prev_n:
                pc = cum.groupby("pr_number")[col].agg(["size", "sum"])
                draw, deff = design_effect(pc["size"], pc["sum"])
                if not use_deff:
                    deff = 1.0
                rec["m_prs"] = len(pc)
                gate = len(pc) >= N_F and n / deff >= N_F       # B.4: n_f PRs and n / deff >= n_f
            if gate:
                i += 1
                a_i = ALPHA * 2.0 ** (-i)
                z = norm.ppf(1 - a_i / 2)
                lo, hi = wilson(rec["ybar"], n / deff, z)
                st = state_of(lo, hi, y_min)
                rec.update(check_i=i, level=1 - a_i, deff_raw=draw, deff=deff, n_eff=n / deff,
                           wilson_lo=lo, wilson_hi=hi, state=st)
                if decided is None and st != "undecided":
                    decided = (st, k, i)
            elif n > prev_n:
                rec["state"] = "no_check_gate"
            else:
                rec["state"] = "no_check_no_new"
            prev_n = n
            out.append(rec)
    return pd.DataFrame(out)


def summarise(bq):
    """Per repository: state at the last check performed (the procedure stops at the first
    decision), quarters and checks to that decision, and the no-stopping last state."""
    rows = []
    for repo, g in bq.groupby("repo"):
        chk = g[g.check_i != ""]
        dec = chk[chk.state != "undecided"]
        r = {"repo": repo, "n_quarters": len(g), "n_checks_all": len(chk)}
        if len(dec):
            f = dec.iloc[0]
            r.update(state=f.state, quarters_to_decision=int(f.quarter_idx), checks_to_decision=int(f.check_i),
                     decision_quarter=f.quarter, n_at_decision=int(f.n_labelled),
                     prs_at_decision=int(f.m_prs), ybar_at_decision=f.ybar,
                     deff_at_last_check=f.deff)
        else:
            r.update(state="undecided" if len(chk) else "undecided_no_check", quarters_to_decision=np.nan,
                     checks_to_decision=np.nan, decision_quarter="", n_at_decision=np.nan, prs_at_decision=np.nan,
                     ybar_at_decision=np.nan,
                     deff_at_last_check=chk.deff.iloc[-1] if len(chk) else np.nan)
        r["state_nostop_last"] = chk.state.iloc[-1] if len(chk) else "undecided_no_check"
        last = g.iloc[-1]
        r.update(n_labelled_total=int(last.n_labelled), ybar_total=last.ybar)
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    th = pd.read_csv(THREADS)
    ai = th[th.starter_class == "ai_assistant"].reset_index(drop=True)
    ai["pos"] = ai.groupby(["repo", "pr_number"]).cumcount()
    cache = os.path.join(OUT, "raw_thread_flags.csv")
    if "--refresh" in sys.argv or not os.path.exists(cache):
        raw_flags(ai).to_csv(cache, index=False)
    fl = pd.read_csv(cache, keep_default_na=False)
    d = ai.merge(fl.drop(columns="created_at"), on=["repo", "pr_number", "pos"], how="left", validate="1:1")
    assert d.is_outdated.notna().all() and len(d) == len(ai)
    mism = int((d.raw_resolved.astype(int) != d.is_resolved).sum())
    assert mism == 0, f"is_resolved differs from raw in {mism} threads"

    pt = pd.read_csv(PR_TABLE, usecols=["repo", "pr_number", "merged_at"])
    d = d.merge(pt, on=["repo", "pr_number"], how="left", validate="m:1")
    when = pd.to_datetime(d.merged_at.fillna(d.created_at), utc=True)
    d["quarter"] = when.dt.tz_localize(None).dt.to_period("Q")

    d["res_all"] = d.is_resolved == 1
    d["res_h"] = d.res_all & (d.resolver_class == "human")
    d["rev_commit"] = d.commits_after > 0
    d["rev_outdated"] = d.is_outdated == 1
    d["rev_commit_or_outdated"] = d.rev_commit | d.rev_outdated
    for v, (rv, rs) in VARIANTS.items():
        d["y_" + v] = label(d, d[rv], d[rs])

    with open(R2_NUMBERS, encoding="utf-8") as f:
        r2 = json.load(f)["react"]
    n_pr = d[["repo", "pr_number"]].drop_duplicates().shape[0]
    assert len(d) == r2["all_threads"] and n_pr == r2["all_prs"], "thread / PR count differs from react.*"

    num = {PFX + "n_repos": int(d.repo.nunique()), PFX + "n_threads": int(len(d)), PFX + "n_prs": int(n_pr),
           PFX + "threads_per_pr": len(d) / n_pr,
           PFX + "check.react_all_threads": r2["all_threads"], PFX + "check.react_all_prs": r2["all_prs"],
           PFX + "cycle_proxy": "квартал",
           PFX + "cycle_note": ("квартали – наближення циклів випуску (випусків у корпусі немає); приклад "
                                "належить кварталу злиття запиту; останній квартал неповний"),
           PFX + "first_quarter": str(d.quarter.min()), PFX + "last_quarter": str(d.quarter.max()),
           PFX + "n_f": N_F, PFX + "n_f_rule": "m >= n_f запитів з позначеними прикладами і n / max(1, deff) >= n_f (Б.4, b73828f)", PFX + "y_min": Y_MIN, PFX + "alpha": ALPHA,
           PFX + "level_rule": "1 - 0.05 * 2^-i, i-та перевірка",
           PFX + "accumulation": "кумулятивно від першого кварталу репозиторію; квартал без нових позначених "
                                 "прикладів не є перевіркою; зупинка на першому рішенні",
           PFX + "deff_rule": "Rao-Scott за запитами (repo + pr_number), не менше 1",
           PFX + "label_rule": ("2.4.4 (499e7ce): 1 – пропозицію застосовано або ревізія і людина вирішила; "
                                "0 – людина вирішила без ревізії; ∅ – інше"),
           PFX + "revision_rule": ("obs_line (основне) – змінено прокоментований рядок (isOutdated); obs_any – новий коміт після "
                                   "зауваження (commits_after > 0) або isOutdated; obs_commit – лише commits_after > 0 "
                                   "(amend із force-push зберігає authoredDate); sens_bots_line – obs_line з вирішеннями будь-ким"),
           PFX + "seed": SEED}
    res = d[d.res_all]
    num[PFX + "resolved_pct"] = 100 * d.res_all.mean()
    for c in ("human", "ai_assistant", "other_bot", "uncertain"):
        num[PFX + f"resolver.{c}_pct"] = 100 * (res.resolver_class == c).mean()
    num[PFX + "resolver.missing_login_pct"] = 100 * (res.resolved_by == "").mean()
    num[PFX + "outdated_pct"] = 100 * d.is_outdated.mean()

    bq_all, by_all, deffs = [], [], []
    for a, (rv, rs) in VARIANTS.items():
        col = "y_" + a
        y = d[col].to_numpy()
        n1, n0, nn = int((y == 1).sum()), int((y == 0).sum()), int((y == NULL).sum())
        N = len(y)
        kinds = pd.Series(null_kind(d, y, d[rv], d[rs]))
        bq = replay(d, col)
        sm = summarise(bq)
        sm.insert(1, "approx", a)
        bq_nd = replay(d, col, use_deff=False)
        sm_nd = summarise(bq_nd)
        st = sm.state.replace({"undecided_no_check": "undecided"})
        dec = sm[sm.state.isin(["rollback", "keep"])]
        pr_ = PFX + a + "."
        num.update({
            pr_ + "n_repos_rollback": int((st == "rollback").sum()), pr_ + "n_repos_keep": int((st == "keep").sum()),
            pr_ + "n_repos_undecided": int((st == "undecided").sum()),
            pr_ + "n_repos_no_check": int((sm.state == "undecided_no_check").sum()),
            pr_ + "median_quarters_to_decision": float(dec.quarters_to_decision.median()) if len(dec) else None,
            pr_ + "min_quarters_to_decision": float(dec.quarters_to_decision.min()) if len(dec) else None,
            pr_ + "max_quarters_to_decision": float(dec.quarters_to_decision.max()) if len(dec) else None,
            pr_ + "median_checks_to_decision": float(dec.checks_to_decision.median()) if len(dec) else None,
            pr_ + "n1": n1, pr_ + "n0": n0, pr_ + "n_null": nn,
            pr_ + "ybar": n1 / (n1 + n0), pr_ + "ybar_pooled": n1 / (n1 + n0), pr_ + "ybar_lo_bound": n1 / N, pr_ + "ybar_hi_bound": (n1 + nn) / N,
            pr_ + "null_pct": 100 * nn / N,
            pr_ + "null_pct_median_repo": float(d.assign(_n=y == NULL).groupby("repo")._n.mean().median() * 100),
            pr_ + "deff_median": float(pd.to_numeric(sm.deff_at_last_check, errors="coerce").median()),
            pr_ + "nostop.n_repos_rollback": int((sm.state_nostop_last == "rollback").sum()),
            pr_ + "nostop.n_repos_keep": int((sm.state_nostop_last == "keep").sum()),
            pr_ + "nodeff.n_repos_rollback": int((sm_nd.state == "rollback").sum()),
            pr_ + "nodeff.n_repos_keep": int((sm_nd.state == "keep").sum()),
            pr_ + "nodeff.median_quarters_to_decision": float(
                sm_nd[sm_nd.state.isin(["rollback", "keep"])].quarters_to_decision.median()),
        })
        dn = pd.to_numeric(dec.n_at_decision)
        num[pr_ + "n_dec_n_lt50"] = int((dn < SMALL_N).sum())
        num[pr_ + "n_dec_eff_lt20"] = int((dn / pd.to_numeric(dec.deff_at_last_check) < SMALL_EFF).sum())
        dp = pd.to_numeric(dec.prs_at_decision)
        if len(dec):
            num.update({pr_ + "dec_n_min": int(dn.min()), pr_ + "dec_n_median": float(dn.median()),
                        pr_ + "dec_prs_min": int(dp.min()), pr_ + "dec_prs_median": float(dp.median()),
                        pr_ + "dec_neff_min": float((dn / pd.to_numeric(dec.deff_at_last_check)).min())})
        lab = d[y != NULL]
        num[pr_ + "n_repos_multi_tool"] = int((lab.groupby("repo").tool.nunique() >= 2).sum())
        dm = (lab.groupby(["repo", "quarter", "tool"]).size().reset_index(name="n")
              .sort_values(["repo", "quarter", "n", "tool"], ascending=[True, True, False, True])
              .drop_duplicates(["repo", "quarter"]))
        num[pr_ + "n_repos_tool_switch"] = int((dm.groupby("repo").tool.nunique() >= 2).sum())
        if a in ("obs_line", "obs_any", "obs_commit"):
            for key, ym in Y_MIN_SENS.items():
                s_ = summarise(replay(d, col, y_min=ym)).state.replace({"undecided_no_check": "undecided"})
                for stn in ("rollback", "keep", "undecided"):
                    num[pr_ + f"{key}.n_repos_{stn}"] = int((s_ == stn).sum())
        for k in ("no_reaction", "revision_only", "reply_unresolved", "bot_resolved_only", "other"):
            num[pr_ + f"null.{k}_pct"] = 100 * float((kinds == k).mean())
        # repo-level ybar bounds
        g = d.assign(_y1=y == 1, _yn=y == NULL).groupby("repo")
        sm = sm.merge(pd.DataFrame({"n_threads": g.size(), "ybar_lo_bound": g._y1.mean(),
                                    "ybar_hi_bound": (g._y1.sum() + g._yn.sum()) / g.size(),
                                    "null_pct": 100 * g._yn.mean()}).reset_index(), on="repo")
        chk = bq[bq.check_i != ""]
        deffs.extend(pd.to_numeric(chk.deff_raw).tolist())
        bq_all.append(bq)
        by_all.append(sm)
    num[PFX + "deff_median"] = float(np.median(np.maximum(1.0, deffs)))
    # assistants per repository: >= 2 tools; dominant tool (most threads) changes between quarters
    n_tools = d.groupby("repo").tool.nunique()
    dom = (d.groupby(["repo", "quarter", "tool"]).size().reset_index(name="n")
           .sort_values(["repo", "quarter", "n", "tool"], ascending=[True, True, False, True])
           .drop_duplicates(["repo", "quarter"]))
    n_dom = dom.groupby("repo").tool.nunique()
    num[PFX + "n_repos_multi_tool"] = int((n_tools >= 2).sum())
    num[PFX + "n_repos_tool_switch"] = int((n_dom >= 2).sum())
    num[PFX + "tool_note"] = ("інструмент = поле tool у r2_threads_v2; домінантний – найбільше обговорень у кварталі "
                              "злиття (за рівності – перший за назвою); зміна – різні домінантні в кварталах з обговореннями")
    num[PFX + "deff_raw_median_all_checks"] = float(np.median(deffs))
    num[PFX + "deff_raw_p90_all_checks"] = float(np.percentile(deffs, 90))

    # threats: obs_any split by a later commit; y = 0 without a human reply
    lab = d[d.y_obs_any != NULL]
    num[PFX + "obs_any.ybar_with_commit"] = float((lab[lab.rev_commit].y_obs_any == 1).mean())
    num[PFX + "obs_any.ybar_without_commit"] = float((lab[~lab.rev_commit].y_obs_any == 1).mean())
    num[PFX + "obs_any.share_labelled_with_commit_pct"] = 100 * float(lab.rev_commit.mean())
    num[PFX + "zero_no_reply_pct"] = 100 * float((d[d.y_obs_any == 0].reacted == 0).mean())
    num[PFX + "obs_line.zero_no_reply_pct"] = 100 * float((d[d.y_obs_line == 0].reacted == 0).mean())
    num[PFX + "zero_no_reply_note"] = "частка y = 0 без відповіді людини; zero_no_reply_pct – за obs_any"
    baseline(d, th, num)

    bq = pd.concat(bq_all, ignore_index=True)
    bq.to_csv(os.path.join(OUT, "by_repo_quarter.csv"), index=False, float_format="%.4f")
    pd.concat(by_all, ignore_index=True).to_csv(os.path.join(OUT, "by_repo.csv"), index=False, float_format="%.4f")
    with open(os.path.join(OUT, "numbers.json"), "w", encoding="utf-8") as f:
        json.dump(num, f, ensure_ascii=False, indent=1, default=float)
    for k, v in num.items():
        print(k, "=", v)


def baseline(d, th, num):
    """Same labels for threads started by human reviewers and deterministic bots."""
    hb = th[th.starter_class.isin(["human", "other_bot"])].reset_index(drop=True)
    hb["pos"] = hb.groupby(["repo", "pr_number"]).cumcount()
    cache = os.path.join(OUT, "raw_thread_flags_base.csv")
    if "--refresh" in sys.argv or not os.path.exists(cache):
        raw_flags_base(hb).to_csv(cache, index=False)
    fl = pd.read_csv(cache, keep_default_na=False)
    b = hb.drop(columns=["is_resolved", "commits_after", "has_suggestion", "apply_commit_after", "sugg_applied",
                         "reacted"], errors="ignore")
    b = b.merge(fl, on=["repo", "pr_number", "pos"], how="left", validate="1:1")
    assert len(b) == len(hb) and b.is_outdated.notna().all()
    b["sugg_applied"] = ((b.has_suggestion == 1) & (b.apply_commit_after == 1) & (b.author_group == "human")).astype(int)
    b["res_h"] = (b.is_resolved == 1) & (b.resolver_class == "human")
    b["rev_outdated"] = b.is_outdated == 1
    b["rev_commit_or_outdated"] = (b.commits_after > 0) | b.rev_outdated
    b["group"] = b.starter_class.map({"human": "human", "other_bot": "bot"})
    a = d.assign(group="ai")
    num[PFX + "base.n_threads.human"] = int((b.group == "human").sum())
    num[PFX + "base.n_threads.bot"] = int((b.group == "bot").sum())
    num[PFX + "base.min_labelled_repo"] = BASE_MIN
    rows = []
    for obs in BASE_OBS:
        rv = VARIANTS[obs][0]
        b["y_" + obs] = label(b, b[rv], b.res_h)
        allg = pd.concat([a[["repo", "group", "y_" + obs]], b[["repo", "group", "y_" + obs]]], ignore_index=True)
        allg = allg[allg["y_" + obs] != NULL]
        pr_ = PFX + f"base.{obs}."
        per = allg.groupby(["repo", "group"])["y_" + obs].agg(["size", "mean"]).reset_index()
        per["obs"] = obs
        rows.append(per)
        for grp in ("ai", "human", "bot"):
            g = allg[allg.group == grp]
            num[pr_ + f"{grp}.ybar_pooled"] = float(g["y_" + obs].mean())
            num[pr_ + f"{grp}.null_pct"] = 100 * float(1 - len(g) / ((a if grp == "ai" else b[b.group == grp]).shape[0]))
            pg = per[(per.group == grp) & (per["size"] >= BASE_MIN)]
            num[pr_ + f"{grp}.n_repos_ge_half"] = int((pg["mean"] >= 0.5).sum())
            num[pr_ + f"{grp}.n_repos_of"] = int(len(pg))
        w = per[per["size"] >= BASE_MIN].pivot(index="repo", columns="group", values="mean")
        w = w.dropna(subset=["ai", "human"])
        num[pr_ + "n_repos_ai_below_human"] = int((w.ai < w.human).sum())
        num[pr_ + "n_repos_of"] = int(len(w))
        num[pr_ + "n_repos_group_ge_half"] = int((w.human >= 0.5).sum())
        num[pr_ + "n_repos_ai_ge_half"] = int((w.ai >= 0.5).sum())
        num[pr_ + "rho_ai_human"] = float(spearmanr(w.ai, w.human).correlation)
    num[PFX + "base.note"] = ("ai = обговорення асистентів (ті самі, що в П4); n_repos_* – репо з ≥ 50 позначеними; "
                              "n_repos_group_ge_half – людський ȳ ≥ 0,5 серед репо з ≥ 50 позначеними і в асистентів, і в людей")
    pd.concat(rows, ignore_index=True).to_csv(os.path.join(OUT, "base_by_repo.csv"), index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
