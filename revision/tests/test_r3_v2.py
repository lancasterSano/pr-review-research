"""Chapter 3 on corpus v2: labels, features at t0, leak guards, reverts, models.

Each test names the production change that would make it fail.
Run from the project root:  python -m pytest revision/tests/test_r3_v2.py -q
"""
import json
import math
import os
import random
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

_REV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REV)

import r3_v2_prep as P  # noqa: E402
import r3_v2_models as M  # noqa: E402
import taskF1_duration_language as f1  # noqa: E402

T0 = datetime(2025, 6, 2, 9, 0, tzinfo=timezone.utc)


def iso(minutes):
    return (T0 + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


def at(minutes):
    return T0 + timedelta(minutes=minutes)


def make_pr(author="alice", number=1, reviews=(), commits=(), threads=(), timeline=(),
            merged_min=24 * 60, state="MERGED", is_draft=False, title="t", body="b",
            base="main", additions=10, deletions=2):
    """reviews: (login, minutes); commits: (minutes, additions, deletions[, oid, message]);
    threads: (login, minutes); timeline: (__typename, minutes)."""
    cnodes = []
    for c in commits:
        m, a, d = c[:3]
        cnodes.append({"commit": {"authoredDate": iso(m), "committedDate": iso(m),
                                  "additions": a, "deletions": d, "changedFilesIfAvailable": 1,
                                  "oid": c[3] if len(c) > 3 else f"{number:07d}{m:033d}",
                                  "message": c[4] if len(c) > 4 else ""}})
    return {
        "number": number, "state": state, "isDraft": is_draft, "title": title, "body": body,
        "author": {"login": author, "__typename": "User"},
        "createdAt": iso(0), "mergedAt": iso(merged_min) if merged_min is not None else None,
        "closedAt": iso(merged_min) if merged_min is not None else None,
        "additions": additions, "deletions": deletions, "baseRefName": base,
        "reviews": {"totalCount": len(reviews), "nodes": [
            {"author": {"login": a, "__typename": "User"}, "state": "COMMENTED",
             "submittedAt": iso(m), "body": ""} for a, m in reviews]},
        "commits": {"totalCount": len(cnodes), "nodes": cnodes},
        "reviewThreads": {"nodes": [{"comments": {"nodes": [
            {"author": {"login": a, "__typename": "User"}, "createdAt": iso(m)}]}} for a, m in threads]},
        "timelineItems": {"nodes": [{"__typename": k, "createdAt": iso(m)} for k, m in timeline]},
    }


# ── t_rd and clocks ─────────────────────────────────────────────────────────

def test_created_as_draft_starts_clock_at_first_ready():
    # fails if t_rd ignores ReadyForReviewEvent
    pr = make_pr(timeline=[("ReadyForReviewEvent", 120), ("ConvertToDraftEvent", 200),
                           ("ReadyForReviewEvent", 300)], reviews=[("bob", 180)])
    as_draft, t_rd, n_ready, n_back, fb = P.draft_info(pr)
    assert as_draft and t_rd == at(120) and n_ready == 2 and n_back == 1 and not fb
    rec = P.pr_record(pr, "o/r")
    assert rec["t_r_h_D"] == pytest.approx(1.0)        # 180 - 120 min
    assert rec["t_r_h_A"] == pytest.approx(3.0)
    assert rec["t_m_h_D"] == pytest.approx((24 * 60 - 120) / 60)
    assert rec["is_draft_t0"] == 1


def test_created_ready_then_converted_keeps_creation_as_t_rd():
    # fails if a later draft round moves t_rd for a PR that was ready at creation
    pr = make_pr(timeline=[("ConvertToDraftEvent", 60), ("ReadyForReviewEvent", 90)])
    as_draft, t_rd, *_ = P.draft_info(pr)
    assert not as_draft and t_rd == at(0)


def test_review_on_draft_does_not_count_for_t_r():
    # fails if a review submitted on the draft still defines t_r under clock D
    pr = make_pr(timeline=[("ReadyForReviewEvent", 300)], reviews=[("bob", 100)])
    rec = P.pr_record(pr, "o/r")
    assert rec["t_r_h_D"] is None and rec["t_r_h_D_raw"] < 0 and rec["first_review_in_draft"] == 1
    pr = make_pr(timeline=[("ReadyForReviewEvent", 300)], reviews=[("bob", 100), ("carol", 420)])
    assert P.pr_record(pr, "o/r")["t_r_h_D"] == pytest.approx(2.0)


def test_bots_and_assistants_do_not_count_as_first_review():
    # fails if participant classification is dropped from the label
    pr = make_pr(reviews=[("copilot-pull-request-reviewer", 5), ("dependabot[bot]", 6), ("bob", 60)])
    assert P.pr_record(pr, "o/r")["t_r_h_A"] == pytest.approx(1.0)


def test_staleness_clock_D_ignores_draft_period():
    # fails if the staleness trace still starts at creation under clock D
    day = 24 * 60
    pr = make_pr(timeline=[("ReadyForReviewEvent", 40 * day)], reviews=[("bob", 41 * day)],
                 merged_min=42 * day)
    rec = P.pr_record(pr, "o/r")
    assert rec["stale30_A"] == 1 and rec["stale30_D"] == 0


def test_clock_A_trace_equals_taskF1_trace():
    # fails if activity_trace_from drifts from the ITIA staleness trace
    pr = make_pr(reviews=[("bob", 50), ("claude", 70)], commits=[(10, 1, 1), (90, 1, 1)],
                 threads=[("carol", 400), ("coderabbitai", 500)])
    inc = P.human_include(pr)
    assert P.activity_trace_from(pr, inc, pr["createdAt"]) == f1.activity_trace(pr, lambda w: f1.participant_class(w) == "human")


def test_issue_comments_are_not_activity():
    # fails if IssueComment starts resetting the staleness timer (decision: variant A)
    day = 24 * 60
    pr = make_pr(merged_min=40 * day, timeline=[("IssueComment", 20 * day)])
    assert P.pr_record(pr, "o/r")["stale30_D"] == 1


# ── features at t0 ──────────────────────────────────────────────────────────

def test_size_at_t0_counts_only_commits_authored_by_creation():
    # fails if the final size or later commits leak into the size feature
    pr = make_pr(commits=[(-30, 5, 1), (0, 2, 0), (60, 100, 50)])
    assert P.size_at_t0(pr) == (7, 1, 2, 2)


def test_v1_commits_without_stats_give_nan_size():
    pr = make_pr(commits=[(-30, 5, 1)])
    del pr["commits"]["nodes"][0]["commit"]["additions"]
    a, d, f, n = P.size_at_t0(pr)
    assert math.isnan(a) and n == 1


def _pr_level_t0(rec):
    hist = {"author_pr_count", "author_tenure_days", "author_merge_rate", "author_avg_review_rounds_h",
            "author_avg_tfr_human", "author_commit_frequency_s", "open_pr_count_t0",
            "merge_velocity_7d_t0", "merge_velocity_30d_t0"}
    return {k: rec[k] for k in P.FEATURES_T0 if k not in hist}


def test_pr_level_features_equal_on_snapshot_at_t0():
    # Leak test: the PR as it looked at t0 (draft, commits pushed by t0, no
    # reviews, not merged) gives the same features as the final snapshot.
    full = make_pr(commits=[(-5, 3, 1), (200, 50, 9)], reviews=[("bob", 100)],
                   timeline=[("ReadyForReviewEvent", 60), ("IssueComment", 70)],
                   body="Fixes #3 ```x``` ![a](b)")
    snap = make_pr(commits=[(-5, 3, 1)], merged_min=None, state="OPEN", is_draft=True,
                   body="Fixes #3 ```x``` ![a](b)")
    snap["mergedAt"] = None
    rec_full = P.pr_record(full, "o/r")
    # pr_record computes features only for merged PRs; evaluate the snapshot
    # through the same functions with a dummy merge far in the future.
    snap2 = dict(snap, mergedAt=iso(10 ** 6), state="MERGED")
    rec_snap = P.pr_record(snap2, "o/r")
    assert _pr_level_t0(rec_full) == _pr_level_t0(rec_snap)


def _random_repo(seed, n=120):
    rnd = random.Random(seed)
    recs = []
    authors = ["a", "b", "c", "d", ""]
    for i in range(n):
        c = T0 + timedelta(hours=rnd.uniform(0, 24 * 120))
        merged = rnd.random() < 0.7
        end = c + timedelta(hours=rnd.expovariate(1 / 72))
        recs.append({"author_login": rnd.choice(authors), "created_at": c,
                     "merged_at": end if merged else None, "closed_at": end,
                     "n_it_human_pre": rnd.randint(0, 3),
                     "t_r_pre_h_D": rnd.choice([None, rnd.uniform(0, 50)]),
                     "t_r_pre_h_A": rnd.uniform(0, 50), "commits_total": rnd.randint(1, 9)})
    return recs


def _truncate_history(recs, t0):
    """Everything as it was known strictly before t0: later PRs are absent and
    merges / closes at or after t0 have not happened (their outcomes unknown)."""
    out = []
    for r in recs:
        if r["created_at"] >= t0:
            continue
        q = dict(r)
        if q["merged_at"] and q["merged_at"] >= t0:
            q.update(merged_at=None, closed_at=None, n_it_human_pre=None, t_r_pre_h_D=None,
                     t_r_pre_h_A=None, commits_total=None)
        elif q["closed_at"] and q["closed_at"] >= t0:
            q["closed_at"] = None
        out.append(q)
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_removing_events_after_t0_does_not_change_history_features(seed):
    # Leak test required by the plan: fails if any history feature reads an
    # event at or after t0 (e.g. merges counted with a negative day difference,
    # the HAIT merge_velocity bug, or the outcome of a PR merged after t0).
    recs = _random_repo(seed)
    targets = random.Random(seed).sample(range(len(recs)), 25)
    full = P.history_features(recs, targets, "D")
    for i in targets:
        t0 = recs[i]["created_at"]
        trunc = _truncate_history(recs, t0) + [dict(recs[i])]
        got = P.history_features(trunc, [len(trunc) - 1], "D")[len(trunc) - 1]
        for k, v in full[i].items():
            w = got[k]
            assert (math.isnan(v) and math.isnan(w)) or v == pytest.approx(w), (k, v, w)


def test_thresholds_use_only_prs_merged_before_t0():
    recs = _random_repo(7)
    for r in recs:
        r.update(t_m_h_D=r["t_r_pre_h_A"], t_r_pre_h_D=r["t_r_pre_h_A"])
    sample = [i for i, r in enumerate(recs) if r["merged_at"]]
    thr = P.thresholds_and_priors(recs, sample, "D")[0]
    for i in sample[:20]:
        prior = [recs[j]["t_m_h_D"] for j in sample if recs[j]["merged_at"] < recs[i]["created_at"]]
        assert thr[i]["n_hist_tm"] == len(prior)
        if prior:
            assert thr[i]["thr_tm_h"] == pytest.approx(float(np.median(prior)))


# ── allow-list ──────────────────────────────────────────────────────────────

def test_feature_sets_stay_inside_allow_list():
    # fails if a model feature set includes a final-state feature
    for name, cols in M.FEATURE_SETS.items():
        M.check_features(cols)
        assert not set(cols) & P.FORBIDDEN, name


@pytest.mark.parametrize("bad", ["lines_added", "label_count", "has_tests", "num_commits",
                                 "merge_velocity_7d", "y_IT__main_D"])
def test_check_features_rejects_forbidden(bad):
    with pytest.raises(ValueError):
        M.check_features(P.FEATURES_T0 + [bad])


# ── reverts ─────────────────────────────────────────────────────────────────

def _t(n, minutes, title="Add x", oids=(), base="main"):
    return n, {"merged_at": at(minutes), "title": title, "oids": set(oids), "base": base}


def _r(n, minutes, nums=(), quoted=None, shas=(), base="main"):
    return {"number": n, "merged_at": at(minutes), "nums": set(nums), "quoted": quoted,
            "shas": set(shas), "base": base}


def test_revert_signals_parse_body_title_commits():
    pr = make_pr(title='Revert "Add x (#12)"', body="Reverts o/r#12\r\nbecause",
                 commits=[(0, 1, 1, "f" * 40, "Revert \"Add x\"\n\nThis reverts commit abcdef1234567.")])
    nums, quoted, shas = P.revert_signals(pr, "o/r")
    assert nums == {12} and quoted == "Add x (#12)" and shas == {"abcdef1234567"}
    assert P.revert_signals(make_pr(body="Partially reverts #12"), "o/r")[0] == set()
    assert P.revert_signals(make_pr(body="Reverts other/repo#12"), "o/r")[0] == set()


def test_match_reverts_evidence_window_base_and_revert_of_revert():
    day = 24 * 60
    targets = dict([_t(1, 0, oids=["abcdef1234567" + "0" * 27]), _t(2, 0, title="B"),
                    _t(3, 0, title="C"), _t(4, 0, title="D", base="release/1"),
                    _t(10, day, title='Revert "Add x"'), _t(11, 5, title="r2"),
                    _t(12, 40 * day, title="r3"), _t(13, day, title="r4"), _t(14, 2 * day, title="rr")])
    reverters = [_r(10, day, nums=[1], quoted="Add x", shas=["abcdef1"]),
                 _r(11, 5, nums=[2]),
                 _r(12, 40 * day, nums=[3]),                       # too late
                 _r(13, day, nums=[4], base="main"),               # other base branch
                 _r(14, 2 * day, nums=[11])]                       # reverts the revert of 2
    out, links = P.match_reverts(targets, reverters, window_end=at(10 ** 6))
    assert out[1]["reverted_30d"] == 1 and out[1]["revert_evidence"] == "body_reverts|commit_hash|title_revert"
    assert out[1]["revert_pr"] == 10
    assert out[2]["reverted_30d"] == 0 and out[2]["revert_reverted"] == 1
    assert out[3]["reverted_30d"] == 0
    assert out[4]["reverted_30d"] == 0
    assert out[10]["is_revert_pr"] == 1 and out[1]["is_revert_pr"] == 0
    assert out[11]["is_revert_pr"] == 1 and out[11]["reverted_30d"] == 1


def test_revert_censoring_near_window_end():
    targets = dict([_t(1, 0)])
    out, _ = P.match_reverts(targets, [], window_end=at(29 * 24 * 60))
    assert out[1]["revert_censored"] == 1


# ── hait filter ─────────────────────────────────────────────────────────────

def test_hait_filter_rules():
    assert P.hait_filter(make_pr(reviews=[("bob", 5)]))
    assert not P.hait_filter(make_pr())                                   # no review
    assert not P.hait_filter(make_pr(reviews=[("bob", 5)], additions=0, deletions=0))
    assert not P.hait_filter(make_pr(reviews=[("bob", 5)], base="feature/x"))


def _merged_rec(n, base, ok=1):
    return {"pr_number": n, "merged_at": at(n), "base_ref": base, "hait_filter": ok}


def test_main_base_fallback_only_when_name_filter_finds_nothing():
    # version branches only: the name filter keeps nothing, most frequent base wins
    recs = [_merged_rec(1, "7.x", 0), _merged_rec(2, "7.x", 0), _merged_rec(3, "6.x", 0),
            {"pr_number": 4, "merged_at": None}]
    nobase = {1: True, 2: False, 3: True}
    assert P.main_base_fallback(recs, nobase) == "7.x"
    assert [r.get("hait_filter") for r in recs[:3]] == [1, 0, 0]
    assert {r["base_rule"] for r in recs[:3]} == {"most_frequent"}
    # glpi-like: maintenance branch is more frequent than main, the name filter is kept
    recs = [_merged_rec(1, "11.0/bugfixes", 0), _merged_rec(2, "11.0/bugfixes", 0),
            _merged_rec(3, "main", 1)]
    assert P.main_base_fallback(recs, {1: True, 2: True, 3: True}) is None
    assert [r["hait_filter"] for r in recs] == [0, 0, 1]
    assert {r["base_rule"] for r in recs} == {"name"}


# ── end to end on a tiny synthetic corpus ───────────────────────────────────

def _synthetic_corpus(tmp_path):
    raw = tmp_path / "raw_v2"
    raw.mkdir()
    langs = {"o/a": ("Go", "company"), "o/b": ("Go", "community"), "o/c": ("PHP", "company"),
             "o/d": ("PHP", "community")}
    with open(raw / "repos.csv", "w") as f:
        f.write("repo,language,group\n")
        for r, (l, g) in langs.items():
            f.write(f"{r},{l},{g}\n")
    rnd = random.Random(0)
    for r in langs:
        d = raw / r.replace("/", "_")
        d.mkdir()
        with open(d / "prs.jsonl", "w") as f:
            for n in range(1, 161):
                start = n * 600
                big = rnd.random() < 0.4
                revs = [("bob", start + rnd.randint(10, 3000))]
                if big:
                    revs.append(("carol", start + 5000))
                pr = make_pr(author=rnd.choice(["u1", "u2", "u3", "dependabot"]), number=n,
                             reviews=[(a, m - start) for a, m in revs],
                             commits=[(-5, rnd.randint(1, 50), 1)] + ([(4000, 9, 9)] if big else []),
                             merged_min=rnd.randint(100, 8000))
                for k in ("createdAt",):
                    pr[k] = (T0 + timedelta(minutes=start)).strftime("%Y-%m-%dT%H:%M:%SZ")
                mm = start + rnd.randint(3100, 9000) if big else start + rnd.randint(100, 3000)
                pr["mergedAt"] = pr["closedAt"] = (T0 + timedelta(minutes=mm)).strftime("%Y-%m-%dT%H:%M:%SZ")
                for rv in pr["reviews"]["nodes"]:
                    t = datetime.strptime(rv["submittedAt"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                    rv["submittedAt"] = (t + timedelta(minutes=start)).strftime("%Y-%m-%dT%H:%M:%SZ")
                for c in pr["commits"]["nodes"]:
                    t = datetime.strptime(c["commit"]["authoredDate"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                    c["commit"]["authoredDate"] = (t + timedelta(minutes=start)).strftime("%Y-%m-%dT%H:%M:%SZ")
                if pr["author"]["login"] == "dependabot":
                    pr["author"]["__typename"] = "Bot"
                f.write(json.dumps(pr) + "\n")
    return raw


def test_end_to_end_prep_and_models(tmp_path, monkeypatch):
    raw = _synthetic_corpus(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setenv("R3_RAW_V2", str(raw))
    monkeypatch.setenv("R3_OUT", str(out))
    monkeypatch.setattr(sys, "argv", ["r3_v2_prep.py"])
    P.main()
    import pandas as pd
    t = pd.read_csv(out / "pr_table_v2.csv")
    assert len(t) == 640 and set(t.language) == {"Go", "PHP"}
    assert (t[t.author_login == "dependabot"].in_main == 0).all()
    for col in ("reverted_30d", "is_revert_pr", "revert_censored", "revert_reverted", "y_IT__main_D"):
        assert col in t.columns
    monkeypatch.setattr(M, "N_THREADS", 2)
    store = M.Store(str(out / "models"))
    tab = M.Table(str(out / "pr_table_v2.csv"))
    M.run_config(store, tab, "main", "IT", "CatBoost", ["loro", "lolo", "logo"])
    M.run_config(store, tab, "main", "IT", "LogReg", ["loro"])
    M.run_config(store, tab, "ctrl", "IT", "CatBoost", ["loro_shuffled", "lolo_randlang"], reps=1)
    M.run_config(store, tab, "main", "IT", "CatBoost", ["temporal"])
    M.prior_baseline(store, tab)
    n = store.n_fit
    assert n >= 4 * 3 + 4 + 8 + 1
    # resume: nothing is refitted
    store2 = M.Store(str(out / "models"))
    M.run_config(store2, tab, "main", "IT", "CatBoost", ["loro", "lolo", "logo"])
    assert store2.n_fit == 0
    recs = [json.loads(l) for l in open(out / "models" / "metrics.jsonl")]
    loro = [r for r in recs if r["protocol"] == "loro" and r["model"] == "CatBoost"]
    assert {r["test"] for r in loro} == {"o/a", "o/b", "o/c", "o/d"}
    z = np.load(out / "models" / "oof" / loro[0]["oof_file"])
    assert len(z["row_id"]) == loro[0]["n_test"]
    # bot-authored rows never reach the main sample
    ids = set(z["row_id"].tolist())
    assert not ids & set(t.index[t.in_main == 0])
