import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import taskE1_common as E  # noqa: E402


def _pr(threads, author="alice", author_type="User", created="2026-01-01T00:00:00Z", commits=(), reviews=()):
    return {
        "number": 7, "createdAt": created, "additions": 100, "deletions": 10, "changedFiles": 3,
        "author": {"login": author, "__typename": author_type},
        "commits": {"nodes": [{"commit": {"authoredDate": d, "message": m}} for d, m in commits]},
        "reviews": {"nodes": [{"author": {"login": l}, "state": s, "submittedAt": created, "body": ""} for l, s in reviews]},
        "reviewThreads": {"totalCount": len(threads), "nodes": threads},
    }


def _thread(first_login, first_body, first_at, replies=(), path="src/a.py", resolved=False):
    nodes = [{"author": {"login": first_login}, "body": first_body, "createdAt": first_at}]
    for login, body, at in replies:
        nodes.append({"author": {"login": login}, "body": body, "createdAt": at})
    return {"isResolved": resolved, "path": path, "comments": {"nodes": nodes}}


def test_severity_levels():
    assert E.severity("_⚠️ Potential issue_\n\nNull dereference here") == 2
    assert E.severity("![high](https://x/high.svg) Priority") == 2
    assert E.severity("_🧹 Nitpick_ rename the variable") == 1
    assert E.severity("Consider extracting a helper") == 0
    assert E.severity("") == 0


def test_apply_commit_regex():
    assert E.APPLY_RX.search("Apply suggestions from code review\n\nCo-authored-by: Copilot <x@y>")
    assert E.APPLY_RX.search("Update src/a.ts\n\nCo-authored-by: coderabbitai[bot] <1@2>")
    assert not E.APPLY_RX.search("fix typo in README")


def test_thread_record_features_and_labels():
    t = _thread("copilot-pull-request-reviewer[bot]", "```suggestion\nx = 1\n```\nThis looks like a bug",
                "2026-01-01T10:00:00Z", replies=[("alice", "Done, fixed", "2026-01-01T12:00:00Z")],
                path="src/app/main.ts", resolved=True)
    pr = _pr([t], commits=(("2025-12-31T00:00:00Z", "initial"), ("2026-01-01T09:30:00Z", "wip"),
                           ("2026-01-01T13:00:00Z", "Apply suggestions from code review\n\nCo-authored-by: Copilot <c@github.com>")))
    rec = E.thread_record("acme/repo", pr, t, 0, 1, {"src/app/main.ts"}, E.pr_commits(pr))
    assert rec["tool"] == "GitHub Copilot code review" and rec["ext"] == "ts"
    assert rec["has_suggestion"] == 1 and rec["severity"] == 2
    assert rec["hours_since_open"] == 10.0 and rec["hours_since_trigger"] == 0.5  # last commit before = 09:30
    assert rec["commits_before"] == 1 and rec["commits_after"] == 1  # the 2025 commit predates the PR
    assert rec["reacted"] == 1 and rec["reply_by_author"] == 1
    assert rec["apply_commit_after"] == 1 and rec["sugg_applied"] == 1
    assert rec["human_same_file"] == 1 and rec["is_resolved"] == 1 and rec["author_is_bot"] == 0
    for outcome in ("is_resolved", "reacted", "reply_by_author", "commits_after", "apply_commit_after", "sugg_applied", "human_same_file"):
        assert outcome not in E.FEATURES


def test_thread_record_bot_author_and_bot_replies():
    t = _thread("coderabbitai[bot]", "```suggestion\ny```", "2026-02-01T10:00:00Z",
                replies=[("coderabbitai[bot]", "Understood", "2026-02-01T10:01:00Z"), ("dependabot[bot]", "bump", "2026-02-01T10:02:00Z")])
    pr = _pr([t], author="copilot-swe-agent[bot]", author_type="Bot",
             commits=(("2026-02-01T11:00:00Z", "Update y\n\nCo-authored-by: Copilot <c@github.com>"),))
    rec = E.thread_record("acme/repo", pr, t, 0, 1, set(), E.pr_commits(pr))
    assert rec["reacted"] == 0 and rec["author_is_bot"] == 1
    assert rec["apply_commit_after"] == 1 and rec["sugg_applied"] == 0  # agent-authored PR: not counted as acceptance


def test_iter_ai_threads_orders_within_pr_and_uses_human_paths(monkeypatch):
    t_human = _thread("bob", "please rename", "2026-03-01T09:00:00Z", path="a.py")
    t_ai2 = _thread("coderabbitai[bot]", "second", "2026-03-01T11:00:00Z", path="b.py")
    t_ai1 = _thread("coderabbitai[bot]", "first", "2026-03-01T10:00:00Z", path="a.py")
    monkeypatch.setattr(E, "iter_raw", lambda kind: iter([("acme/repo", _pr([t_human, t_ai2, t_ai1]))]))
    recs = list(E.iter_ai_threads())
    assert [r["ai_idx"] for r in recs] == [0, 1] and all(r["n_ai_threads"] == 2 for r in recs)
    assert [r["human_same_file"] for r in recs] == [1, 0]


def test_iter_ai_reviews_keeps_only_assistant_reviews(monkeypatch):
    pr = _pr([], reviews=[("coderabbitai[bot]", "APPROVED"), ("bob", "APPROVED"), ("copilot-pull-request-reviewer[bot]", "COMMENTED")])
    monkeypatch.setattr(E, "iter_raw", lambda kind: iter([("acme/repo", pr)]))
    recs = list(E.iter_ai_reviews())
    assert [(r["tool"], r["state"]) for r in recs] == [("CodeRabbit", "APPROVED"), ("GitHub Copilot code review", "COMMENTED")]
    assert all(r["repo"] == "acme/repo" and r["pr_number"] == 7 for r in recs)


def test_chrono_split_and_ap():
    df = pd.DataFrame({"created_at": ["2026-01-03", "2026-01-01", "2026-01-04", "2026-01-02"], "v": [3, 1, 4, 2]})
    hist, test = E.chrono_split(df, test_frac=0.25)
    assert list(hist["v"]) == [1, 2, 3] and list(test["v"]) == [4]
    assert np.isnan(E.ap(np.array([0, 0]), np.array([0.1, 0.2])))
    assert E.ap(np.array([1, 0]), np.array([0.9, 0.1])) == 1.0
