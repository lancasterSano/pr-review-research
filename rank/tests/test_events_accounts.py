"""Account filter of the ranking pipeline: legacy regex vs shared classification.

Fails if RANK_ACCOUNTS=shared stops using the audited lists, or if the default
changes (the AAIT 2026 numbers depend on the regex)."""
from rank import events


def test_default_is_the_paper_regex():
    assert events.ACCOUNTS in ("regex", "shared")
    # the paper's regex lets these through as people (finding of 29.10.2026)
    for login in ("claude", "github-advanced-security", "grafana-pr-approver", "snyk-io"):
        assert not events.is_bot(login, accounts="regex")
    assert events.is_bot("dependabot[bot]", accounts="regex")


def test_shared_classification_drops_assistants_and_bots():
    for login in ("claude", "chatgpt-codex-connector", "github-advanced-security",
                  "grafana-pr-approver", "snyk-io", "semgrep-code-grafana", "angular-builds",
                  "copilot-pull-request-reviewer", "dependabot[bot]"):
        assert events.is_bot(login, accounts="shared"), login
    assert events.is_bot("some-unlisted-app", "Bot", accounts="shared")
    for login in ("abbot", "alice", "robotics-dev"):
        assert not events.is_bot(login, "User", accounts="shared"), login
    assert not events.is_bot("", accounts="shared")


def test_shared_mode_changes_positives(monkeypatch):
    monkeypatch.setattr(events, "ACCOUNTS", "shared")
    pr = {"number": 1, "author": {"login": "a"}, "createdAt": "2025-01-01T00:00:00Z",
          "reviews": {"nodes": [{"author": {"login": "claude", "__typename": "Bot"},
                                 "submittedAt": "2025-01-01T01:00:00Z"},
                                {"author": {"login": "bob", "__typename": "User"},
                                 "submittedAt": "2025-01-01T02:00:00Z"}]},
          "reviewThreads": {"nodes": []}}
    assert events.positives(pr) == {"bob"}
    monkeypatch.setattr(events, "ACCOUNTS", "regex")
    assert events.positives(pr) == {"claude", "bob"}
