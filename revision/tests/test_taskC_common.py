"""Account classification shared by all analyses (corpus v1 and v2).

Each test names the production change that would make it fail.
Run from the project root:  .venv/bin/python -m pytest revision/tests -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taskC_common as c  # noqa: E402


@pytest.mark.parametrize("login, typename, expected", [
    # unlisted GitHub App is automation, not a person (rule dropped -> 'human')
    ("some-new-app", "Bot", "other_bot"),
    ("some-new-app[bot]", None, "other_bot"),
    # the lists decide first: a listed LLM reviewer stays an assistant even
    # though GitHub reports it as Bot (order swapped -> 'other_bot')
    ("copilot-pull-request-reviewer", "Bot", "ai_assistant"),
    ("sentry", "Bot", "ai_assistant"),
    ("aikido-pr-checks", "Bot", "uncertain"),
    # bots on user accounts are caught by the list, not by the type
    ("ciliumbot", "User", "other_bot"),
    # people stay people; a ghost account is a person
    ("octocat", "User", "human"),
    ("octocat", None, "human"),
    (None, None, "human"),
])
def test_classify_account(login, typename, expected):
    assert c.classify_account(login, typename) == expected


def test_classify_account_without_typename_matches_classify_login():
    # corpus v1 callers pass no type: classify_account must agree with classify_login
    # (the account lists themselves were extended in v1.3.0)
    for login in ["dependabot", "coderabbitai", "agent-dave", "octocat"]:
        assert c.classify_account(login) == c.classify_login(login)


@pytest.mark.parametrize("login, typename, group, main", [
    ("octocat", "User", "human", True),
    # LLM coding agent: reviewed by people, stays in the analysis
    ("copilot-swe-agent", "Bot", "ai_agent", True),
    # dependency / release automation: excluded from the main analysis
    ("dependabot", "Bot", "other_bot", False),
    ("cilium-renovate", "Bot", "other_bot", False),
    ("decidim-bot", "User", "other_bot", False),
    ("some-new-app", "Bot", "other_bot", False),
    ("agent-dave", "User", "uncertain", False),
])
def test_author_group_and_main_population(login, typename, group, main):
    assert c.author_group(login, typename) == group
    assert c.in_main_population(login, typename) is main


def test_lists_do_not_overlap():
    a, b, u = set(c.AI_ASSISTANTS), set(c.NON_AI_BOTS), set(c.UNCERTAIN)
    assert not (a & b) and not (a & u) and not (b & u)
    assert all(k == k.lower() for k in a | b | u)
