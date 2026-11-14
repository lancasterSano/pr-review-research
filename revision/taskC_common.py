#!/usr/bin/env python3
"""Shared helpers for HAIT 2026 revision Task C — AI-assistant participation.

READ-ONLY with respect to every existing artefact. Reads:
  - data/raw/<repo>/{merged_prs.json,open_prs.json}
  - data/processed/{all_merged_features.csv,all_open_features.csv}

Writes only into <DISSERTATION_DIR>/journal_articles/HAIT_2026/revision_results/
(see paths.py).

Scope note
----------
data/raw also contains legacy directories from an earlier pilot (angular,
discourse, kubernetes, rust, spark, ...). Only the 12 directories that back
Table 2 of the article are used here; the list is pinned explicitly so that a
stray directory cannot silently change the corpus.
"""

import csv
import json
import os
from datetime import datetime

csv.field_size_limit(10 ** 9)

from paths import DISSERTATION

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# PRR_RAW_DIR / PRR_PROCESSED_DIR point a separate clone at the data (read only).
RAW_DIR = os.environ.get("PRR_RAW_DIR") or os.path.join(ROOT, "data", "raw")
PROCESSED_DIR = os.environ.get("PRR_PROCESSED_DIR") or os.path.join(ROOT, "data", "processed")
MERGED_CSV = os.path.join(PROCESSED_DIR, "all_merged_features.csv")
OPEN_CSV = os.path.join(PROCESSED_DIR, "all_open_features.csv")
OUT_DIR = os.path.join(DISSERTATION, "journal_articles", "HAIT_2026", "revision_results")

# The 12 study repositories, as directory names under data/raw.
REPO_DIRS = [
    "angular_angular",
    "discourse_discourse",
    "django_django",
    "facebook_react",
    "grafana_grafana",
    "mattermost_mattermost",
    "microsoft_vscode",
    "nodejs_node",
    "strapi_strapi",
    "supabase_supabase",
    "vercel_next.js",
    "woocommerce_woocommerce",
]

# Row order of Table 2 in the article (repo slugs as they appear in the CSVs).
TABLE2_ORDER = [
    "woocommerce/woocommerce",
    "facebook/react",
    "microsoft/vscode",
    "vercel/next.js",
    "angular/angular",
    "supabase/supabase",
    "strapi/strapi",
    "grafana/grafana",
    "mattermost/mattermost",
    "nodejs/node",
    "discourse/discourse",
    "django/django",
]

# Concurrency budget — other agents run on this machine at the same time.
THREADS = 4


def dir_to_repo(d):
    """'woocommerce_woocommerce' -> 'woocommerce/woocommerce' (compute_features.py rule)."""
    return d.replace("_", "/", 1)


def parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def iter_raw(pr_type):
    """Yield (repo_slug, pr_dict) for every PR of the given type in the 12 study repos."""
    for d in REPO_DIRS:
        path = os.path.join(RAW_DIR, d, f"{pr_type}_prs.json")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path) as f:
            prs = json.load(f)
        repo = dir_to_repo(d)
        for pr in prs:
            yield repo, pr


def pr_participants(pr):
    """Return (review_authors, thread_comment_authors) as sets of logins.

    The PR author is *not* excluded here: an AI agent can also open a PR
    (copilot-swe-agent does), and we want the raw participation picture.
    Callers that need "reviewer other than the author" filter downstream.
    """
    review_authors = set()
    for r in pr.get("reviews", {}).get("nodes", []) or []:
        a = r.get("author") or {}
        login = a.get("login")
        if login:
            review_authors.add(login)

    thread_authors = set()
    for t in pr.get("reviewThreads", {}).get("nodes", []) or []:
        for c in (t.get("comments", {}) or {}).get("nodes", []) or []:
            a = c.get("author") or {}
            login = a.get("login")
            if login:
                thread_authors.add(login)

    return review_authors, thread_authors


# ── Classification of participant logins ────────────────────────────────────
#
# Task 1: an explicit allowlist, not substring matching. Substring matching on
# "ai", "bot", "copilot" etc. produced false positives in the preliminary scan
# (several human logins contain those substrings).
#
# Definition used by the article's BPMN AI Assistant lane:
#   "an automated participant based on a large language model that produces
#    review suggestions, summaries, or draft fixes".
# So membership requires: (i) automated account, (ii) LLM-based, (iii) its output
# enters the review comment stream. Deterministic automation (CI reporters,
# dependency updaters, coverage bots, linters, changelog/label bots) is excluded
# and tracked separately as NON_AI_BOTS.

AI_ASSISTANTS = {
    # ── Present in this corpus; every one was verified by reading sample
    #    review bodies (see taskC1_audit_login.py / taskC1_marker_scan.py).
    # GitHub Copilot code review (LLM review suggestions on the diff)
    "copilot-pull-request-reviewer": "GitHub Copilot code review",
    # GitHub Copilot coding agent (LLM agent that authors/updates changes and
    # replies in review threads)
    "copilot-swe-agent": "GitHub Copilot coding agent",
    "copilot": "GitHub Copilot (generic app account)",
    # CodeRabbit — LLM PR reviewer
    "coderabbitai": "CodeRabbit",
    "coderabbitai-bot": "CodeRabbit (alternate delivery account)",
    # Vercel Agent (a.k.a. VADE) — LLM review agent. In this corpus the `vercel`
    # app account emits ONLY agent review content inside reviews/reviewThreads
    # (1,071 of 1,072 merged PRs carry an explicit VADE- marker); the account's
    # deployment-preview messages are *issue* comments, a field we never collect.
    "vercel": "Vercel Agent (VADE) code review",
    # Graphite Diamond — LLM PR reviewer. 100% of non-empty bodies from this
    # account in the corpus are Diamond review comments.
    "graphite-app": "Graphite Diamond",
    # Claude Code GitHub app used as a reviewer (mattermost, strapi, supabase,
    # woocommerce). Verified: structured LLM review comments.
    "claude": "Claude Code (GitHub app)",
    "claude-bot": "Claude Code (GitHub app)",
    # OpenAI Codex code review connector ("💡 Codex Review" header)
    "chatgpt-codex-connector": "OpenAI Codex code review",
    # Cursor Bugbot — LLM PR reviewer
    "cursor": "Cursor Bugbot",
    "cursoragent": "Cursor agent",
    # Google Gemini Code Assist — LLM PR reviewer ("## Code Review" + priority badges)
    "gemini-code-assist": "Gemini Code Assist",
    # Greptile — LLM PR reviewer
    "greptile-apps": "Greptile",
    "greptile-apps-local": "Greptile (local app)",
    # Diffray — multi-agent LLM reviewer (diffray.ai)
    "diffray-bot": "Diffray",
    # Devin — LLM software agent
    "devin-ai-integration": "Devin",
    # ── Found by the corpus v2 audit (taskV2_login_audit.py, 40 repositories);
    #    each verified by reading sample review / thread bodies.
    # Sentry AI code review (Seer): "**Bug:** …" / "Potential bug" findings
    "sentry": "Sentry AI code review (Seer)",
    "seer-by-sentry": "Sentry AI code review (Seer)",
    # Sentry Warden — LLM review agents (security findings in prose)
    "sentry-warden": "Sentry Warden",
    # Ito — AI QA agent posting failures found in a diff run (app.ito.ai)
    "itoqa": "Ito AI QA",
    # Discourse "Patch Triage Review" — LLM reviewer that also requests changes
    "discourse-patch-triage": "Discourse Patch Triage (LLM reviewer)",
    # Hugging Face "Serge" — LLM reviewer ("🤗 **Serge** says: …")
    "sergereview": "Hugging Face Serge",
    # Datadog Bits AI code review ("BITS CODE REVIEW" badge)
    "datadoghq-integration": "Datadog Bits AI code review",
    # Qodo agent posting LLM suggestions from a user account
    "qodoai-agent": "Qodo agent",
    # LLM coding agents that author changes and answer review threads
    # ("Addressed in the latest push: …", "Agreed — applying your suggestion")
    "vs-code-engineering": "VS Code engineering agent",
    "vercel-fleet-prod": "Vercel Fleet agent",
    "vercel-fleet": "Vercel Fleet agent",
    "anthropic-code-agent": "Claude Code agent",
    "agent-sandbox-woocommerce": "coding agent (woocommerce)",
    # Accounts whose events are machine-written: structured LLM reviews or
    # thread replies ("🤖 …", JSON-wrapped bodies)
    "phillip-kruger-bot": "LLM review agent",
    "opilot-agent": "LLM coding agent",
    "perfloop-agent": "LLM coding agent",
    "joestump-agent": "LLM coding agent",
    "blut-agent": "LLM coding agent",
    # ── Not observed in this corpus, kept so a corpus refresh classifies them
    #    correctly without another manual audit.
    "sourcery-ai": "Sourcery AI",
    "codium-ai-agent": "Qodo / CodiumAI PR-Agent",
    "qodo-merge-pro": "Qodo Merge",
    "qodo-merge-pro-for-open-source": "Qodo Merge (OSS)",
    "codeant-ai": "CodeAnt AI",
    "ellipsis-dev": "Ellipsis",
    "sweep-ai": "Sweep AI",
    "korbit-ai": "Korbit AI",
    "trag-bot": "Trag",
    "pr-agent": "Qodo PR-Agent",
    "bito-ai": "Bito AI",
}

# Logins whose status could not be settled from the data. They are NOT counted
# as AI assistants in any reported figure; their total impact is stated in the
# report so a reader can see the size of the ambiguity.
UNCERTAIN = {
    # Posts one structured, LLM-styled review on woocommerce/woocommerce#55508.
    # No tool signature, no app-account naming convention, single occurrence.
    "agent-dave": "structured LLM-style review, 1 PR, provenance unverified",
    # Corpus v2: security scanner with optional AI triage; bodies look rule-based
    "aikido-pr-checks": "security scanner, AI involvement unverified",
    # Corpus v2: four empty-body reviews on next.js, nothing else to go on
    "javanio627-bot": "empty-body reviews, provenance unverified",
}

# Deliberately NOT in AI_ASSISTANTS, although they are automation:
#   codescene-delta-analysis, sonarcloud, codeql — static/heuristic analysis,
#   no language model; dependabot/renovate — template-driven dependency bumps.
NON_AI_BOTS = {
    # dependency / release automation
    "dependabot": "dependency updates",
    "dependabot-preview": "dependency updates",
    "renovate": "dependency updates",
    "renovate-bot": "dependency updates",
    "github-actions": "CI workflow reporter",
    "codecov": "coverage reporter",
    "codecov-commenter": "coverage reporter",
    "codecov-io": "coverage reporter",
    "sonarcloud": "static analysis reporter",
    "sonarqubecloud": "static analysis reporter",
    "netlify": "deployment previews",
    "changeset-bot": "changelog automation",
    "socket-security": "supply-chain scanner",
    "codescene-delta-analysis": "static analysis reporter",
    "mattermost-build": "release automation",
    "mattermost-buildjunkie": "release automation",
    "discourse-translator-bot": "translation automation",
    "greenkeeper": "dependency updates",
    "snyk-bot": "security scanner",
    "gitpod-io": "dev-environment automation",
    "kodiakhq": "merge queue",
    "mergify": "merge queue",
    "allcontributors": "contributor list automation",
    "stale": "staleness automation",
    "github-advanced-security": "CodeQL reporter",
    "semantic-release-bot": "release automation",
    "size-limit-action": "bundle size reporter",
    "next-stats-bot": "bundle size reporter",
    # ── Found by the corpus v2 audit (taskV2_login_audit.py). Accounts that
    #    GitHub reports as Bot, or user accounts whose events are all automated.
    # auto-approvers: approve dependency / translation PRs, empty bodies
    "ciliumbot": "auto-approver (cilium)",
    "servo-bot": "auto-approver (servo)",
    "grafana-pr-approver": "auto-approver (grafana)",
    "woocommercebot": "service account (woocommerce)",
    "keycloak-github-bot": "flaky-test reporter (keycloak)",
    # merge queues, required-review and policy apps
    "github-merge-queue": "merge queue",
    "pullapprove": "review policy app",
    "gh-policy-bot-grafana-ops": "policy app",
    "microsoft-github-policy-service": "policy app",
    "maintainer-s-little-helper": "triage automation (cilium)",
    # dependency, release, backport and sync automation
    "cilium-renovate": "dependency updates",
    "renovate-sh-app": "dependency updates",
    "hf-dependantbot-rollout": "dependency updates",
    "cilium-release-bot": "release automation",
    "vercel-release-bot": "release automation",
    "supabase-cli-releaser": "release automation",
    "internal-releases": "release automation",
    "nextjs-bot": "release automation",
    "next-js-bot": "release automation",
    "scikit-learn-bot": "release automation",
    "eslintbot": "release automation",
    "decidim-bot": "translation / release automation",
    "giteabot": "backport automation",
    "angular-robot": "repository automation",
    "nodejs-github-bot": "repository automation",
    "grafana-pr-automation": "repository automation",
    "grafana-go-workspace-bot": "repository automation",
    "grafana-writer": "repository automation",
    "alerting-team": "repository automation",
    "auto-committer": "repository automation",
    "unified-ci-app": "CI app",
    "blacksmith-sh": "CI runners",
    "airlock-confluentinc": "repository automation",
    "terraform-aws-provider": "repository automation",
    "oss-core-libraries-dashboard": "repository automation",
    "hashicorp-copywrite": "licence header automation",
    "supabase-supabase-autofixer": "content sync automation",
    "sync-l10n-for-pdf-js": "translation sync",
    "meta-codesync": "code sync",
    "brew-commit-app": "repository automation",
    "imgbot": "image optimisation",
    "trunk-io": "merge queue / CI",
    "ecc-tools": "repository automation",
    "linear-code": "issue tracker integration",
    "mm-prodsec-bot": "security automation (mattermost)",
    "jlaportebot": "personal automation account, no bodies",
    # static analysis and security scanners (rule-based findings)
    "github-code-quality": "static analysis reporter",
    "snyk-io": "security scanner",
    "semgrep-code-grafana": "static analysis reporter",
    "semgrep-code-glpi-alex-test": "static analysis reporter",
    "semgrep-code-getsentry": "static analysis reporter",
    "wiz-inc-0e7a25329d": "security scanner",
    "orca-security-us": "security scanner",
    "ibm-mend-app": "dependency scanner",
    "qltysh": "static analysis reporter",
    "hf-security-analysis": "security scanner",
}


def _key(login):
    k = login.lower()
    return k[:-5] if k.endswith("[bot]") else k


def classify_login(login):
    """Return one of: 'ai_assistant', 'other_bot', 'uncertain', 'human'.

    Classification is by explicit allowlist / blocklist only. Anything not on a
    list is reported as 'human' but is also flagged for manual audit by the
    caller when its name pattern is bot-like, so no login is silently guessed.
    """
    k = _key(login)
    if k in AI_ASSISTANTS:
        return "ai_assistant"
    if k in NON_AI_BOTS:
        return "other_bot"
    if k in UNCERTAIN:
        return "uncertain"
    return "human"


def classify_account(login, typename=None):
    """classify_login() plus the account type GitHub reports (corpus v2).

    An account GitHub marks as Bot (a GitHub App) that is on no list is
    automation, not a person: 'other_bot'. The lists still decide first, so a
    listed LLM reviewer stays 'ai_assistant'. Bots running under ordinary user
    accounts (ciliumbot, servo-bot, ...) are caught only by the lists, which is
    why taskV2_login_audit.py has to be rerun on every corpus refresh. A missing
    login (deleted "ghost" account) is human, as in participant_class().
    """
    if not login:
        return "human"
    cls = classify_login(login)
    if cls == "human" and (typename == "Bot" or login.lower().endswith("[bot]")):
        return "other_bot"
    return cls


def author_group(login, typename=None):
    """Who prepared the change: 'human', 'ai_agent', 'other_bot' or 'uncertain'.

    LLM coding agents (copilot-swe-agent, ...) author changes that people
    review for real, so their PRs stay in the analysis as a separate group.
    """
    cls = classify_account(login, typename)
    return "ai_agent" if cls == "ai_assistant" else cls


def in_main_population(login, typename=None):
    """Author decision 29.10.2026: the main analysis of corpus v2 excludes PRs
    whose author is other automation (dependency updates, releases, backports,
    syncs): auto-approvers and batch merges distort time and iterations there.
    PRs by people and by LLM coding agents stay; the variant with bot-authored
    PRs is reported as a sensitivity check.
    """
    return author_group(login, typename) in ("human", "ai_agent")


def ai_tool_name(login):
    return AI_ASSISTANTS.get(_key(login))


def is_ai(login):
    return _key(login) in AI_ASSISTANTS


def load_processed(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(x, nd=3):
    if x is None:
        return "n/a"
    try:
        if x != x:  # NaN
            return "n/a"
    except TypeError:
        pass
    return f"{x:.{nd}f}"
