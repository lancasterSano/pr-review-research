#!/usr/bin/env python3
"""
Compute features for PR review prediction from raw JSON data.

Groups:
  1. PR-level features (from single PR node)
  2. Developer history features (expanding window over prior PRs)
  3. Project context features (snapshot at PR creation time)

Usage:
    python compute_features.py                           # all repos
    python compute_features.py --repo django_django            # single repo
    python compute_features.py --type merged             # only merged PRs
    python compute_features.py --type open               # only open PRs

Output:
    data/processed/{repo_name}_merged_features.csv
    data/processed/{repo_name}_open_features.csv
    data/processed/all_merged_features.csv
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from review_rounds import compute_review_rounds, parse_dt

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")

# ── Patterns for feature detection ──────────────────────────────────────────

TEST_PATTERNS = re.compile(
    r"(test[s_/]|spec[s_/]|__tests__|\.test\.|\.spec\.|_test\.)",
    re.IGNORECASE,
)
DOC_PATTERNS = re.compile(
    r"(docs?[/\\]|\.md$|readme|changelog|\.rst$|\.txt$)",
    re.IGNORECASE,
)
CI_PATTERNS = re.compile(
    r"(\.github/workflows/|\.circleci/|Jenkinsfile|\.travis\.yml|\.gitlab-ci|Makefile|Dockerfile|docker-compose)",
    re.IGNORECASE,
)
IMAGE_PATTERN = re.compile(r"!\[.*?\]\(.*?\)|<img\s", re.IGNORECASE)
CODE_BLOCK_PATTERN = re.compile(r"```")
ISSUE_REF_PATTERN = re.compile(r"(?:closes?|fixes?|resolves?)\s+#\d+", re.IGNORECASE)

# ── Group 1: PR-level features ──────────────────────────────────────────────

def compute_pr_features(pr):
    """Extract Group 1 features from a single PR node."""
    body = pr.get("body") or ""
    created = parse_dt(pr.get("createdAt"))
    merged = parse_dt(pr.get("mergedAt"))

    # Commit info
    commits = pr.get("commits", {}).get("nodes", [])
    commit_messages = [c.get("commit", {}).get("message", "") for c in commits]

    # File type diversity from commit messages and head/base ref
    # (GraphQL doesn't return file paths directly — we use changedFiles count
    # and infer from review threads paths)
    thread_paths = []
    for t in pr.get("reviewThreads", {}).get("nodes", []):
        path = t.get("path")
        if path:
            thread_paths.append(path)

    extensions = set()
    for path in thread_paths:
        ext = os.path.splitext(path)[1]
        if ext:
            extensions.add(ext.lower())

    has_tests = any(TEST_PATTERNS.search(p) for p in thread_paths) or \
                any(TEST_PATTERNS.search(m) for m in commit_messages)
    has_docs = any(DOC_PATTERNS.search(p) for p in thread_paths) or \
               any(DOC_PATTERNS.search(m) for m in commit_messages)
    touches_ci = any(CI_PATTERNS.search(p) for p in thread_paths) or \
                 any(CI_PATTERNS.search(m) for m in commit_messages)

    # Labels
    labels = [l["name"] for l in pr.get("labels", {}).get("nodes", [])]

    # Review info
    reviews = pr.get("reviews", {}).get("nodes", [])
    author_login = (pr.get("author") or {}).get("login", "")

    # Time to first review (hours)
    time_to_first_review = None
    if created and reviews:
        non_self_reviews = [
            r for r in reviews
            if (r.get("author") or {}).get("login", "") != author_login
        ]
        if non_self_reviews:
            review_times = [parse_dt(r.get("submittedAt")) for r in non_self_reviews]
            review_times = [t for t in review_times if t]
            if review_times:
                first_review = min(review_times)
                time_to_first_review = round((first_review - created).total_seconds() / 3600, 2)

    # Time to merge (hours)
    time_to_merge = None
    if created and merged:
        time_to_merge = round((merged - created).total_seconds() / 3600, 2)

    # Review rounds
    review_rounds = compute_review_rounds(pr)

    # Bot detection
    author_typename = (pr.get("author") or {}).get("__typename", "User")
    is_bot = author_typename == "Bot" or (author_login and author_login.endswith("[bot]"))

    # Staleness (for open PRs — DP4)
    updated_at = parse_dt(pr.get("updatedAt"))
    is_stale = None
    if pr.get("state") == "OPEN" and updated_at:
        now = datetime.now(timezone.utc)
        days_since_update = (now - updated_at).days
        is_stale = days_since_update > 30

    return {
        "repo": "",  # filled by caller
        "pr_number": pr["number"],
        "state": pr.get("state"),
        "author_login": author_login,
        "author_typename": author_typename,
        "is_bot": is_bot,
        "created_at": pr.get("createdAt", ""),
        "merged_at": pr.get("mergedAt", ""),

        # Group 1: PR-level
        "lines_added": pr.get("additions", 0),
        "lines_deleted": pr.get("deletions", 0),
        "files_changed": pr.get("changedFiles", 0),
        "num_commits": pr.get("commits", {}).get("totalCount", 0),
        "has_tests": has_tests,
        "has_docs": has_docs,
        "touches_ci": touches_ci,
        "num_file_types": len(extensions) if extensions else None,
        "desc_length": len(body),
        "title_length": len(pr.get("title", "")),
        "has_screenshots": bool(IMAGE_PATTERN.search(body)),
        "has_code_blocks": bool(CODE_BLOCK_PATTERN.search(body)),
        "references_issues": bool(ISSUE_REF_PATTERN.search(body)),
        "is_draft": pr.get("isDraft", False),
        "has_milestone": bool(pr.get("milestone")),
        "label_count": len(labels),
        "labels": "|".join(labels),
        "day_of_week": created.weekday() if created else None,
        "hour_of_day": created.hour if created else None,

        # Review metrics
        "review_count": pr.get("reviews", {}).get("totalCount", 0),
        "review_rounds": review_rounds,
        "review_thread_count": pr.get("reviewThreads", {}).get("totalCount", 0),
        "resolved_threads": sum(
            1 for t in pr.get("reviewThreads", {}).get("nodes", []) if t.get("isResolved")
        ),
        "review_request_count": pr.get("reviewRequests", {}).get("totalCount", 0),
        "reviewer_logins": "|".join(sorted(set(
            (r.get("author") or {}).get("login", "")
            for r in reviews
            if (r.get("author") or {}).get("login") and
               (r.get("author") or {}).get("login") != author_login
        ))),
        "num_reviewers": len(set(
            (r.get("author") or {}).get("login", "")
            for r in reviews
            if (r.get("author") or {}).get("login") and
               (r.get("author") or {}).get("login") != author_login
        )),
        "has_approved": any(r.get("state") == "APPROVED" for r in reviews),
        "has_changes_requested": any(r.get("state") == "CHANGES_REQUESTED" for r in reviews),

        # Target variables
        "time_to_first_review_hours": time_to_first_review,
        "time_to_merge_hours": time_to_merge,

        # DP4: staleness (open PRs only)
        "is_stale": is_stale,
    }


# ── Group 2: Developer history (expanding window) ───────────────────────────

def compute_developer_history(features_list):
    """Add Group 2 features using expanding window over chronologically sorted PRs.

    Modifies features_list in place, sorted by created_at.
    """
    # Sort by creation time
    features_list.sort(key=lambda f: f["created_at"] or "")

    # Per-author history
    author_history = {}  # login -> list of past PR records

    for f in features_list:
        login = f["author_login"]
        if not login:
            f.update({
                "author_merge_rate": None,
                "author_avg_review_time": None,
                "author_avg_review_rounds": None,
                "author_commit_frequency": None,
                "author_tenure_days": None,
                "author_pr_count": 0,
            })
            continue

        history = author_history.get(login, [])
        created = parse_dt(f["created_at"])

        if history:
            # Merge rate: fraction of past PRs that were merged
            merged_count = sum(1 for h in history if h["state"] == "MERGED")
            f["author_merge_rate"] = round(merged_count / len(history), 4)

            # Average review time (median of past time_to_first_review)
            review_times = [h["time_to_first_review_hours"] for h in history
                            if h["time_to_first_review_hours"] is not None]
            if review_times:
                review_times_sorted = sorted(review_times)
                mid = len(review_times_sorted) // 2
                if len(review_times_sorted) % 2 == 0:
                    f["author_avg_review_time"] = round(
                        (review_times_sorted[mid - 1] + review_times_sorted[mid]) / 2, 2
                    )
                else:
                    f["author_avg_review_time"] = review_times_sorted[mid]
            else:
                f["author_avg_review_time"] = None

            # Average review rounds
            rounds = [h["review_rounds"] for h in history if h["review_rounds"] is not None]
            f["author_avg_review_rounds"] = round(sum(rounds) / len(rounds), 2) if rounds else None

            # Commit frequency (90-day sliding window)
            if created:
                window_start = created - timedelta(days=90)
                recent_prs = [h for h in history if h["created_at"] and
                              parse_dt(h["created_at"]) and
                              parse_dt(h["created_at"]) >= window_start]
                total_commits = sum(h.get("num_commits", 0) for h in recent_prs)
                f["author_commit_frequency"] = total_commits
            else:
                f["author_commit_frequency"] = None

            # Tenure: days since first PR
            first_pr_date = parse_dt(history[0]["created_at"])
            if created and first_pr_date:
                f["author_tenure_days"] = (created - first_pr_date).days
            else:
                f["author_tenure_days"] = None

            f["author_pr_count"] = len(history)
        else:
            # First PR by this author
            f["author_merge_rate"] = None
            f["author_avg_review_time"] = None
            f["author_avg_review_rounds"] = None
            f["author_commit_frequency"] = 0
            f["author_tenure_days"] = 0
            f["author_pr_count"] = 0

        # Add current PR to history for future PRs
        if login not in author_history:
            author_history[login] = []
        author_history[login].append(f)


# ── Group 3: Project context features ───────────────────────────────────────

def compute_project_context(features_list):
    """Add Group 3 features: project-level context at PR creation time.

    Modifies features_list in place.
    Must be called AFTER sorting by created_at.
    """
    # features_list is already sorted by created_at from Group 2 computation

    for i, f in enumerate(features_list):
        created = parse_dt(f["created_at"])
        if not created:
            f.update({
                "open_pr_count": None,
                "merge_velocity_7d": None,
                "merge_velocity_30d": None,
            })
            continue

        # Count PRs created before this one that weren't yet merged/closed
        # (approximation: created before, merged after or not yet merged)
        open_count = 0
        merged_7d = 0
        merged_30d = 0

        for j in range(i):
            prior = features_list[j]
            prior_created = parse_dt(prior["created_at"])
            prior_merged = parse_dt(prior.get("merged_at"))

            if not prior_created:
                continue

            # Open at time of current PR creation
            if prior.get("state") == "MERGED" and prior_merged:
                if prior_merged > created:
                    open_count += 1
                # Merge velocity
                days_ago = (created - prior_merged).days
                if days_ago <= 7:
                    merged_7d += 1
                if days_ago <= 30:
                    merged_30d += 1
            elif prior.get("state") == "OPEN":
                open_count += 1

        f["open_pr_count"] = open_count
        f["merge_velocity_7d"] = merged_7d
        f["merge_velocity_30d"] = merged_30d


# ── Filtering of merged PRs ─────────────────────────────────────────────────

def should_include_merged(pr, features):
    """Apply inclusion/exclusion criteria for merged PRs."""
    if features["time_to_merge_hours"] is not None and features["time_to_merge_hours"] < 1 / 60:
        return False  # merged in < 1 minute (auto-merge)
    if features["time_to_merge_hours"] is not None and features["time_to_merge_hours"] > 365 * 24:
        return False  # open > 365 days
    if features["lines_added"] == 0 and features["lines_deleted"] == 0:
        return False  # no code changes
    if features["review_count"] == 0:
        return False  # no reviews
    base = pr.get("baseRefName", "")
    allowed_bases = ("main", "master", "trunk", "develop", "release", "canary", "dev")
    if base not in allowed_bases:
        # Allow common main/dev branches; skip feature-to-feature PRs
        if not any(p in base for p in ("main", "master", "trunk", "release", "canary")):
            return False
    return True


# ── Main pipeline ───────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "repo", "pr_number", "state", "author_login", "author_typename", "is_bot",
    "created_at", "merged_at",
    # Group 1
    "lines_added", "lines_deleted", "files_changed", "num_commits",
    "has_tests", "has_docs", "touches_ci", "num_file_types",
    "desc_length", "title_length", "has_screenshots", "has_code_blocks",
    "references_issues", "is_draft", "has_milestone", "label_count", "labels",
    "day_of_week", "hour_of_day",
    # Review metrics
    "review_count", "review_rounds", "review_thread_count", "resolved_threads",
    "review_request_count", "reviewer_logins", "num_reviewers",
    "has_approved", "has_changes_requested",
    # Targets
    "time_to_first_review_hours", "time_to_merge_hours", "is_stale",
    # Group 2: developer history
    "author_merge_rate", "author_avg_review_time", "author_avg_review_rounds",
    "author_commit_frequency", "author_tenure_days", "author_pr_count",
    # Group 3: project context
    "open_pr_count", "merge_velocity_7d", "merge_velocity_30d",
]


def process_repo(repo_name, pr_type="merged"):
    """Process a single repo's raw JSON into features CSV.

    Args:
        repo_name: e.g. "django_django"
        pr_type: "merged" or "open"

    Returns:
        list of feature dicts
    """
    raw_path = os.path.join(RAW_DIR, repo_name, f"{pr_type}_prs.json")
    if not os.path.exists(raw_path):
        print(f"  Skipping {repo_name}/{pr_type}: {raw_path} not found")
        return []

    with open(raw_path) as f:
        prs = json.load(f)

    print(f"  {repo_name}/{pr_type}: {len(prs)} raw PRs")

    # Compute Group 1 features
    features_list = []
    for pr in prs:
        features = compute_pr_features(pr)
        features["repo"] = repo_name.replace("_", "/", 1)

        # Apply filters for merged PRs
        if pr_type == "merged":
            if not should_include_merged(pr, features):
                continue

        features_list.append(features)

    print(f"    After filtering: {len(features_list)} PRs")

    if not features_list:
        return []

    # Compute Group 2 (developer history) — expanding window
    compute_developer_history(features_list)

    # Compute Group 3 (project context)
    compute_project_context(features_list)

    # Save per-repo CSV
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DIR, f"{repo_name}_{pr_type}_features.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(features_list)
    print(f"    Saved {len(features_list)} records to {output_path}")

    return features_list


def main():
    parser = argparse.ArgumentParser(description="Compute features from raw PR JSON")
    parser.add_argument("--repo", type=str, help="Single repo directory name (e.g. django_django)")
    parser.add_argument("--type", choices=["merged", "open", "both"], default="both",
                        help="PR type to process (default: both)")
    args = parser.parse_args()

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Discover repos
    if args.repo:
        repo_dirs = [args.repo]
    else:
        repo_dirs = sorted([
            d for d in os.listdir(RAW_DIR)
            if os.path.isdir(os.path.join(RAW_DIR, d)) and not d.startswith(".")
        ])

    pr_types = []
    if args.type in ("merged", "both"):
        pr_types.append("merged")
    if args.type in ("open", "both"):
        pr_types.append("open")

    print(f"Processing {len(repo_dirs)} repo(s), types: {pr_types}")

    all_merged = []
    all_open = []

    for repo_name in repo_dirs:
        for pr_type in pr_types:
            features = process_repo(repo_name, pr_type)
            if pr_type == "merged":
                all_merged.extend(features)
            else:
                all_open.extend(features)

    # Save combined CSVs
    for name, data in [("all_merged_features.csv", all_merged), ("all_open_features.csv", all_open)]:
        if data:
            output_path = os.path.join(PROCESSED_DIR, name)
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(data)
            print(f"\nCombined: {len(data)} records -> {output_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  Merged PRs: {len(all_merged)}")
    print(f"  Open PRs:   {len(all_open)}")
    if all_merged:
        multi_round = sum(1 for f in all_merged if f["review_rounds"] >= 2)
        print(f"  Multi-round (2+): {multi_round} ({100 * multi_round / len(all_merged):.1f}%)")
        bot_prs = sum(1 for f in all_merged if f["is_bot"])
        print(f"  Bot PRs: {bot_prs} ({100 * bot_prs / len(all_merged):.1f}%)")
    if all_open:
        stale = sum(1 for f in all_open if f.get("is_stale"))
        print(f"  Stale open PRs (>30d): {stale} ({100 * stale / len(all_open):.1f}%)")


if __name__ == "__main__":
    main()
