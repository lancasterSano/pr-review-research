#!/usr/bin/env python3
"""
Fetch merged and open PRs from GitHub GraphQL API for 12 repositories.

Usage:
    python fetch_prs.py                        # fetch all repos
    python fetch_prs.py --repo django/django   # single repo
    python fetch_prs.py --open-only            # only open PRs (DP4)
    python fetch_prs.py --merged-only          # only merged PRs
    python fetch_prs.py --limit 500            # limit PRs per repo

Output:
    data/raw/{owner}_{repo}/merged_prs.json
    data/raw/{owner}_{repo}/open_prs.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

# ── Repository list (the 12 repositories of the HAIT 2026 corpus) ──────────

REPOS = [
    # Tier 1
    ("woocommerce", "woocommerce"),
    ("mattermost", "mattermost"),
    ("strapi", "strapi"),
    # Tier 2
    ("microsoft", "vscode"),
    ("discourse", "discourse"),
    ("vercel", "next.js"),
    ("grafana", "grafana"),
    ("supabase", "supabase"),
    ("nodejs", "node"),
    # Tier 3
    ("django", "django"),
    ("facebook", "react"),
    ("angular", "angular"),
]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

# ── GraphQL queries ─────────────────────────────────────────────────────────

MERGED_PR_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(first: 50, after: $cursor, states: [MERGED], orderBy: {field: CREATED_AT, direction: DESC}) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        title
        body
        createdAt
        mergedAt
        closedAt
        state
        additions
        deletions
        changedFiles
        baseRefName
        headRefName
        isDraft
        author { login __typename }
        labels(first: 10) { nodes { name } }
        commits(first: 100) {
          totalCount
          nodes {
            commit {
              message
              authoredDate
            }
          }
        }
        reviews(first: 50) {
          totalCount
          nodes {
            author { login }
            state
            submittedAt
            body
          }
        }
        reviewThreads(first: 50) {
          totalCount
          nodes {
            isResolved
            path
            comments(first: 20) {
              nodes {
                author { login }
                body
                createdAt
              }
            }
          }
        }
        reviewRequests(first: 10) {
          totalCount
          nodes {
            requestedReviewer {
              ... on User { login }
            }
          }
        }
      }
    }
  }
}
"""

OPEN_PR_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(first: 50, after: $cursor, states: [OPEN], orderBy: {field: CREATED_AT, direction: DESC}) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        title
        body
        createdAt
        updatedAt
        state
        additions
        deletions
        changedFiles
        baseRefName
        headRefName
        isDraft
        author { login __typename }
        labels(first: 10) { nodes { name } }
        commits(first: 100) {
          totalCount
          nodes {
            commit {
              message
              authoredDate
            }
          }
        }
        reviews(first: 50) {
          totalCount
          nodes {
            author { login }
            state
            submittedAt
            body
          }
        }
        reviewRequests(first: 10) {
          totalCount
          nodes {
            requestedReviewer {
              ... on User { login }
            }
          }
        }
      }
    }
  }
}
"""

# ── GraphQL execution ──────────────────────────────────────────────────────

def run_graphql(query, owner, name, cursor=None):
    """Execute a GraphQL query via gh CLI."""
    cmd = [
        "gh", "api", "graphql",
        "-f", f"query={query}",
        "-F", f"owner={owner}",
        "-F", f"name={name}",
    ]
    if cursor:
        cmd += ["-F", f"cursor={cursor}"]

    for attempt in range(3):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            print(f"    Timeout on attempt {attempt + 1}, retrying...")
            time.sleep(5)
            continue

        if result.returncode != 0:
            stderr = result.stderr[:500]
            if "rate limit" in stderr.lower() or "API rate" in stderr:
                wait = 60 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if "502" in stderr or "503" in stderr:
                print(f"    Server error, retrying in 10s...")
                time.sleep(10)
                continue
            print(f"    ERROR: {stderr}", file=sys.stderr)
            return None

        data = json.loads(result.stdout)
        if "errors" in data:
            errors = data["errors"]
            if any("rate" in str(e).lower() for e in errors):
                wait = 60 * (attempt + 1)
                print(f"    Rate limited (GraphQL), waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"    GraphQL errors: {json.dumps(errors[:2], indent=2)}", file=sys.stderr)
            return None

        return data

    return None

# ── Checkpointing ──────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path):
    """Load checkpoint: returns (cursor, count_fetched)."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            cp = json.load(f)
        return cp.get("cursor"), cp.get("count", 0)
    return None, 0


def save_checkpoint(checkpoint_path, cursor, count):
    """Save checkpoint for resumable fetching."""
    with open(checkpoint_path, "w") as f:
        json.dump({"cursor": cursor, "count": count, "updated_at": datetime.utcnow().isoformat()}, f)


def clear_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

# ── Fetching ───────────────────────────────────────────────────────────────

def fetch_prs(owner, name, query, limit=None):
    """Fetch PRs with pagination and checkpointing.

    Args:
        owner: GitHub org/user
        name: repo name
        query: GraphQL query (MERGED or OPEN)
        limit: max PRs to fetch (None = all)

    Returns:
        list of PR node dicts
    """
    repo_dir = os.path.join(DATA_DIR, f"{owner}_{name}")
    os.makedirs(repo_dir, exist_ok=True)

    pr_type = "merged" if "MERGED" in query else "open"
    checkpoint_path = os.path.join(repo_dir, f".checkpoint_{pr_type}.json")
    output_path = os.path.join(repo_dir, f"{pr_type}_prs.json")

    # Load existing data + checkpoint
    existing_prs = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing_prs = json.load(f)

    cursor, count_fetched = load_checkpoint(checkpoint_path)
    if count_fetched > 0:
        print(f"    Resuming from checkpoint: {count_fetched} PRs fetched, cursor={cursor[:20] if cursor else 'None'}...")

    all_prs = list(existing_prs)
    seen_numbers = {pr["number"] for pr in all_prs}
    page = 0

    while True:
        if limit and len(all_prs) >= limit:
            print(f"    Reached limit of {limit} PRs")
            break

        page += 1
        print(f"    Page {page} (have {len(all_prs)} PRs)...")

        data = run_graphql(query, owner, name, cursor)
        if data is None:
            print(f"    Failed to fetch page {page}, saving progress...")
            break

        pr_connection = data["data"]["repository"]["pullRequests"]
        nodes = pr_connection["nodes"]

        if not nodes:
            break

        new_count = 0
        for pr in nodes:
            if pr["number"] not in seen_numbers:
                all_prs.append(pr)
                seen_numbers.add(pr["number"])
                new_count += 1

        if new_count == 0 and not pr_connection["pageInfo"]["hasNextPage"]:
            break

        cursor = pr_connection["pageInfo"]["endCursor"]
        save_checkpoint(checkpoint_path, cursor, len(all_prs))

        # Save intermediate results every 5 pages
        if page % 5 == 0:
            with open(output_path, "w") as f:
                json.dump(all_prs, f)
            print(f"    Intermediate save: {len(all_prs)} PRs")

        if not pr_connection["pageInfo"]["hasNextPage"]:
            break

        # Small delay to stay within rate limits
        time.sleep(0.5)

    # Final save
    with open(output_path, "w") as f:
        json.dump(all_prs, f)
    clear_checkpoint(checkpoint_path)

    total_available = data["data"]["repository"]["pullRequests"]["totalCount"] if data else "?"
    print(f"    Done: {len(all_prs)} PRs saved (total available: {total_available})")
    return all_prs


def fetch_repo(owner, name, fetch_merged=True, fetch_open=True, limit=None):
    """Fetch merged and/or open PRs for a single repo."""
    print(f"\n{'=' * 60}")
    print(f"  {owner}/{name}")
    print(f"{'=' * 60}")

    if fetch_merged:
        print(f"  Fetching MERGED PRs...")
        fetch_prs(owner, name, MERGED_PR_QUERY, limit=limit)

    if fetch_open:
        print(f"  Fetching OPEN PRs (for DP4 staleness)...")
        fetch_prs(owner, name, OPEN_PR_QUERY, limit=limit)


def main():
    parser = argparse.ArgumentParser(description="Fetch PRs from GitHub GraphQL API")
    parser.add_argument("--repo", type=str, help="Single repo: owner/name")
    parser.add_argument("--merged-only", action="store_true", help="Only fetch merged PRs")
    parser.add_argument("--open-only", action="store_true", help="Only fetch open PRs")
    parser.add_argument("--limit", type=int, default=None, help="Max PRs per repo (default: all)")
    args = parser.parse_args()

    fetch_merged = not args.open_only
    fetch_open = not args.merged_only

    if args.repo:
        parts = args.repo.split("/")
        if len(parts) != 2:
            print("ERROR: --repo must be in owner/name format", file=sys.stderr)
            sys.exit(1)
        repos = [(parts[0], parts[1])]
    else:
        repos = REPOS

    print(f"Fetching PRs for {len(repos)} repo(s)")
    print(f"  Merged: {fetch_merged}, Open: {fetch_open}, Limit: {args.limit or 'all'}")
    print(f"  Output: {DATA_DIR}/")

    for owner, name in repos:
        fetch_repo(owner, name, fetch_merged, fetch_open, args.limit)

    print(f"\nAll done! Data saved to {DATA_DIR}/")


if __name__ == "__main__":
    main()
