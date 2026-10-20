"""
Compute review rounds (the implementation behind the published HAIT 2026 results).

A "review round" is one cycle of:
  1. Reviewer (not author) submits a review event (any state: APPROVED, CHANGES_REQUESTED, COMMENTED)
  2. Author pushes one or more new commits after that review
  3. Another review event occurs

Counting:
  - Round 0 = no reviews from others
  - Round 1 = at least one review, but no commits pushed between review events
  - Round 2+ = commits pushed between review events, causing additional review cycles

Edge cases:
  - Multiple reviews before a push count within the same round
  - Commits before any review are ignored (initial submission)
  - Self-reviews are excluded
  - Commits authored before PR creation are excluded (rebase, cherry-pick)
"""

from datetime import datetime


def parse_dt(s):
    """Parse ISO 8601 datetime string."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def compute_review_rounds(pr):
    """Compute review round count for a single PR node.

    Args:
        pr: dict — raw PR node from GraphQL

    Returns:
        int — number of review rounds (0, 1, 2, ...)
    """
    author_login = (pr.get("author") or {}).get("login", "")
    created_at = parse_dt(pr.get("createdAt"))

    # Collect review events (excluding self-reviews)
    reviews = pr.get("reviews", {}).get("nodes", [])
    review_events = []
    for r in reviews:
        reviewer = (r.get("author") or {}).get("login", "")
        if reviewer == author_login:
            continue  # skip self-reviews
        submitted = parse_dt(r.get("submittedAt"))
        if submitted:
            review_events.append({"type": "review", "time": submitted, "state": r.get("state")})

    if not review_events:
        return 0  # Round 0: no reviews from others

    # Collect commit events (excluding commits before PR creation)
    commits = pr.get("commits", {}).get("nodes", [])
    commit_events = []
    for c in commits:
        commit_data = c.get("commit", {})
        authored = parse_dt(commit_data.get("authoredDate"))
        if authored and created_at and authored > created_at:
            commit_events.append({"type": "commit", "time": authored})

    # Merge and sort chronologically
    all_events = review_events + commit_events
    all_events.sort(key=lambda e: e["time"])

    # Count rounds: transitions from review → commit → review
    rounds = 0
    last_review_time = None
    commit_after_review = False

    for event in all_events:
        if event["type"] == "review":
            if commit_after_review:
                rounds += 1
                commit_after_review = False
            if rounds == 0:
                rounds = 1  # First review = round 1
            last_review_time = event["time"]
        elif event["type"] == "commit":
            if last_review_time is not None:
                commit_after_review = True

    # If there were commits after the last review but no final review yet,
    # the round is still "in progress" — count it for merged PRs
    if commit_after_review and pr.get("state") == "MERGED":
        rounds += 1

    return rounds


def compute_review_rounds_batch(prs):
    """Compute review rounds for a list of PR nodes.

    Returns:
        list of (pr_number, review_rounds) tuples
    """
    return [(pr["number"], compute_review_rounds(pr)) for pr in prs]
