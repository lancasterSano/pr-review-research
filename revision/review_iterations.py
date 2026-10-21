"""Review iterations of a pull request: the single implementation for the dissertation.

Rule (author decision 14.10.2026, literal reading of HAIT Definition 2)
----------------------------------------------------------------------
An iteration is opened by the first counted review event. A NEW iteration starts
only when the author pushed commits after a counted review AND a further counted
review followed. Commits after the last counted review that are followed by the
merge (no new review) do NOT add an iteration.

    RULE_LITERAL ("literal", default)  the rule above (dissertation, ITIA-2026)
    RULE_CODE    ("code")              enrich/review_rounds.compute_review_rounds:
                                       additionally, a merged PR with commits after
                                       the last counted review gets one more
                                       iteration (published HAIT / IKT numbers)

Under both rules:
  * review events are submissions in `reviews` by non-authors (self-reviews are
    excluded) accepted by `include(login)` (e.g. humans only);
  * commits authored at or before PR creation are ignored;
  * reviews and commits are ordered by time; on ties a review comes before a
    commit (stable sort, reviews listed first), as in compute_review_rounds.

With include=include_all and rule="code" the result equals compute_review_rounds
for every PR (checked on all 64,820 merged PRs by taskF1 stage prep).
"""
from datetime import datetime

RULE_LITERAL = "literal"
RULE_CODE = "code"
RULES = (RULE_LITERAL, RULE_CODE)


def parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def login_of(node):
    return ((node or {}).get("author") or {}).get("login", "") or ""


def include_all(_login):
    return True


def reviewer_events(pr, include):
    """[(time, login, state)] of review submissions by non-authors accepted by `include`."""
    author = login_of(pr)
    out = []
    for r in (pr.get("reviews") or {}).get("nodes") or []:
        who = login_of(r)
        if who == author:
            continue  # self-review, excluded as in compute_review_rounds
        if not include(who):
            continue
        t = parse_dt((r or {}).get("submittedAt"))
        if t:
            out.append((t, who, (r or {}).get("state")))
    out.sort(key=lambda e: e[0])
    return out


def iteration_trace(pr, include):
    """(n_literal, trailing_rework) for one PR.

    n_literal       iterations under the literal rule;
    trailing_rework True if commits follow the last counted review (the case in
                    which the code rule adds one iteration to a merged PR).
    """
    created = parse_dt(pr.get("createdAt"))
    reviews = [(t, 0) for t, _, _ in reviewer_events(pr, include)]
    if not reviews:
        return 0, False
    commits = []
    for c in (pr.get("commits") or {}).get("nodes") or []:
        authored = parse_dt(((c or {}).get("commit") or {}).get("authoredDate"))
        if authored and created and authored > created:
            commits.append((authored, 1))
    events = reviews + commits
    events.sort(key=lambda e: e[0])  # stable: reviews before commits on ties

    n = 0
    seen_review = False
    rework = False
    for _, kind in events:
        if kind == 0:  # review
            if rework:
                n += 1
                rework = False
            if n == 0:
                n = 1
            seen_review = True
        elif seen_review:  # commit after a counted review
            rework = True
    return n, rework


def count_iterations(pr, include=include_all, rule=RULE_LITERAL):
    """Number of review iterations of `pr` counting review events accepted by `include`."""
    if rule not in RULES:
        raise ValueError(f"unknown iteration rule: {rule!r}")
    n, rework = iteration_trace(pr, include)
    if rule == RULE_CODE and rework and pr.get("state") == "MERGED":
        n += 1
    return n
