# pr-review-research

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23534421.svg)](https://doi.org/10.5281/zenodo.23534421)

Collector of pull request data from GitHub, built for the analysis in a PhD
dissertation on the efficiency of the code review process (National University
"Odesa Polytechnic").

The repository contains the code that downloads closed pull requests with their
review history from public GitHub repositories, a script that turns the raw
data into per-PR features, the reviewer ranking experiments, and a manifest of
the collected corpus. It does not
contain the raw data itself (see [Personal data](#personal-data-and-github-terms)).

## Contents

| Path | What it does |
|---|---|
| `fetch/fetch_prs_v2.py` | Corpus v2 collector: closed and merged PRs of one repository created in a fixed window, via the GraphQL API. |
| `fetch/run_corpus_v2.py` | Runs the collector over the 40 corpus repositories in order, writes repository metadata (`repos.csv`). Resumable. |
| `fetch/verify_corpus_v2.py` | Checks one collected repository: counts against GitHub search, duplicates, gaps, field fill rates, completeness of nested lists. |
| `fetch/corpus_v2_status.py` | Progress, liveness and ETA of a running collection. |
| `fetch/fetch_prs.py` | Frozen collector of the earlier 12-repository corpus (HAIT 2026 paper). Kept unchanged for reproducibility. |
| `enrich/compute_features.py` | Builds per-PR features (PR, author history, project context, targets) from the frozen corpus. |
| `enrich/review_rounds.py` | Counts review rounds of a PR; used by `compute_features.py`. |
| `rank/` | Reviewer ranking experiments of the AAIT 2026 paper: candidate pools, history features, learned ranker, baselines, constrained re-ranking. See [Reviewer ranking](#reviewer-ranking). |
| `fetch/tests/` | Offline tests for the corpus v2 collector (no network). |
| `rank/tests/` | Offline tests for the ranking pipeline (synthetic data). |
| `corpus/manifest_v2.csv` | Corpus v2 manifest: repository, primary language, window and PR counts. |

## What is collected

`fetch_prs_v2.py` requests closed pull requests (states `MERGED` and `CLOSED`)
created between `2024-10-01T00:00:00Z` and `2026-08-31T23:59:59Z` (UTC,
inclusive) and stores each PR node as returned by the GraphQL API. The list
below is complete for corpus v2; the query text is in the constants named in
brackets in `fetch/fetch_prs_v2.py`. "Actor" means `login` and `__typename`
(the account type, for example `User`, `Bot` or `Mannequin`).

- **PR** (`PR_SCALARS`): `id`, `number`, `title`, `body`, `url`, `createdAt`,
  `updatedAt`, `mergedAt`, `closedAt`, `state`, `isDraft`, `additions`,
  `deletions`, `changedFiles`, `baseRefName`, `headRefName`, `baseRefOid`,
  `headRefOid`, `isCrossRepository`, `authorAssociation`,
  `totalCommentsCount`; `author` (actor), `mergedBy` (actor),
  `milestone.title`.
- **Labels** (`NESTED["labels"]`): `name`.
- **Commits** (`COMMIT_NODE`): `oid`, `message`, `authoredDate`,
  `committedDate`, `additions`, `deletions`, `changedFilesIfAvailable`,
  `parents.totalCount`, `author.user.login` (login only).
- **Reviews** (`REVIEW_NODE`): `id`, `author` (actor), `authorAssociation`,
  `authorCanPushToRepository`, `state`, `submittedAt`, `createdAt`, `body`,
  `commit.oid`.
- **Review threads** (`THREAD_NODE`): `id`, `isResolved`, `isOutdated`,
  `path`, `line`, `originalLine`, `resolvedBy.login` (login only); comments
  (`COMMENT_NODE`): `author` (actor), `authorAssociation`, `body`,
  `createdAt`, `originalCommit.oid`, `pullRequestReview.id`.
- **Review requests** (`REQUESTED_REVIEWER`): `requestedReviewer` with
  `__typename` and `login` (user, bot, mannequin) or `slug` (team).
- **Timeline** (`TIMELINE_NODE`, event types in `TIMELINE_TYPES`):
  `__typename` and `createdAt` of each event, plus: the actor for ready for
  review, convert to draft, review requested/removed (with the requested
  reviewer), head ref force-pushed (with `beforeCommit.oid` and
  `afterCommit.oid`), merged (with `commit.oid`), closed, reopened,
  auto-merge enabled/disabled, added to/removed from the merge queue, review
  dismissed (with `review.id`); for issue comments `author` (actor) and
  `authorAssociation`, without the comment text; for PR commits
  `commit.oid`, `authoredDate`, `committedDate` (no `createdAt`).
- **CI**, only for PRs created on or after `2025-10-01` (GitHub keeps check
  suites for about 13 months):
  - `CI_FIELDS`: the last 30 commits (`commits(last: 30)`, with
    `totalCount`), for each `oid`, `committedDate`, the first 20 check suites
    (`createdAt`, `updatedAt`, `status`, `conclusion`, `app.slug`, with
    `totalCount`) and the commit status (`state`, and `context`, `state`,
    `createdAt` of each context);
  - `CI_HEAD_FIELDS`: for the head commit, the first 20 check suites
    (`createdAt`, `status`, `conclusion`, `app.slug`,
    `workflowRun.workflow.name`) and the first 50 check runs of each suite
    (`name`, `status`, `conclusion`, `startedAt`, `completedAt`).
- **`_meta`** (added by the collector): `repo`, `fetched_at`, `collector`,
  `ci_fetched`, `ci_window`, and, when a fallback was used,
  `commit_stats_per_commit`, `commit_stats_missing`,
  `thread_review_link_missing`.

Labels, commits, reviews, review threads and their comments, review requests
and timeline events are paginated to the end with follow-up queries
(`totalCount` is stored with each list). **CI lists are not paginated:** they
are capped at the last 30 commits, 20 check suites per commit and 50 check
runs per head-commit suite. The stored `totalCount` values show when a cap was
reached.

`run_corpus_v2.py` also stores repository metadata in `repo_meta.json`
(`REPO_META_QUERY`): `nameWithOwner`, `isArchived`, `isMirror`, `isFork`,
`createdAt`, `stargazerCount`, owner account type, primary language,
language byte shares, and GitHub search counts of closed PRs in the window.
The query also requests `owner.login`; the stored `nameWithOwner` contains
it.

### Fields that concern people

The raw data contains information about the people who took part in the
reviews:

- GitHub login and account type (`__typename`) of: PR authors, users who
  merged a PR, review authors, review-thread comment authors, issue comment
  authors (timeline), actors of timeline events, and requested reviewers;
- GitHub login only (no account type) of: commit authors and users who
  resolved a review thread;
- team slugs of requested team reviewers;
- `authorAssociation` (for example `MEMBER`, `CONTRIBUTOR`,
  `FIRST_TIME_CONTRIBUTOR`) of PR authors, review authors, review-thread
  comment authors and issue comment authors;
- `authorCanPushToRepository` of review authors;
- the repository owner's login (`owner.login`, part of `nameWithOwner`),
  which is a person when the repository belongs to a user account;
- free text written by people: PR titles and bodies, review bodies,
  review-thread comment bodies, commit messages, branch names, milestone
  titles and labels;
- timestamps of every action, which show when people work.

The collector does not request e-mail addresses, names or profile data.
The frozen collector `fetch_prs.py` stores a subset of the same kinds of data
(logins of PR authors with account type, logins of review and comment authors
and requested users, titles, bodies, review and comment texts, commit
messages, branch names); its query text is in `MERGED_PR_QUERY` and
`OPEN_PR_QUERY`.

## Installation

Tested with Python 3.9 and 3.12.

```bash
git clone https://github.com/lancasterSano/pr-review-research.git
cd pr-review-research
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

## GitHub token

The collector needs a GitHub token. A fine-grained personal access token is
enough:

1. GitHub → Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → Generate new token.
2. Repository access: **Public repositories (read-only)**. No other
   permissions are needed.

Pass the token in one of these ways (checked in this order):

```bash
export GITHUB_TOKEN=<your token>                 # environment variable
echo 'GITHUB_TOKEN=<your token>' > .env          # or a .env file in the project root (git-ignored)
gh auth login                                    # or the GitHub CLI keyring as a fallback
```

The token is never printed or written to the output.

## Collecting one repository

```bash
python fetch/fetch_prs_v2.py --repo eslint/eslint
```

This collects every closed PR of the repository in the window. There is no
limit option: the window is part of the corpus definition
(`WINDOW_START` and `WINDOW_END` in `fetch/fetch_prs_v2.py`). For scale,
`eslint/eslint` has about 1,350 PRs in the window and took about 20 minutes
and 1,200 GraphQL points.

Output, under `data/raw_v2/<owner>_<name>/`:

| File | Content |
|---|---|
| `prs.jsonl` | One PR per line (GraphQL shape plus `_meta`). |
| `checkpoint.json` | Cursor, counters and status; a rerun resumes from here. |
| `fetch.log` | Timestamped progress log. |

`data/raw_v2/progress.csv` holds one status row per repository. Other
options: `--raw-dir` (output folder) and `--page-size`.

A rerun resumes only if the checkpoint was written with the same window, CI
cut-off, PR states and collector version; otherwise it stops with an error,
and the data has to go to a new `--raw-dir`.

The whole corpus (40 repositories, several days of API budget):

```bash
python fetch/run_corpus_v2.py --list                          # show the queue
python fetch/run_corpus_v2.py --only eslint/eslint            # a subset
python fetch/run_corpus_v2.py                                 # everything
python fetch/corpus_v2_status.py                              # progress and ETA
```

`run_corpus_v2.py` first fetches metadata for all 40 repositories
(`repo_meta.json`, `repos.csv`). Stop it with `touch data/raw_v2/STOP`; run the
same command again to resume.

## Checking the result

```bash
python fetch/verify_corpus_v2.py --repo eslint/eslint             # uses GitHub search
python fetch/verify_corpus_v2.py --repo eslint/eslint --no-search # offline checks only
```

It writes `data/raw_v2/<owner>_<name>/verify.json` and prints the summary:
the number of collected PRs against GitHub search, duplicates, PRs outside the
window, the largest gap between PRs, PRs per month, field fill rates, and
whether every nested list has all its pages. A list that is absent (null after
a partial GraphQL response) is not counted as incomplete; the manifest column
`partial_errors` shows how many partial responses were accepted. Small
differences from the search count are possible: search counts are computed by
GitHub separately and change over time (per-repository numbers are in
`corpus/manifest_v2.csv`). The corpus is also a snapshot of the PRs that were
closed by the time each repository's collection finished (`collected_at`), so
PRs closed later appear in search but not in the data. For example, in
`huggingface/transformers` about 600 open PRs were closed by the maintainers on
2026-10-20, after the collection and before the verification.

## The two collectors

| | `fetch/fetch_prs.py` | `fetch/fetch_prs_v2.py` |
|---|---|---|
| Corpus | 12 repositories, HAIT 2026 paper | 40 repositories, corpus v2 |
| Status | frozen, kept for reproducibility | current |
| PRs | merged PRs, and open PRs as a snapshot | merged and closed-unmerged PRs in a fixed creation window |
| Limit | `--limit N` per repository (checked after each page of 50) | none (fixed window) |
| Nested lists | first page only (fixed sizes) | paginated to the end, except CI lists (capped) |
| Timeline, CI | no | yes |
| API access | `gh api graphql` (GitHub CLI login) | HTTPS with a token |
| Output | `data/raw/<owner>_<name>/{merged,open}_prs.json` | `data/raw_v2/<owner>_<name>/prs.jsonl` |

## Features

`compute_features.py` works on the frozen corpus only: it reads
`data/raw/<owner>_<name>/{merged,open}_prs.json`, written by `fetch_prs.py`,
not `data/raw_v2/`. Collect that data first. `fetch_prs.py` calls
`gh api graphql`, so it needs the GitHub CLI logged in (`gh auth login`, or
`GH_TOKEN` set):

```bash
python fetch/fetch_prs.py --repo django/django          # writes data/raw/django_django/
python enrich/compute_features.py --repo django_django  # one repository
python enrich/compute_features.py                       # every folder in data/raw
```

Without `--repo` the script lists `data/raw/` and fails if that folder does not
exist. It writes `data/processed/<repo>_{merged,open}_features.csv` for each
repository and PR type that has records after filtering, and the combined
`all_merged_features.csv` / `all_open_features.csv` only when there are
merged / open records.

## Reviewer ranking

`rank/` holds the experiments of the paper "Ranking pull request reviewers by
expertise, workload, and expected response time" (Applied Aspects of
Information Technology, 2026). Like `compute_features.py`, it reads the frozen
corpus: `data/raw/<owner>_<name>/merged_prs.json` for the 12 repositories in
`rank.build.REPOS`. Collect them with `fetch/fetch_prs.py` first.

```bash
python -m rank.build             # pair matrix (PR x candidate) per repository -> rank/output/
python -m rank.run_experiments   # dataset profile, baselines, rankers, leave-one-repo-out -> rank/output/results/
python -m rank.run_e4            # trade-off: weighted sum and constrained re-ranking -> rank/output/results/
python -m rank.run_r1            # review round: most-familiar baseline, paired realized latency -> rank/output/results/r1_*.csv
```

`run_r1` retrains nothing: it reads `rank/output/results/test_scored.parquet`
written by `run_experiments`. It adds the "Most familiar" baseline (candidates
ordered by how many of the author's earlier pull requests they reviewed, ties
broken by 90-day activity) and compares the realized response time of the
relevance-only ranker and the latency-constrained re-ranking on the same pull
requests.

| Module | What it does |
|---|---|
| `events.py` | Reviewer events and actual reviewers of a PR; bot filter. |
| `history.py` | Reviewer and author history strictly before PR creation. |
| `build.py` | Candidate pools (180 days), STRICT and CONTENT features, 90-day warm-up. |
| `split.py` | Chronological 60/20/20 split per repository. |
| `ranker.py` | LightGBM lambdarank model. |
| `evaluate.py`, `metrics.py` | MRR, Hit@k, NDCG@5, Gini of first choices, bootstrap intervals. |
| `tradeoff.py` | Weighted-sum penalty on workload and expected latency. |
| `constrained.py` | Re-ranking within a relevance tolerance by expected latency or exposure; rules inspired by prior work. |
| `selection.py` | Choice of configurations on the validation split. |

The paper used a dump collected in April 2026. A new collection returns newer
pull requests, so the numbers will differ from the published ones; the
pipeline and the protocol stay the same. `rank/output/` is not committed.

## Corpus manifest

`corpus/manifest_v2.csv` lists the 40 repositories of corpus v2 as collected.
It contains no PR content and no logins.

| Column | Meaning |
|---|---|
| `repo` | `owner/name` |
| `language` | primary language by bytes of code |
| `window_start`, `window_end` | creation window of the collected PRs |
| `prs_collected`, `prs_merged`, `prs_closed_unmerged` | PR counts in the collected data |
| `github_search_closed` | GitHub search count of closed PRs in the window at verification time (empty for `facebook/react`, which search does not index) |
| `collector_version` | collector version stored in the checkpoint |
| `collected_at` | when the collection of the repository finished (checkpoint, UTC) |
| `verified_at` | when `verify.json` was written (UTC) |
| `partial_errors` | partial GraphQL responses accepted during collection (counted from `fetch.log`) |
| `notes` | explanation of notable differences, if any |

In total: 298,222 PRs, 234,035 of them merged, in 8 languages.

## Personal data and GitHub terms

The raw data is built from public GitHub content, but it includes personal
data (see [Fields that concern people](#fields-that-concern-people)).

- Do not republish the raw data (`data/`), and do not publish logins or
  quotes of comments that identify a person.
- Publish aggregate results only.
- Use the data in line with the
  [GitHub Acceptable Use Policies](https://docs.github.com/en/site-policy/acceptable-use-policies/github-acceptable-use-policies)
  (information usage restrictions) and the
  [GitHub General Privacy Statement](https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement).

This repository therefore ships only code and the aggregate manifest. To
obtain the data, run the collector.

## Citation

If you use this code, please cite it with the metadata in
[`CITATION.cff`](CITATION.cff) (GitHub shows it as "Cite this repository").

The code is archived on Zenodo:

- all versions (resolves to the latest): [10.5281/zenodo.23534421](https://doi.org/10.5281/zenodo.23534421)
- v1.1.0, the version used in the AAIT 2026 paper: [10.5281/zenodo.23534422](https://doi.org/10.5281/zenodo.23534422)

## Contact

Oleksandr Aratovskyi, alex.aratovskiy@stud.op.edu.ua

## License

[MIT](LICENSE)
