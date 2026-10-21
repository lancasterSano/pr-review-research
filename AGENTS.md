# AGENTS.md

Guidance for coding agents working in this repository.

## Structure

- `fetch/fetch_prs_v2.py`: corpus v2 collector (GraphQL, fixed window, resumable).
- `fetch/run_corpus_v2.py`: runs the collector over the 40 corpus repositories.
- `fetch/verify_corpus_v2.py`, `fetch/corpus_v2_status.py`: checks and progress.
- `fetch/fetch_prs.py`: frozen collector of the earlier 12-repository corpus.
- `enrich/compute_features.py`, `enrich/review_rounds.py`: per-PR features from `data/raw/`.
- `rank/`: reviewer ranking experiments (AAIT 2026 paper) on `data/raw/`;
  outputs go to `rank/output/` (not committed).
- `fetch/tests/`, `rank/tests/`: offline tests (no network).
- `corpus/manifest_v2.csv`: aggregate corpus manifest.

## Commands

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Tests must not call the network. Use the fake clients in `fetch/tests/`.

## Rules

- Do not change `fetch/fetch_prs.py` or `enrich/review_rounds.py`. Published
  results depend on them. Fix bugs in new code paths instead.
- Do not change the window, CI cut-off, states or `COLLECTOR_VERSION` in
  `fetch/fetch_prs_v2.py` without a new output folder: existing checkpoints
  refuse to resume with different corpus parameters.
- Keep the field lists in `README.md` ("What is collected", "Fields that
  concern people") in step with the GraphQL queries.
- Never commit `data/`, `.env`, or any collected PR data.
- Never print, log or write the GitHub token. Read it only through
  `load_token()`.
