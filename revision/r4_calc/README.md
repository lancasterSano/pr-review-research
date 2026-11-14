# Chapter 4 calculations

Scripts for the chapter 4 checks of the chapter 2 premises and chapter 3 models.
Outputs go to `revision/output/r4/<name>/` (ignored by git). Run from the repository root:

```bash
PY=python   # interpreter with requirements.txt
```

## P2: share of review threads at the model trigger (`r4_p2_trigger.py`)

Input (read only): `$R2_DIR/v2/r2_threads_v2.csv` (from `revision/r2/r2_raw_v2.py`) and
`$R2_DIR/r2_numbers.json` (from `revision/r2/make_tables_r2.py`) for the reference values.

```bash
$PY revision/r4_calc/r4_p2_trigger.py   # revision/output/r4/p2_trigger/{numbers.json, by_repo.csv}
```

Pooled shares of threads started within 1 h of a trigger, as in `make_tables_r2.reactions()`
(assistants over the 23 repositories with >= 100 assistant threads, humans and bots over all).
Cluster bootstrap over repositories, B = 10 000, seed 20261114, percentile 95 % intervals.
The script asserts that the point estimates equal `react.total.trig_pct`, `react.trig_h_pct`,
`react.trig_bot_pct` and `react.trig_all_pct`. Sensitivity variants: assistants over all
repositories (`sens_all`), the same 17 repositories for all groups (`sens_common`), unweighted
mean of per-repository shares with >= 20 threads (`sens_repomean`), trigger windows of 5 min, 15 min, 30 min,
2 h and 3 h (`sens_window.h0083` ... `h3`), without the repository with most assistant
threads (`sens_no_top_ai`) or most bot threads (`sens_no_top_bot`); medians of hours from
the trigger by group (`median_h.*`).

## R4 without overlap between P_tr training and the ranking test PRs (`r4_no_overlap.py`)

The chapter 3 model P_tr for R4 (`r3_report_models.block_r4`) is trained on all 40 repositories,
which include the 12 ranking repositories. This script runs the same `block_r4` (CatBoost,
`t0_v1compat`, train-median imputation, seed `r3_v2_models.SEED`, `R3_THREADS=6` as in the
02.11 rerun) on a table whose training rows leave out the 12 repositories (28 remain), and the
same `r3_report_r4.main` on its predictions. Thresholds are not recomputed:
`revision/output/r3_b40d/report/details/thresholds_by_repo.csv` (k = tr) is copied.

```bash
$PY revision/r4_calc/r4_no_overlap.py [--repro-old] [--skip-fit]
# revision/output/r4/no_overlap/{numbers.json, per_pr.csv, model_info.json, checks.json, model/, old_check/}
```

Checks (`checks.json`; the script stops if the first fails): the thresholds file equals
`r3_report.rule_thresholds` on the raw LORO output of P_tr (class weights `Balanced`, no prior
correction, no calibration, `MIN_FIRES = 20`); the old R4 numbers are reproduced from the old
predictions with the current thresholds file; the main-sample firing share `r4.*.trigger_pct` is
reproduced from that file; `--repro-old` refits the 40-repository model and compares its
predictions with the saved ones. Also reported: descriptive AUC of the new and old P_tr on the
evaluated ranking test PRs (label `y_TFR__main_D` of the v1 table, history >= 30) and how many of
those PRs were training rows of the old model (by repo + pr_number).

## P5: assistant effect under two event definitions (`r4_p5_definition.py`)

Does the event definition (human events only vs events of all accounts) change the conclusion
about PRs with and without an assistant review? Within-repository r_rb come from the chapter 2
code in `revision/r2/`, imported: `make_tables_r2.load()` (table `r3_b40d`) and `r2_ai_compare_v2.compare()`
(2025-2026 PRs, group `u_ai`, `MIN_GROUP = 20`); the script asserts its r_rb equal `compare()`.
Characteristics: iterations (`n_it_human` vs `n_it_all_lit`) and time to first review
(`t_r_h_D` vs `t_r_all_h_D`, PRs with a human review, as in `compare()`).
Pooled effect: mean r_rb over repositories with >= 20 PRs in both groups under both definitions;
cluster bootstrap over repositories, B = 10 000, seed 20261114, percentile 95 %; no Holm.
Per-repository changes use `compare.boot_ci` (2 000 draws within groups, seed 20261115).

```bash
$PY revision/r4_calc/r4_p5_definition.py   # revision/output/r4/p5_definition/{numbers.json, by_repo.csv}
```

## P4: rollback rule of the feedback loop on the corpus history (`r4_p4_feedback.py`)

Would the feedback loop give the chapter 2 rollback rule (second condition ybar < y_min,
appendix B.4) enough labelled examples to decide, had it run on the corpus history? A
feasibility check, not a pilot. Input (read only): `r2_threads_v2.csv` (assistant-started
threads, 2025-2026 PRs), `r2_numbers.json` (the script asserts 69 352 threads and 22 105 PRs =
`react.all_threads`, `react.all_prs`), `revision/output/r3_b40d/v2/pr_table_v2.csv` (merged_at)
and raw `prs.jsonl` for `isOutdated` and `resolvedBy` (matched to the CSV by position within
the PR and checked by `createdAt`; cached in `raw_thread_flags.csv`, `--refresh` redoes it).

```bash
$PY revision/r4_calc/r4_p4_feedback.py [--refresh]
# revision/output/r4/p4_feedback/{numbers.json, by_repo_quarter.csv, by_repo.csv, base_by_repo.csv, raw_thread_flags*.csv}
```

Label (chapter 2, 2.4.4, events only): y = 1 if the suggestion was applied or
there was a revision and a human resolved the thread (`resolvedBy` by `participant_class`);
y = 0 if a human resolved it without a revision; null otherwise. What is observed as the
revision (key prefix): `obs_line` (main) = the commented line changed (`isOutdated`);
`obs_any` = a later commit (`commits_after`, by authoredDate) or `isOutdated`; `obs_commit` = a
later commit only (misses amend + force-push); `sens_bots_line` = obs_line with resolutions by
any account. Baseline `base.<obs>.*`: the same labels for threads started by human reviewers
and deterministic bots (second raw pass, cached in `raw_thread_flags_base.csv`;
`base_by_repo.csv`), repo-level comparisons over repositories with >= 50 labelled threads.
Check rule: quarters (of the PR merge) stand in for release cycles; cumulative examples; a check
once n >= n_f = 20 and new examples arrived; Wilson interval at level 1 - 0.05 * 2^-i for the
i-th check with n / deff (Rao-Scott by PR, floored at 1); states rollback / keep / undecided;
stop at the first decision. Sensitivity in `numbers.json`: no stopping (`nostop`), deff = 1
(`nodeff`). No random step; seed 20261114 for the record.
Also y_min = 0.6 and 0.7 for the three obs_* variants (`ymin06`, `ymin07`), decisions taken on
n < 50 or n / deff < 20, and repositories with >= 2 assistants or a change of the dominant one.

## P4: ratio rollback rule of the dissertation (`r4_p4_ratio.py`)

The rule the dissertation reports (chapter 2, B.4; chapter 4, P4). `r4_p4_feedback.py` above is
the earlier feasibility replay with an absolute threshold; it is kept because it builds the raw
caches this script reads (`raw_thread_flags.csv`, `raw_thread_flags_base.csv`), so run it first.

```bash
$PY revision/r4_calc/r4_p4_feedback.py
$PY revision/r4_calc/r4_p4_ratio.py   # revision/output/r4/p4_ratio/{numbers.json, by_repo.csv, by_repo_quarter.csv, by_repo_all_tools.csv, power.csv}
```

Label: 1 if the suggestion was applied or the commented line changed before the merge, 0 if
the PR was merged and the line did not change. Ratio R of the assistant share to the share of
human reviewers in PRs without an assistant in the same quarters; MOVER-R interval from Wilson
intervals with n / deff (deff by PR author); level 1 - 0.05 / K with K = 8 quarters; a check
needs labelled examples in >= n_f = 20 PRs per group. Unit: one assistant in one repository.
Rollback if the upper bound < kappa = 0.6 (called zeta in the dissertation text), keep if the
lower bound >= kappa. Sensitivity keys: kappa 0.5 and 0.7, control in the same PRs, deff by PR,
the repository as the unit. Seed 20261114 for the power simulation.

## R4 without overlap: thresholds on the same model (`r4_no_overlap_thr.py`)

Run after `r4_no_overlap.py`. Chooses the R4 thresholds on the predictions of the same
28-repository model that scores the ranking test PRs, instead of the thresholds chosen on the
full-feature LORO predictions, and reruns `r3_report_r4.py` on a copy of its inputs.

```bash
$PY revision/r4_calc/r4_no_overlap_thr.py   # revision/output/r4/no_overlap_thr/{numbers.json, checks.json, by_repo.csv, per_pr.csv, ...}
```

## E1: deviation categories over repeated alignment runs (`r4_e1_repeats.py`)

Among equal-cost alignments pm4py returns one arbitrarily, so the categories of the
non-fitting traces (variant `main_noreapprove`) move slightly between runs of
`revision/r2/r2_alignment_v2.py` on the same sample; fitness, the share of fitting traces
and precision do not. Copy `r2_alignment_v2.json` of each run into
`revision/output/r4/e1_repeats/runs/` (the run the text reports as `ref.json`), then:

```bash
$PY revision/r4_calc/r4_e1_repeats.py   # revision/output/r4/e1_repeats/numbers.json (keys e1.rep.*)
```

## E5: calibration and threshold on a short project history (`r4_e5_numbers.py`)

Collects the chapter 4 numbers of experiment E5 from the finished outputs of `r3_calib_stability.py` and `r3_calib_adversarial/`; it fits
nothing. Inputs (read only): `revision/output/r3/calib_stability/` (from
`revision/r3_calib_stability.py`) and `revision/output/r3/calib_adversarial/` (from
`revision/r3_calib_adversarial/`: run `adv_calib.py`, `adv_reliab.py`, `adv_rules2.py` and
`adv_thr_nominfire.py` first; they write `whole_half.csv`, `threshold_short.csv`,
`reliab_diff.csv`, `rules_rolling2.csv` and `threshold_nominfire.csv`).

```bash
$PY revision/r4_calc/r4_e5_numbers.py   # revision/output/r4/e5_calibration/{numbers.json, sources.json}
```

Flat keys `e5.*`; `sources.json` names the file and column of each. Class balance and the history
needed for >= 100 PRs of each class repeat the computation of adv_calib.py on
`whole_half.csv` (rate = mean of the earlier and later half rates). The script asserts that the
sklearn re-fit (`whole_half.csv`) reproduces the ECE and the b1 <= 0 count of
`reference_summary.csv`. ECE of the switching rules is null: `rules_rolling2.csv` keeps per-block
sums only.
Also: the direct count of the merged history with >= 100 PRs of
each class (from the LORO forecasts, via `adv_calib.load/preds`), the constant early-half rate as a
reference for the class-weight correction (paired cluster bootstrap, B = 10 000, seed 20261114),
O/E outside [0.8; 1.25], `stab.<k>.stable_first_n`, firings on the history for thresholds without
the floor, the threshold on the whole earlier half (`thr_early`) and `rules.none` (= `rules.prior`).
ECE mirror keys in percentage points: `ece_pct.*` and `diff_ece_pct_const_early_minus_corr*` (sign: constant
minus correction, positive = correction better).
