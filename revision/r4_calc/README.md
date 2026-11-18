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

## E9 preparation: baseline efficiency Eff without the technology (`r4_e9_baseline.py`)

All scripts from here to the end of this file read `revision/output/` written by steps 0 to 7 of the
run order in the main README, which need the collected corpus; the raw data is not published (see
*Personal data and GitHub terms* there).
`r4_numbers.json` holds no input data, but six of the eight scripts need it: they check their results
against it and stop when it is missing (see *Reference numbers of chapter 4* below).
Run them in this order, each on the main (not `--dev`) output of the previous ones:
`r4_e9_baseline.py`, `r4_e9_eff.py` (its `numbers.json` is read by `r4_e9b.py` and `r4_e9c.py`),
`r4_e9b.py`, `r4_e9c.py`, `r4_newdata.py` (reads `variants_e9b_r4.csv` and `numbers_e9b.json` of
the main `r4_e9b.py` run), `r4_mrr_tol.py` (reads the `r4_newdata.py` output); `r4_hparams.py`
and `r4_negtraces.py` need only the chapter 3 and chapter 2 outputs.
The variants, seeds and criteria named below as fixed before a run were recorded in the author's
notes, which are not part of this repository.

Eff(p) of chapter 2 (2.3.4; appendix B.3) on the main sample of the `r3_b40d` table
(`hait_filter == 1 & in_main == 1`, 189 646 PRs, 40 repositories). Components by human events:
mid-rank F_r of `t_m_h_D` (negative values clipped to 0) and `n_it_human`, plus `stale30_D`;
weights 1/3 each, components alone reported too. Normalisation A: F_r over PRs of the project
merged strictly before t_0(p), Eff not computed below n_min = 30. Normalisation B: F_r fixed on
the earlier half of the project (sort by created, h = n // 2, as `rule_thresholds` of E8), Eff for
the later half. Checks (asserts and `revision/tests/test_r4_e9_baseline.py`): mean F = 0.5 on the
earlier half against itself, row order and monotone transforms of t_m do not change the result.
The script also asserts n, it2 and stale shares and the t_m median of every repository against the
chapter 4 numbers of the dissertation (`R4_NUMBERS_JSON`, see the end of this file).
Diagnostic `diag.*`: rolling F over PRs created (not merged) before t_0.

```bash
$PY revision/r4_calc/r4_e9_baseline.py   # revision/output/r4/e9_eff/baseline/{numbers.json, by_repo.csv, by_pr.parquet}
```

## E9: share of Eff losses on the requests the rules act on (`r4_e9_eff.py`)

The variants, seeds, measures and criteria were fixed in the author's notes before any E9 result was seen.
Rules R_it (R1/R2 on the LORO forecast P_it), R_st (R3 on P_st) and both; thresholds
`r3_report_stats.choose_threshold` (n_f = 20, c = 0.1/0.2/0.3) re-chosen on the history of every
variant; removal n_it -> 1 (if >= 2), stale -> 0, t_m unchanged; Eff with F fixed on the history
(`r4_e9_baseline.mid_cdf`). Variants per project: `shuf` (200 permutations of whole requests,
first n // 2 = history), `block` (100 weekly block bootstraps inside the E8 halves), `fact` (the E8
split). A = removed share of the next-period Eff sum, A_rand = (k / n) * removable share,
lift = A / A_rand; mean over the 40 projects, cluster bootstrap over repositories (B = 10 000,
seed 20261115). Checks: K5 reproduces the E8 thresholds (`thresholds_by_repo.csv` of the chapter 3
reporting step and the chapter 4 numbers) and stops on a mismatch; K1 null action, K2 shuffled
forecasts, K3 input row order (recorded in `checks.json` and `runs.jsonl`, they do not stop the run); K4 oracle and the vectorised threshold in
`revision/tests/test_r4_e9_eff.py`.

```bash
$PY revision/r4_calc/r4_e9_eff.py --dev   # seed 1, 3 projects -> revision/output/r4/e9_eff/dev/
$PY revision/r4_calc/r4_e9_eff.py         # main -> revision/output/r4/e9_eff/{numbers.json, by_repo.csv, variants.csv, checks.json}
```

Flat keys `e9.<scheme>.<rule>.<c>.<w>.*` (w3 = weights 1/3; tm, nit, st = one component with weight 1),
controls `e9.ctrl.shuf.*`. Every run is appended to `revision/output/r4/e9_eff/runs.jsonl`.

## E9 extension: rule R4 and the R3 c grid (`r4_e9b.py`)

Added after the E9 results were seen; fixed in the author's notes before this calculation. E9b: rule R4 on the
ranking test PRs of the 12 ranking repositories that are in the main sample (dt_r from
`output/r4/no_overlap_thr/per_pr.csv`, written by `r4_no_overlap_thr.py`), removal
t_m' = max(t_m - k * dt_r, 0) with k = 1 and the empirical k (OLS on logs with repository fixed
effects and t_0 controls; median regression in hours, bootstrap in parallel processes); R1+R2+R4 and
re-ranking on all requests. R3grid: R3 on the c grid 0.2 ... 0.95 with the E9 permutations. Output
`output/r4/e9_eff/{numbers_e9b.json, checks_e9b.json, variants_e9b_r4.csv, variants_e9b_r3.csv}`,
keys `e9.r4.*`, `e9.r3grid.*`; runs appended to `runs.jsonl`.

```bash
$PY revision/r4_calc/r4_e9b.py --dev   # seed 1, 3 repositories -> output/r4/e9_eff/dev/
$PY revision/r4_calc/r4_e9b.py         # main (about an hour or more: 400 median regressions)
```

## E9 extension: simple selection baselines and the bound (`r4_e9c.py`)

Added after the E9 results were seen; fixed in the author's notes before this calculation. B1: the same number k of requests as rule R1/R2 (c = 0.2) in every project and variant,
chosen by a heuristic known at submission instead of the forecast P_it: largest change size, most
changed files, newest author (missing value = lowest priority, ties by a fixed random stream).
Criterion named in advance: the lower bound of the 95 % cluster bootstrap interval of
G_model / G_size is above 1; the other heuristics are reported without a criterion. B3: sensitivity
of the bound, on the requests with an R1/R2 action and n_it >= 2 also t_m -> min(t_m, Q_r), Q_r =
median t_m of the history requests with n_it = 1. `descr`: the oracle ceiling (all n_it >= 2 -> 1),
G as its share and the realised precision of R1/R2 on the next period, on the same E9 variants
(seed 20261115); the run stops unless it reproduces `e9.shuf.it.<c>.w3.A_pct` and `Arand_pct`.
B1 and B3 use scheme `shuf` of E9 on seed 20261116.

```bash
$PY revision/r4_calc/r4_e9c.py --dev   # seed 1, 3 projects
$PY revision/r4_calc/r4_e9c.py         # main -> revision/output/r4/e9_eff/*_e9c.*
```

## Prediction and ranking on constructed next periods (`r4_newdata.py`)

An addendum after the results were seen; the variants, seeds and measures were fixed in the
author's notes before this code. No confirmation criterion, a description of stability. AUC: the saved LORO
forecasts of chapter 3 (IT: n_it >= 2; TFR: long wait for the first review), no refitting, on the
200 next periods of E9 (scheme `shuf`, seed 20261115); per-repository AUC, mean over repositories.
Ranking: the fixed ranking model (`test_scored.parquet` of the `rank_shared` run of
`r3_night.sh`) on 200 next periods made of half of each repository's ranking test PRs; Hit@1,
Hit@3 and MRR of the model, the most familiar rule and the two tolerance re-rankings. Rule R4 at
c = 0.2 on the E9b infrastructure: thresholds on the history, firing share, expected latency of the
first candidate before and after, Hit@1 / Hit@3 / MRR on the firings and on all next-period PRs.
E6/E7 on the factual split in Hit@1, Hit@3 and MRR. The factual numbers are reproduced first; a
mismatch stops the run.

```bash
$PY revision/r4_calc/r4_newdata.py --dev   # seed 1, 10 variants -> revision/output/r4/newdata/dev/
$PY revision/r4_calc/r4_newdata.py         # main -> revision/output/r4/newdata/
```

## Admissibility bound by MRR (`r4_mrr_tol.py`)

A ranking sensitivity row fixed in the author's notes before this code; the main bound stays Hit@3. Same level
(95 %), family, objectives and tie rule as `rank/run_e4.select_and_headline` on the same validation
table; only the admissibility measure is MRR instead of Hit@3. Reports on the test PRs, next to the
main choice: tolerance, Gini of first places, distinct first-place people, median expected latency of
the first candidate, cost in Hit@1, Hit@3 and MRR; then the costs on the 200 next periods of
`r4_newdata.py` and rule R4 at c = 0.2 with the MRR latency tolerance. Run after `r4_newdata.py`.

```bash
$PY revision/r4_calc/r4_mrr_tol.py   # main -> revision/output/r4/mrr_tol/  (--no-variants: factual split only)
```

## E3 robustness to the CatBoost hyperparameters (`r4_hparams.py`)

Three alternative sets fixed before the run (H1 depth 4; H2 depth 8; H3 learning rate 0.025,
1 000 trees, l2_leaf_reg 10); everything else is the main model of chapter 3 (`r3_v2_models`: table,
features t0, clock D, population main, train-median imputation, LORO folds, seed 42, class weights).
Targets IT, TFR, ST, TM. Mean AUC over repositories with a cluster bootstrap interval, and the
paired difference with the main set H0 (B = 10 000, seed 42). The E3 verdict is robust for a target
if no set changes it and no difference is material (the interval excludes 0 and |delta| >= 0.02).
H0 is not refitted: its numbers come from the stored `r3_b40d` predictions, and two folds per target
are refitted to check the reproduction (the summary stops if max |delta p| > 1e-6). Folds are saved after every
fit, so an interrupted run resumes.

```bash
$PY revision/r4_calc/r4_hparams.py --set H0check   # reproduction of 2 folds per target
$PY revision/r4_calc/r4_hparams.py --set H1        # one process per set; resumable
$PY revision/r4_calc/r4_hparams.py --set H2
$PY revision/r4_calc/r4_hparams.py --set H3
$PY revision/r4_calc/r4_hparams.py --summarise     # revision/output/r4/hparams/{numbers.json, checks.json}
```

## E1 negative traces (`r4_negtraces.py`)

Paired design: every one of the 8 000 alignment traces of chapter 2 is mutated separately in each
class, and the fitness of the mutant on the chapter 2 workflow net is compared with the fitness of the
same original trace. Classes that must be detected (criterion: fitness drops in >= 95 % of pairs):
M1 all approvals removed, M2 merge moved to the first approval, M3 a review or revision after the
merge, M4 a change request right after the first approval, M6a no opening, M6b merge first. M5 (one
adjacent pair swapped) and M4 on the relaxed net are reported without a criterion. Harmless changes
that must not be punished (fitness drops in <= 5 % of pairs): Kpa a repeated comment, Kpb a revision
before the first approval, Kpc an assistant run. The state automaton (one rejection rule: a human
approval before the merge) is applied to the same sequences only as an interpretation. The chapter 2
code `revision/r2/r2_alignment_v2.py` is imported unchanged, so the script needs the environment of
`requirements-alignment.txt` and the output of `revision/r2/` (`R2_DIR`). The mutation tests need no
pm4py; the alignment tests run in that environment with
`python revision/tests/test_r4_negtraces.py`. The pm4py playout of the 1 000 control traces is not
reproducible run to run even with the seed fixed, so `neg.playout.*` (e.g. the M5 drop share) varies
slightly between runs; the numbers on the 8 000 alignment traces are reproducible.

```bash
$PY revision/r4_calc/r4_negtraces.py --dev   # 2 repositories x 20 traces
$PY revision/r4_calc/r4_negtraces.py         # main -> revision/output/r4/negtraces/{numbers.json, checks.json, pairs.csv, runs.jsonl}
```

## Reference numbers of chapter 4 (`R4_NUMBERS_JSON`)

`r4_e9_baseline.py`, `r4_e9_eff.py`, `r4_e9b.py`, `r4_e9c.py` (the last two through the E8 reproduction
check of `r4_e9_eff.py`), `r4_newdata.py`, `r4_mrr_tol.py` and `r4_hparams.py` compare what they
recompute with the chapter 4 numbers of the dissertation, a nested JSON with the
sections the scripts read: `corpus`, `repo_table`, `repo_total`, `thr`, `r4`, `r4_noxt`, `pred`
and `rank`. The file is included as `revision/r4_calc/r4_numbers.json` (these sections as reported
in the dissertation, 16 November 2026); it holds aggregates only: repository names, AI tool names,
languages, numbers and a few Ukrainian labels of the dissertation tables (verdicts "так"/"ні",
"переноситься", groups "компанія"/"спільнота", total "Разом"), no per-request records and no
account logins. Counts of AI-assistant
requests below 5 per repository (`repo_table[].ai_n`, `ai_pct`, `ai_top`) are set to null; no script
reads them.
Set `R4_NUMBERS_JSON` to compare against another copy. `r4_hparams.py` skips the comparison when
the file is missing; the other scripts stop.
