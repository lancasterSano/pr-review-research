# Independent check of the chapter 3 calibration

Independent check of `revision/r3_calib_stability.py` and of the threshold on short histories.

Input (read only): `revision/output/r3_b40d/v2` (LORO out-of-fold forecasts, `metrics.jsonl`, PR table).
Output: `revision/output/r3/calib_adversarial/`.

Run from the repository root, after `revision/r3_calib_stability.py` (`adv_thr_nominfire.py` reads its
`per_repo_by_n.csv`):

```bash
PY=python   # interpreter with requirements.txt
D=revision/r3_calib_adversarial   # relative to the repository root
$PY $D/adv_calib.py          # whole_half.csv, threshold_short.csv, rolling_fits.csv, rolling_measures.csv
$PY $D/adv_reliab.py         # reliab_diff.csv, reliab_st_deciles.csv (reads whole_half.csv); also prints
$PY $D/adv_rules2.py         # rules_rolling2.csv, first_prs2.csv (calibration rules, rolling origin)
$PY $D/adv_thr_nominfire.py  # threshold_nominfire.csv, threshold_nominfire_summary.csv, b1_sklearn_check.csv
                             # (threshold by (2.9) without the 20-firing floor, history n = 30/50/100/200); also prints
```

`adv_calib.py` must run first: the others import it, and `adv_reliab.py` reads its `whole_half.csv`.
