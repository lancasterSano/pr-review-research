#!/bin/bash
# Chapter 3 rerun of 02.11.2026, the run the dissertation reports: human reviews
# of a draft leave t_r and the iterations, LORO on the temporal test PRs,
# partial-feature baselines. Output: revision/output/r3_b40d.
# The corpus v1 and ranking steps do not read pr_table_v2.csv and are linked
# from the earlier full run in revision/output/r3; everything else reruns.
#
#   nohup caffeinate -i -s bash revision/r3_rerun_0211.sh > revision/output/r3_b40d.log 2>&1 < /dev/null & disown
set -u
CLONE="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DATA="${SRC_DATA:-$CLONE/data}"
PY="${PY:-python}"
SRC="$CLONE/revision/output/r3"
export OUT="$CLONE/revision/output/r3_b40d" R3_ROOT="$CLONE/revision/output/r3_b40d"
export SRC_DATA PY R3_RAW_V2="$SRC_DATA/raw_v2" R3_RAW_V1="$SRC_DATA/raw"
ts() { date "+%F %T"; }

# The rerun reuses steps of an earlier full run in revision/output/r3: r3_night.sh
# (OUT=revision/output/r3) and r3_rank_sens.sh. Stop if they have not been run.
for d in itia_v120_out itia_v120_src ladder_v1 rank_regex rank_sens rank_shared; do
  if [ ! -e "$SRC/$d" ]; then
    echo "missing $SRC/$d: run revision/r3_night.sh and revision/r3_rank_sens.sh first" >&2
    exit 1
  fi
done
mkdir -p "$OUT/state"
for d in itia_v120_out itia_v120_src ladder_v1 rank_regex rank_sens rank_shared; do
  [ -e "$OUT/$d" ] || ln -s "$SRC/$d" "$OUT/$d"
done
for s in 04a_itia_v120_exact 04b_ladder_v1 05_rank_shared 08_rank_regex_repro; do
  [ -f "$OUT/state/$s.done" ] || echo "linked from revision/output/r3" > "$OUT/state/$s.done"
done

echo "[$(ts)] rerun 02.11 @ $(git -C "$CLONE" rev-parse --short HEAD)"
bash "$CLONE/revision/r3_night.sh"
export R3_OUT="$OUT/v2" R3_THREADS=6
if [ ! -f "$OUT/state/09_models_heuristics.done" ]; then
  echo "[$(ts)] start 09_models_heuristics"
  (cd "$OUT/v2" && "$PY" "$CLONE/revision/r3_v2_models.py" --block heuristics) &&
    date "+%Y-%m-%dT%H:%M:%S" > "$OUT/state/09_models_heuristics.done"
  echo "[$(ts)] 09_models_heuristics exit $?"
fi
cd "$CLONE/revision"
for b in r4 q75 nmin10; do echo "[$(ts)] start report block $b"; "$PY" r3_report_models.py --block $b; echo "[$(ts)] $b exit $?"; done
cd "$CLONE"
echo "[$(ts)] start rank report"; "$PY" revision/r3_report_rank.py; echo "[$(ts)] rank exit $?"
echo "[$(ts)] start report"; "$PY" revision/r3_report.py; echo "[$(ts)] report exit $?"
echo "[$(ts)] start r4 report"; "$PY" revision/r3_report_r4.py; echo "[$(ts)] r4 report exit $?"
echo "[$(ts)] rerun 02.11 finished"
