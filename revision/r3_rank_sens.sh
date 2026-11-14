#!/bin/bash
# Sensitivity of the ranking constants (dissertation, chapter 3).
# One constant changes at a time; the rest is
# AAIT 2026, accounts "shared", the 12 AAIT repositories of corpus v1.
# The base is step 05_rank_shared of r3_night.sh and is only read here.
#
# Every step is idempotent: a finished step leaves <STATE>/<step>.done and is
# skipped on restart; inside a variant, finished stages are skipped too.
#
#   nohup caffeinate -i -s bash revision/r3_rank_sens.sh > <OUT>/run.log 2>&1 &
#
# Inputs are read only: SRC_DATA is the data folder of the collector (default: data/).
set -u

CLONE="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DATA="${SRC_DATA:-$CLONE/data}"
PY="${PY:-python}"
R3="${R3:-$CLONE/revision/output/r3}"
BASE="$R3/rank_shared"
OUT="${OUT:-$R3/rank_sens}"
STATE="$OUT/state"
mkdir -p "$STATE"

ts() { date "+%Y-%m-%d %H:%M:%S"; }

step() {   # step <name> <command...>
  local name="$1"; shift
  if [ -f "$STATE/$name.done" ]; then
    echo "[$(ts)] skip $name (done)"; return 0
  fi
  echo "[$(ts)] start $name"
  if "$@"; then
    date "+%Y-%m-%dT%H:%M:%S" > "$STATE/$name.done"
    echo "[$(ts)] done $name"
  else
    echo "[$(ts)] FAILED $name (exit $?); continuing with the next step"
  fi
}

echo "[$(ts)] rank sensitivity: clone $CLONE @ $(git -C "$CLONE" rev-parse --short HEAD), out $OUT"
pmset -g batt | head -2

# 0. the raw files read here must be the ones the base read (night manifest)
manifest() {
  ( cd "$SRC_DATA" && shasum -a 256 raw/*/merged_prs.json ) > "$OUT/data_manifest_sha256.txt" &&
  if [ -f "$R3/data_manifest_sha256.txt" ]; then
    grep -F -f "$OUT/data_manifest_sha256.txt" "$R3/data_manifest_sha256.txt" | wc -l | tr -d ' ' > "$OUT/manifest_matches.txt"
    [ "$(cat "$OUT/manifest_matches.txt")" = "$(wc -l < "$OUT/data_manifest_sha256.txt" | tr -d ' ')" ] ||
      { echo "raw data differ from the night run manifest"; return 1; }
  fi
}
step 00_manifest manifest || true
[ -f "$STATE/00_manifest.done" ] || { echo "[$(ts)] stop: data check failed"; exit 1; }

# 1. admissibility bound: re-select on the saved E4v2 tables of the base, no retraining
adm() {   # adm <delta> <outdir>
  local DELTA="$1" RD="$2"
  mkdir -p "$RD/results"
  for f in e4v2_val_all.csv e4v2_test_all.csv e4v2_actual_concentration.csv; do
    [ -f "$RD/results/$f" ] || cp "$BASE/results/$f" "$RD/results/$f" || return 1
  done
  (cd "$CLONE" && export RANK_ACCOUNTS=shared RANK_RAW_DIR="$SRC_DATA/raw" RANK_OUT_DIR="$RD" &&
   "$PY" -m rank.run_e4 --select-only --delta "$DELTA")
}
step adm90 adm 0.10 "$OUT/adm90"
step adm98 adm 0.02 "$OUT/adm98"

# 2. rebuild variants: pairs, E0-E2 (no LORO), E4v2, R1
rank_variant() {   # rank_variant <outdir> <build args...>
  local RD="$1"; shift
  mkdir -p "$RD"
  (cd "$CLONE" && export RANK_ACCOUNTS=shared RANK_RAW_DIR="$SRC_DATA/raw" RANK_OUT_DIR="$RD" &&
   { [ -f "$RD/.built" ] || { "$PY" -m rank.build "$@" > "$RD/build_log.txt" 2>&1 && touch "$RD/.built"; }; } &&
   { [ -f "$RD/results/test_scored.parquet" ] || "$PY" -m rank.run_experiments --skip-loro; } &&
   { [ -f "$RD/results/e4v2_headline_test.csv" ] || "$PY" -m rank.run_e4; } &&
   "$PY" -m rank.run_r1)
}
step split70 rank_variant "$OUT/split70" --fractions 0.7,0.15,0.15
step warm30  rank_variant "$OUT/warm30"  --warmup-days 30
step warm180 rank_variant "$OUT/warm180" --warmup-days 180
step pool90  rank_variant "$OUT/pool90"  --cand-window-days 90
step pool365 rank_variant "$OUT/pool365" --cand-window-days 365

# 3. summary.csv and rank_sens_numbers.json
step summary "$PY" "$CLONE/revision/r3_rank_sens_summary.py" --base "$BASE" --out "$OUT"

echo "[$(ts)] rank sensitivity finished"
ls "$STATE"
