#!/bin/bash
# Night run for chapter 3 on corpus v2.
# Every step is idempotent: a finished step leaves <STATE>/<step>.done and is
# skipped on restart; model blocks also resume fit by fit from metrics.jsonl.
# Restart after an interruption with the same command.
#
#   nohup caffeinate -i -s bash revision/r3_night.sh > <OUT>/night.log 2>&1 &
#
# Inputs are read only: SRC_DATA is the data folder of the collector (default: data/).
set -u

CLONE="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DATA="${SRC_DATA:-$CLONE/data}"
DISS="${DISS:-${DISSERTATION_DIR:-$CLONE/revision/output}}"   # output root of revision/paths.py
PY="${PY:-python}"
OUT="${OUT:-$CLONE/revision/output/r3}"
STATE="$OUT/state"
mkdir -p "$STATE"

export R3_RAW_V2="$SRC_DATA/raw_v2"
export R3_OUT="$OUT/v2"
export R3_THREADS="${R3_THREADS:-8}"
D_LABELS="$DISS/journal_articles/HAIT_2026/revision_results/D-definition3-labels.csv"

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

in_dir() { local d="$1"; shift; mkdir -p "$d" && (cd "$d" && "$@"); }

echo "[$(ts)] night run: clone $CLONE @ $(git -C "$CLONE" rev-parse --short HEAD), out $OUT"
pmset -g batt | head -2

# 0. data manifest of the snapshot (SHA-256 of every input file read)
manifest() {
  ( cd "$SRC_DATA" && find raw_v2 -name prs.jsonl -o -name repos.csv | sort | xargs shasum -a 256 ) > "$OUT/data_manifest_sha256.txt" &&
  ( cd "$SRC_DATA" && shasum -a 256 processed/all_merged_features.csv raw/*/merged_prs.json ) >> "$OUT/data_manifest_sha256.txt"
}
step 00_manifest manifest

# 1. per-PR table of corpus v2 (labels D and A, features at t0, reverts)
step 01_prep_v2 in_dir "$OUT/v2" "$PY" "$CLONE/revision/r3_v2_prep.py"

# 2. main models (transfer across projects): LORO, LOLO (CatBoost, LogReg), project prior, LOGO (groups)
step 02_models_main in_dir "$OUT/v2" "$PY" "$CLONE/revision/r3_v2_models.py" --block main

# 3. controls: shuffled labels, random "language"
step 03_models_controls in_dir "$OUT/v2" "$PY" "$CLONE/revision/r3_v2_models.py" --block controls

# 4. reproduction and contribution of changes
#    S0: exact ITIA (tag v1.2.0) from an archive of the tag, data via symlinks
s0() {
  local SRC="$OUT/itia_v120_src" DST="$OUT/itia_v120_out"
  mkdir -p "$SRC/data" "$DST/journal_articles/HAIT_2026/revision_results"
  [ -f "$SRC/revision/taskF1_duration_language.py" ] || git -C "$CLONE" archive v1.2.0 revision predict enrich | tar -x -C "$SRC"
  ln -sfn "$SRC_DATA/raw" "$SRC/data/raw"; ln -sfn "$SRC_DATA/processed" "$SRC/data/processed"
  ln -sfn "$D_LABELS" "$DST/journal_articles/HAIT_2026/revision_results/D-definition3-labels.csv"
  (cd "$DST" && DISSERTATION_DIR="$DST" "$PY" "$SRC/revision/taskF1_duration_language.py" --stage all)
}
step 04a_itia_v120_exact s0
#    S1-S3 on corpus v1 with this version (new account lists, then one change at a time)
ladder_v1() {
  local D="$OUT/ladder_v1"
  mkdir -p "$D"
  (cd "$D" && export PRR_RAW_DIR="$SRC_DATA/raw" PRR_PROCESSED_DIR="$SRC_DATA/processed" \
      F1_OUT_DIR="$D" F1_D_LABELS="$D_LABELS" &&
   { [ -f "$D/F-pr-level.csv" ] || "$PY" "$CLONE/revision/taskF1_duration_language.py" --stage prep; } &&
   "$PY" "$CLONE/revision/r3_ladder_v1.py")
}
step 04b_ladder_v1 ladder_v1
#    S4a-S5 on corpus v2
step 04c_ladder_v2 in_dir "$OUT/v2" "$PY" "$CLONE/revision/r3_v2_models.py" --block ladder

# 5. Ranking on the 12 AAIT repositories with the shared account classification
rank_run() {   # rank_run <accounts> <outdir>
  local ACC="$1" RD="$2"
  mkdir -p "$RD"
  (cd "$CLONE" && export RANK_ACCOUNTS="$ACC" RANK_RAW_DIR="$SRC_DATA/raw" RANK_OUT_DIR="$RD" &&
   { [ -f "$RD/.built" ] || { "$PY" -m rank.build > "$RD/build_log.txt" 2>&1 && touch "$RD/.built"; }; } &&
   { [ -f "$RD/results/test_scored.parquet" ] || "$PY" -m rank.run_experiments; } &&
   { [ -f "$RD/results/e4v2_headline_test.csv" ] || "$PY" -m rank.run_e4; } &&
   "$PY" -m rank.run_r1)
}
step 05_rank_shared rank_run shared "$OUT/rank_shared"

# 6. sensitivity (with bot authors, no text features, history minimum, fixed 24 h)
step 06_models_sensitivity in_dir "$OUT/v2" "$PY" "$CLONE/revision/r3_v2_models.py" --block sensitivity

# 7. temporal split (last: first to drop if time runs out)
step 07_models_temporal in_dir "$OUT/v2" "$PY" "$CLONE/revision/r3_v2_models.py" --block temporal

# 8. ranking with the paper's regex filter, same code and data: separates the
#    effect of the account classification from any run-to-run difference
step 08_rank_regex_repro rank_run regex "$OUT/rank_regex"

echo "[$(ts)] night run finished"
ls "$STATE"
