#!/bin/bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage: $0 DATA_FRAME_PATH"
  exit 1
fi

DATA_FRAME_PATH="$1"
shift

for seed in {0..19}; do
  echo "Running session-date batch-effect experiment: seed ${seed}"
  python -m all_gcl_manuscript.batch_effects.run_session_date_effects \
    "${DATA_FRAME_PATH}" \
    --features chirp_8Hz_average_norm preproc_bar \
    --seed "${seed}" \
    --exp-num 1 \
    --max-age-diff-weeks 10 \
    --max-field-distance-um -1 \
    --no-match-retinal-quadrant \
    --field-ids 0 1 \
    "$@"
done
