#!/bin/bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage: $0 DATA_FRAME_PATH"
  exit 1
fi

DATA_FRAME_PATH="$1"
EXPERIMENTER_PAIRS=("Cai Dyszkant" "Cai Gonschorek" "Cai Szatko" "Dyszkant Gonschorek" "Dyszkant Szatko" "Gonschorek Szatko")

for experimenters in "${EXPERIMENTER_PAIRS[@]}"; do
  echo $experimenters
  for seed in {0..19}; do
    echo "Running: ${seed} ${experimenters}"
    run-random-forest ${DATA_FRAME_PATH} --features chirp_8Hz_average_norm preproc_bar --seed ${seed} --prediction experimenter --experimenters ${experimenters} ;
  done
done

