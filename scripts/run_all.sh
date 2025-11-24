#!/bin/bash

if [ $# -lt 1 ]; then
  echo "Usage: $0 DATA_FRAME_PATH"
  exit 1
fi

DATA_FRAME_PATH="$1"

for prediction in "ageBinary" "sex" "fieldIdBinary-0-2" "ventralDorsalBinary" "setupid" ; do
  for seed in {0..19}; do
    echo "Running: seed ${seed} prediction ${prediction}"
    run-random-forest ${DATA_FRAME_PATH} --features chirp_8Hz_average_norm preproc_bar --seed ${seed} --prediction ${prediction} ;
    done
  done
done
