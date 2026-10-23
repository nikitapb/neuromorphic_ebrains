#!/bin/bash

# Output folder for all dANN-C Figure-2 models.
DIRPARAMS="../DATA_C"

# Fixed Figure-2 training parameters.
GPU_ID=0
SEQ_FLAG=0
ESTOP_FLAG=0
SIGMA=0
DATA="fmnist"
NLAYERS=1
NSYN=16
DROP_FLAG=0
DROP_RATE=0.0
LR=0.001

# Train model 13 across the Figure-2 architecture grid.
for trial in 1 2 3 4 5; do
    for dends in 2 4 8 16 32 64; do
        for soma in 32 64 128 256 512; do

            echo
            echo "========================================"
            echo "dANN-C | trial=$trial | D=$dends | S=$soma"
            echo "========================================"

            python main.py \
                $GPU_ID \
                $SEQ_FLAG \
                $ESTOP_FLAG \
                $trial \
                13 \
                $SIGMA \
                $DATA \
                $dends \
                $soma \
                $NLAYERS \
                $NSYN \
                $DROP_FLAG \
                $DROP_RATE \
                $LR \
                $DIRPARAMS

        done
    done
done
