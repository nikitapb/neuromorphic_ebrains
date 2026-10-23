#!/bin/bash

# ============================================================
# Run complete dANN-LRF grid locally
#
# Model type 2 = dANN-LRF
#
# Full Figure-2 architecture grid:
# D = 2, 4, 8, 16, 32, 64
# S = 32, 64, 128, 256, 512
#
# Trials = 1-5
#
# Existing completed runs are automatically skipped.
#
# Usage:
#
# bash run_dannLRF_condition.sh
# ============================================================

set -e


# ------------------------------------------------------------
# FULL FIGURE-2 GRID
# ------------------------------------------------------------

DENDS=(2 4 8 16 32 64)

SOMATA=(32 64 128 256 512)

TRIALS=(1 2 3 4 5)


# ------------------------------------------------------------
# OUTPUT DIRECTORY
#
# model_type = 2 generates:
# dend_ann_local_rfs
# ------------------------------------------------------------

DATA_DIR="../DATA"

MODEL_DIR="${DATA_DIR}/results_fmnist_1_layer/dend_ann_local_rfs"


echo ""
echo "============================================================"
echo "dANN-LRF FULL GRID"
echo "============================================================"
echo "Model type = 2"
echo "Trials     = ${TRIALS[*]}"
echo ""
echo "Output folder:"
echo "${MODEL_DIR}"
echo "============================================================"
echo ""


# ------------------------------------------------------------
# CREATE OUTPUT ROOT IF NEEDED
# ------------------------------------------------------------

mkdir -p "${DATA_DIR}"


# ------------------------------------------------------------
# RUN GRID
# ------------------------------------------------------------

for D in "${DENDS[@]}"
do

    for S in "${SOMATA[@]}"
    do

        for TRIAL in "${TRIALS[@]}"
        do

            RESULT_FILE="${MODEL_DIR}/results_sigma_0.0_trial_${TRIAL}_dends_${D}_soma_${S}.pkl"


            # =================================================
            # SKIP ALREADY COMPLETED RUNS
            # =================================================

            if [ -f "${RESULT_FILE}" ]; then

                echo "SKIP: trial=${TRIAL} D=${D} S=${S}"

                continue

            fi


            echo ""
            echo "------------------------------------------------------------"
            echo "TRAINING dANN-LRF"
            echo "trial = ${TRIAL}"
            echo "D     = ${D}"
            echo "S     = ${S}"
            echo "------------------------------------------------------------"
            echo ""


            # =================================================
            # main.py arguments
            #
            # 0          GPU
            # 0          sequential
            # 0          early stopping
            # TRIAL      trial
            # 2          MODEL TYPE = dANN-LRF
            # 0          sigma
            # fmnist     dataset
            # D          dendrites / soma
            # S          somata
            # 1          number of layers
            # 16         synapses / dendrite
            # 0          dropout
            # 0          dropout rate
            # 0.001      learning rate
            # DATA_DIR   output root
            #
            # Last two arguments are harmless dummy values.
            # They are irrelevant because model type 2 has
            # coupling disabled.
            # =================================================

            python main.py \
                0 \
                0 \
                0 \
                ${TRIAL} \
                2 \
                0 \
                fmnist \
                ${D} \
                ${S} \
                1 \
                16 \
                0 \
                0 \
                0.001 \
                ${DATA_DIR} \
                0.0 \
                0.0

        done

    done

done


echo ""
echo "============================================================"
echo "dANN-LRF FULL GRID COMPLETE"
echo "============================================================"