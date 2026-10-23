#!/bin/bash

# ============================================================
# Run ONE complete dANN-C parameter condition
#
# Usage:
#
# bash run_dannC_condition.sh FRACTION LAMBDA
#
# Examples:
# bash run_dannC_condition.sh 0.20 0.05
# bash run_dannC_condition.sh 0.20 0.20
# bash run_dannC_condition.sh 0.10 0.10
#
# Existing runs are automatically skipped.
# ============================================================

set -e


# ------------------------------------------------------------
# PARAMETERS FROM COMMAND LINE
# ------------------------------------------------------------

FRACTION=$1
LAMBDA=$2


if [ -z "$FRACTION" ] || [ -z "$LAMBDA" ]; then
    echo "Usage:"
    echo "bash run_dannC_condition.sh FRACTION LAMBDA"
    exit 1
fi


# ------------------------------------------------------------
# FULL FIGURE-2 GRID
# ------------------------------------------------------------

DENDS=(2 4 8 16 32 64)

SOMATA=(32 64 128 256 512)

TRIALS=(1 2 3 4 5)

DATA_DIR="../DATA_C"


# ------------------------------------------------------------
# CREATE THE SAME FOLDER TAG AS main.py
# ------------------------------------------------------------

FRAC_TAG=$(python -c "print(int(round(float('${FRACTION}') * 100)))")

LAM_TAG=$(python -c "print(str(float('${LAMBDA}')).replace('.', 'p'))")


MODEL_DIR="${DATA_DIR}/results_fmnist_1_layer/dend_ann_local_rfs_C_frac${FRAC_TAG}_lam${LAM_TAG}"


echo ""
echo "============================================================"
echo "dANN-C FULL GRID"
echo "============================================================"
echo "Coupled fraction = ${FRACTION}"
echo "Coupling lambda  = ${LAMBDA}"
echo "Output folder:"
echo "${MODEL_DIR}"
echo "============================================================"
echo ""


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
            echo "TRAINING"
            echo "fraction = ${FRACTION}"
            echo "lambda   = ${LAMBDA}"
            echo "trial    = ${TRIAL}"
            echo "D        = ${D}"
            echo "S        = ${S}"
            echo "------------------------------------------------------------"
            echo ""


            python main.py \
                0 \
                0 \
                0 \
                ${TRIAL} \
                13 \
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
                ${FRACTION} \
                ${LAMBDA}

        done

    done

done


echo ""
echo "============================================================"
echo "CONDITION COMPLETE"
echo "fraction = ${FRACTION}"
echo "lambda   = ${LAMBDA}"
echo "============================================================"
