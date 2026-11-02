#!/usr/bin/env bash
set -e

DATA="fmnist"

D=4
S=512

NLAYERS=1
NSYN=16

SIGMA=0.0
LR=0.001

DROP=0
DROP_RATE=0

OUT="../DATA_C"

MAIN="main.py"

MAIN30="_main_frac30_tmp.py"
MAIN40="_main_frac40_tmp.py"

echo
echo "Preparing temporary training scripts..."

cp "$MAIN" "$MAIN30"

sed -i \
    's/coupling_fraction=0\.20/coupling_fraction=0.30/g' \
    "$MAIN30"

sed -i \
    's/coupling_fraction = 0\.20/coupling_fraction = 0.30/g' \
    "$MAIN30"

sed -i \
    's/fname_model += "_C"/fname_model += "_C_frac30_lam0p1"/' \
    "$MAIN30"


cp "$MAIN" "$MAIN40"

sed -i \
    's/coupling_fraction=0\.20/coupling_fraction=0.40/g' \
    "$MAIN40"

sed -i \
    's/coupling_fraction = 0\.20/coupling_fraction = 0.40/g' \
    "$MAIN40"

sed -i \
    's/fname_model += "_C"/fname_model += "_C_frac40_lam0p1"/' \
    "$MAIN40"


run_model () {

    SCRIPT=$1
    MODEL=$2
    TRIAL=$3
    FOLDER=$4

    MODELFILE="${OUT}/results_fmnist_1_layer/${FOLDER}/model_sigma_${SIGMA}_trial_${TRIAL}_dends_${D}_soma_${S}.keras"

    RESULTFILE="${OUT}/results_fmnist_1_layer/${FOLDER}/results_sigma_${SIGMA}_trial_${TRIAL}_dends_${D}_soma_${S}.pkl"

    if [[ -f "$MODELFILE" && -f "$RESULTFILE" ]]; then
        echo "SKIP ${FOLDER} trial=${TRIAL} D=${D} S=${S}"
        return
    fi

    echo
    echo "============================================================"
    echo "RUN ${FOLDER}"
    echo "trial=${TRIAL}"
    echo "D=${D}, S=${S}"
    echo "============================================================"

    python "$SCRIPT" \
        0 \
        0 \
        0 \
        "$TRIAL" \
        "$MODEL" \
        "$SIGMA" \
        "$DATA" \
        "$D" \
        "$S" \
        "$NLAYERS" \
        "$NSYN" \
        "$DROP" \
        "$DROP_RATE" \
        "$LR" \
        "$OUT"
}


echo
echo "############################################################"
echo "# dANN-LRF"
echo "############################################################"

for TRIAL in $(seq 1 15)
do
    run_model \
        "$MAIN" \
        2 \
        "$TRIAL" \
        "dend_ann_local_rfs"
done


echo
echo "############################################################"
echo "# 20% coupling"
echo "############################################################"

for TRIAL in $(seq 6 15)
do
    run_model \
        "$MAIN" \
        13 \
        "$TRIAL" \
        "dend_ann_local_rfs_C"
done


echo
echo "############################################################"
echo "# 30% coupling"
echo "############################################################"

for TRIAL in $(seq 6 15)
do
    run_model \
        "$MAIN30" \
        13 \
        "$TRIAL" \
        "dend_ann_local_rfs_C_frac30_lam0p1"
done


echo
echo "############################################################"
echo "# 40% coupling"
echo "############################################################"

for TRIAL in $(seq 6 15)
do
    run_model \
        "$MAIN40" \
        13 \
        "$TRIAL" \
        "dend_ann_local_rfs_C_frac40_lam0p1"
done


rm -f "$MAIN30"
rm -f "$MAIN40"

echo
echo "COMPLETE"
