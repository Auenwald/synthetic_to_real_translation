set -euo pipefail

TRAIN_SCRIPT="train_crossattention_wrapperv3.py"
SOURCE_PATH="./synthia"
SOURCE_NAME="synthiabranched"
LOG_DIR="./logs_diss"
GPU="${1:-0}"

mkdir -p "${LOG_DIR}"

for SEED in 0 1 2; do
    LOG_FILE="${LOG_DIR}/synthia_bidirectional_hsv_fixed_seed${SEED}.json"

    echo "============================================================"
    echo "SYNTHIA -> Cityscapes, BDD"
    echo "Architecture: dynamic bidirectional"
    echo "View: HSV"
    echo "Seed: ${SEED}"
    echo "GPU: ${GPU}"
    echo "Log: ${LOG_FILE}"
    echo "============================================================"

    python "${TRAIN_SCRIPT}" \
        --source_path "${SOURCE_PATH}" \
        --source_dataset_name "${SOURCE_NAME}" \
        --target_paths ./cityscapes ./bdd \
        --batch_size 2 \
        --lr 1.0e-5 \
        --weight_decay 1e-3 \
        --epochs 30 \
        --use_logging True \
        --log_file "${LOG_FILE}" \
        --skip_val_source False \
        --decay_factor 0.995 \
        --weight_averaging True \
        --averaging_interval 20 \
        --train_print_steps 50 \
        --gpu "${GPU}" \
        --mode hsv \
        --seed "${SEED}"
done