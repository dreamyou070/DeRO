#!/usr/bin/env bash
set -euo pipefail

BASE_FOLDER="/MIR/nas1/dreamyou070/NegatingDiffusion/data_final/3_negation_imagenet"
REAL_DIR="/MIR/nas1/dreamyou070/hf_cache/annotations/val2017"


OUT_BASE="/MIR/nas1/dreamyou070/NegatingDiffusion/data_final/fid_iterative"
mkdir -p "${OUT_BASE}"

NEG_MODE="negation_mode"
POS_MODE="positive_mode"

# NEG only
for LR_DIR in "${BASE_FOLDER}/${NEG_MODE}"/*; do
  [ -d "${LR_DIR}" ] || continue
  LR_NAME=$(basename "${LR_DIR}")

  for W_DIR in "${LR_DIR}"/*; do
    [ -d "${W_DIR}" ] || continue
    W_NAME=$(basename "${W_DIR}")

    echo "[NEG] lr=${LR_NAME} weight=${W_NAME}"

    python run_pos_neg.py \
      --base_folder "${BASE_FOLDER}" \
      --real_dir "${REAL_DIR}" \
      --out_csv "${OUT_BASE}/neg_${LR_NAME}_${W_NAME}_detail.csv" \
      --out_summary_csv "${OUT_BASE}/neg_${LR_NAME}_${W_NAME}_summary.csv" \
      --negation_mode "${NEG_MODE}" \
      --positive_mode "__DUMMY__" \
      --save_every 1 \
      --summary_every 1 \
      --overwrite
  done
done

# POS only
for LR_DIR in "${BASE_FOLDER}/${POS_MODE}"/*; do
  [ -d "${LR_DIR}" ] || continue
  LR_NAME=$(basename "${LR_DIR}")

  for W_DIR in "${LR_DIR}"/*; do
    [ -d "${W_DIR}" ] || continue
    W_NAME=$(basename "${W_DIR}")

    echo "[POS] lr=${LR_NAME} weight=${W_NAME}"

    python run_pos_neg.py \
      --base_folder "${BASE_FOLDER}" \
      --real_dir "${REAL_DIR}" \
      --out_csv "${OUT_BASE}/pos_${LR_NAME}_${W_NAME}_detail.csv" \
      --out_summary_csv "${OUT_BASE}/pos_${LR_NAME}_${W_NAME}_summary.csv" \
      --negation_mode "__DUMMY__" \
      --positive_mode "${POS_MODE}" \
      --save_every 1 \
      --summary_every 1 \
      --overwrite
  done
done


#chmod +x run_pos_neg.sh
#./run_pos_neg.sh