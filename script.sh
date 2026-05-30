#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash script.sh [dataset] [scene] [guidance] [image_guidance] [color_mode] [entropy_weight]
# Example:
# bash script.sh dynerf cook_spinach 10.5 1.2 sh 0.0
# bash script.sh dynerf cook_spinach 10.5 1.2 lite 0.002

DATASET="${1:-dynerf}"
SCENE="${2:-cook_spinach}"
GUIDANCE="${3:-10.5}"
IMAGE_GUIDANCE="${4:-1.2}"
COLOR_MODE="${5:-sh}"
ENTROPY_WEIGHT="${6:-0.0}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="./output/${DATASET}/${SCENE}/hybrid_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${COLOR_MODE}_entropy${ENTROPY_WEIGHT}_${TIMESTAMP}.log"

PROMPTS=(
  "Make it look like a fauvism painting"
  "Make it look like a sculpture"
  "Turn the man into a woman"
)

echo "dataset=${DATASET}, scene=${SCENE}, color_mode=${COLOR_MODE}, entropy=${ENTROPY_WEIGHT}" | tee -a "${LOG_FILE}"
echo "log_file=${LOG_FILE}" | tee -a "${LOG_FILE}"

for PROMPT in "${PROMPTS[@]}"; do
  echo "======================================================" | tee -a "${LOG_FILE}"
  echo "Running prompt: ${PROMPT}" | tee -a "${LOG_FILE}"
  START="$(date +%s)"

  bash run_instruct_4dgs.sh \
    "${DATASET}" \
    "${SCENE}" \
    "${PROMPT}" \
    "${GUIDANCE}" \
    "${IMAGE_GUIDANCE}" \
    "${COLOR_MODE}" \
    "${ENTROPY_WEIGHT}" | tee -a "${LOG_FILE}"

  END="$(date +%s)"
  echo "Prompt runtime (sec): $((END - START))" | tee -a "${LOG_FILE}"
done

python scripts/cal_modelsize.py \
  --model_root "./output/${DATASET}/${SCENE}" \
  --report_path "${LOG_DIR}/modelsize_${COLOR_MODE}_${TIMESTAMP}.json" | tee -a "${LOG_FILE}"

echo "All prompts finished." | tee -a "${LOG_FILE}"