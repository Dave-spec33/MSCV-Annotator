#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-VL-8B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
VAL_DATASET="${VAL_DATASET:-${PROJECT_ROOT}/portable_dataset/runtime_data/val.jsonl}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${SCRIPT_DIR}/outputs}"
MODE="${1:-${MODE:-both}}"

GPU="${GPU:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-6}"

if [[ ! -f "${VAL_DATASET}" ]]; then
  echo "Missing validation dataset: ${VAL_DATASET}" >&2
  exit 1
fi
if ! command -v swift >/dev/null 2>&1; then
  echo "The 'swift' command is not available." >&2
  exit 1
fi

mkdir -p "${EVAL_OUTPUT_DIR}"
BASE_OUTPUT="${EVAL_OUTPUT_DIR}/qwen3vl8b_base.jsonl"
LORA_OUTPUT="${EVAL_OUTPUT_DIR}/qwen3vl8b_qlora.jsonl"

run_base() {
  echo "Running Qwen3-VL-8B base inference"
  rm -f "${BASE_OUTPUT}"
  swift infer \
    --model "${MODEL_PATH}" \
    --load_args false \
    --val_dataset "${VAL_DATASET}" \
    --infer_backend transformers \
    --torch_dtype bfloat16 \
    --max_batch_size "${MAX_BATCH_SIZE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0 \
    --stream false \
    --val_dataset_shuffle false \
    --result_path "${BASE_OUTPUT}"
}

run_lora() {
  if [[ -z "${ADAPTER_PATH}" ]]; then
    echo "Set ADAPTER_PATH before running QLoRA inference." >&2
    exit 1
  fi
  echo "Running Qwen3-VL-8B + QLoRA inference"
  rm -f "${LORA_OUTPUT}"
  swift infer \
    --model "${MODEL_PATH}" \
    --adapters "${ADAPTER_PATH}" \
    --load_args false \
    --val_dataset "${VAL_DATASET}" \
    --infer_backend transformers \
    --torch_dtype bfloat16 \
    --max_batch_size "${MAX_BATCH_SIZE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0 \
    --stream false \
    --val_dataset_shuffle false \
    --result_path "${LORA_OUTPUT}"
}

case "${MODE}" in
  base) run_base ;;
  lora) run_lora ;;
  both)
    run_base
    run_lora
    ;;
  *)
    echo "Usage: bash eval/infer_compare.sh [base|lora|both]" >&2
    exit 1
    ;;
esac

echo "Inference outputs: ${EVAL_OUTPUT_DIR}"
