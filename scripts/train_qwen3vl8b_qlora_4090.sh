#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-VL-8B-Instruct}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/portable_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/qwen3vl8b_cv_semantic_qlora}"

if [[ ! -f "${DATA_ROOT}/runtime_data/train.jsonl" ]]; then
  echo "Missing training data: ${DATA_ROOT}/runtime_data/train.jsonl" >&2
  exit 1
fi
if [[ ! -f "${DATA_ROOT}/runtime_data/val.jsonl" ]]; then
  echo "Missing validation data: ${DATA_ROOT}/runtime_data/val.jsonl" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Missing model directory: ${MODEL_PATH}" >&2
  exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-6}"

swift sft \
  --model "${MODEL_PATH}" \
  --dataset "${DATA_ROOT}/runtime_data/train.jsonl" \
  --val_dataset "${DATA_ROOT}/runtime_data/val.jsonl" \
  --tuner_type lora \
  --quant_method bnb \
  --quant_bits 4 \
  --bnb_4bit_quant_type nf4 \
  --bnb_4bit_compute_dtype bfloat16 \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --gradient_checkpointing true \
  --max_length 2048 \
  --eval_steps 500 \
  --save_steps 500 \
  --save_total_limit 3 \
  --logging_steps 5 \
  --warmup_ratio 0.05 \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --load_best_model_at_end true \
  --metric_for_best_model loss \
  --greater_is_better false \
  --eval_strategy steps \
  --save_strategy steps \
  --output_dir "${OUTPUT_DIR}"
