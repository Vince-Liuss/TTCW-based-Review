#!/bin/bash
set -e

export NCCL_P2P_LEVEL=NVL
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export WANDB_DIR="${PROJECT_ROOT}/logs/wandb"
mkdir -p "${WANDB_DIR}"

cd "${PROJECT_ROOT}/train"
mkdir -p logs

DATASET="/path/to/TTCW_sft_dataset"
BASE_OUTPUT="/path/to/models/ttcw-reviewer"
MODEL="nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
MODEL_SHORT="Nemotron-8B"
EPOCHS=1
LORA_R=64

declare -A SEQ_LENGTH
SEQ_LENGTH["score_with_reviews"]=16384
SEQ_LENGTH["review_with_reasoning"]=32768

declare -A BATCH_SIZE
BATCH_SIZE["score_with_reviews"]=4
BATCH_SIZE["review_with_reasoning"]=1

declare -A ACCUM_STEPS_MAP
ACCUM_STEPS_MAP["score_with_reviews"]=8
ACCUM_STEPS_MAP["review_with_reasoning"]=8

MODES=(
    "review_with_reasoning"
)

for MODE in "${MODES[@]}"; do
    RUN_NAME="${MODEL_SHORT}-${MODE}"

    echo "=========================================="
    echo "Model: ${MODEL_SHORT}  Mode: ${MODE}"
    echo "=========================================="

    accelerate launch --deepspeed_config_file "${PROJECT_ROOT}/config/ds_config.json" train.py \
        --dataset            "${DATASET}" \
        --model              "${MODEL}" \
        --output_dir         "${BASE_OUTPUT}" \
        --messages_column    "${MODE}" \
        --epochs             "${EPOCHS}" \
        --max_seq_length     "${SEQ_LENGTH[${MODE}]}" \
        --batch_size         "${BATCH_SIZE[${MODE}]}" \
        --accumulation_steps "${ACCUM_STEPS_MAP[${MODE}]}" \
        --learning_rate      2e-4 \
        --lora_r             "${LORA_R}" \
        --lora_alpha         $((LORA_R * 2)) \
        --wandb_project      "TTCW_reviewer"

    echo "Finished: ${RUN_NAME}"
    echo ""
done

echo "All Nemotron training runs complete."
