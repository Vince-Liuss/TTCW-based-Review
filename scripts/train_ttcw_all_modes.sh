#!/bin/bash
set -e


PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export WANDB_DIR="${PROJECT_ROOT}/logs/wandb"
mkdir -p "${WANDB_DIR}"

cd "${PROJECT_ROOT}/train"
mkdir -p logs

DATASET="/path/to/TTCW_sft_dataset"
BASE_OUTPUT="/path/to/models/ttcw-reviewer"
EPOCHS=1

MODES=(
    # "score_only"
    # "score_with_reasoning"
    "score_with_reviews"
    "review_with_reasoning"
)

declare -A SEQ_LENGTH
SEQ_LENGTH["score_only"]=8192
SEQ_LENGTH["score_with_reasoning"]=24576
SEQ_LENGTH["score_with_reviews"]=16384
SEQ_LENGTH["review_with_reasoning"]=32768

declare -A THINKING_BUDGET
THINKING_BUDGET["score_with_reasoning"]=8192
THINKING_BUDGET["review_with_reasoning"]=8192

ACCUM_STEPS=8

# Models to train: "short_name|hf_model_id"
MODELS=(
    # "Qwen3-8B|Qwen/Qwen3-8B"  # already trained
    "Qwen3-4B|Qwen/Qwen3-4B"
)

# Per-model LoRA rank (r=48 for 4B gives ~2.3% trainable, matching 8B's 2.13% at r=64)
declare -A LORA_R
LORA_R["Qwen3-8B"]=64
LORA_R["Qwen3-4B"]=48

# Per-model batch sizes: MODEL_SHORT__MODE=N
declare -A BATCH_SIZE
BATCH_SIZE["Qwen3-8B__score_only"]=16
BATCH_SIZE["Qwen3-8B__score_with_reasoning"]=3
BATCH_SIZE["Qwen3-8B__score_with_reviews"]=7
BATCH_SIZE["Qwen3-8B__review_with_reasoning"]=2
BATCH_SIZE["Qwen3-4B__score_only"]=16
BATCH_SIZE["Qwen3-4B__score_with_reasoning"]=4
BATCH_SIZE["Qwen3-4B__score_with_reviews"]=8
BATCH_SIZE["Qwen3-4B__review_with_reasoning"]=3

for MODEL_ENTRY in "${MODELS[@]}"; do
    MODEL_SHORT="${MODEL_ENTRY%%|*}"
    MODEL="${MODEL_ENTRY##*|}"

    for MODE in "${MODES[@]}"; do
        RUN_NAME="${MODEL_SHORT}-${MODE}"
        MODEL_PATH="${BASE_OUTPUT}/${RUN_NAME}"
        EVAL_OUTPUT="${BASE_OUTPUT}/eval_results_${RUN_NAME}.json"
        BS="${BATCH_SIZE[${MODEL_SHORT}__${MODE}]}"
        LR="${LORA_R[${MODEL_SHORT}]}"

        echo "=========================================="
        echo "Model: ${MODEL_SHORT}  Mode: ${MODE}"
        echo "=========================================="

        accelerate launch --deepspeed_config_file "${PROJECT_ROOT}/config/ds_config.json" train.py \
            --dataset         "${DATASET}" \
            --model           "${MODEL}" \
            --output_dir      "${BASE_OUTPUT}" \
            --messages_column "${MODE}" \
            --epochs          "${EPOCHS}" \
            --max_seq_length  "${SEQ_LENGTH[${MODE}]}" \
            --batch_size      "${BS}" \
            --accumulation_steps "${ACCUM_STEPS}" \
            --learning_rate   2e-4 \
            --lora_r          "${LR}" \
            --lora_alpha      $((LR * 2)) \
            --wandb_project   "TTCW_reviewer"

        BUDGET_ARG=""
        if [[ -n "${THINKING_BUDGET[${MODE}]+x}" ]]; then
            BUDGET_ARG="--thinking_budget ${THINKING_BUDGET[${MODE}]}"
        fi
        echo "Running post-training vLLM evaluation for ${RUN_NAME}..."
        python "${PROJECT_ROOT}/eval/evaluate_vllm.py" \
            --model          "${MODEL_PATH}" \
            --dataset        "${DATASET}" \
            --mode           "${MODE}" \
            --max_seq_length "${SEQ_LENGTH[${MODE}]}" \
            --wandb_project  "TTCW_reviewer_evaluation" \
            --wandb_run      "eval-vllm-${RUN_NAME}" \
            --output         "${EVAL_OUTPUT}" \
            ${BUDGET_ARG}

        echo "Finished: ${RUN_NAME}"
        echo ""
    done
done

echo "All training + evaluation runs complete."
