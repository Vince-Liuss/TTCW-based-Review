#!/bin/bash
# Auto-running evaluator — grouped by model to avoid reloading between
# standard and cross-mode runs that share the same model weights.
#
# Safe to rerun: a run is skipped if its output JSON already exists
# (written only on full success by evaluate_vllm.py).
# Predictions cache survives crashes so a rerun skips vLLM inference.
#
# Usage:
#   bash evaluate_ttcw_auto.sh           # skip already-completed runs
#   bash evaluate_ttcw_auto.sh --fresh   # wipe all output JSONs and caches first

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_LEVEL=NVL
export VLLM_WORKER_MULTIPROC_METHOD=spawn

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export WANDB_DIR="${PROJECT_ROOT}/logs/wandb"
mkdir -p "${WANDB_DIR}"

EVAL_SCRIPT="${PROJECT_ROOT}/eval/evaluate_vllm.py"

DATASET="/path/to/TTCW_sft_dataset"
BASE_OUTPUT="/path/to/models/ttcw-reviewer"
mkdir -p "${BASE_OUTPUT}"

# -- fresh mode: wipe all previous output JSONs and prediction caches --
for arg in "$@"; do
    if [ "$arg" = "--fresh" ]; then
        echo "[FRESH] Removing all eval output JSONs and prediction caches..."
        rm -f "${BASE_OUTPUT}"/eval_results_*.json
        rm -f "${BASE_OUTPUT}"/*_predictions_cache.json
        echo "[FRESH] Done."
        break
    fi
done

# ---------------------------------------------------------------------------
# run_eval <run_key> <output_file> [evaluate_vllm.py args...]
#   Skips if output_file already exists. Fails immediately on error (no retry).
# ---------------------------------------------------------------------------
run_eval() {
    local run_key="$1"
    local output_file="$2"
    shift 2

    if [ -f "${output_file}" ]; then
        echo "[SKIP] ${run_key} — output file exists"
        return 0
    fi

    echo ""
    echo "[RUN] ${run_key}"
    if python "${EVAL_SCRIPT}" "$@" --output "${output_file}"; then
        echo "[OK]  ${run_key}"
        return 0
    fi
    echo "[FAIL] ${run_key} — exit code $? (skipping)"
    return 1
}

# ---------------------------------------------------------------------------

MODELS=("Qwen3-8B" "Qwen3-4B")

for MODEL_SHORT in "${MODELS[@]}"; do
    echo ""
    echo "========== Model: ${MODEL_SHORT} =========="

    # --- score_with_reviews model: standard then cross-mode (same weights) ---
    run_eval "${MODEL_SHORT}-score_with_reviews" \
        "${BASE_OUTPUT}/eval_results_${MODEL_SHORT}_score_with_reviews.json" \
        --model         "${BASE_OUTPUT}/${MODEL_SHORT}-score_with_reviews" \
        --dataset       "${DATASET}" \
        --mode          "score_with_reviews" \
        --wandb_project "TTCW_reviewer_evaluation" \
        --wandb_run     "eval-vllm-${MODEL_SHORT}-score_with_reviews" \
        || true

    run_eval "${MODEL_SHORT}-score_with_reviews-thinkon" \
        "${BASE_OUTPUT}/eval_results_${MODEL_SHORT}_score_with_reviews_thinkon.json" \
        --model         "${BASE_OUTPUT}/${MODEL_SHORT}-score_with_reviews" \
        --dataset       "${DATASET}" \
        --mode          "score_with_reviews" \
        --enable_thinking true \
        --wandb_project "TTCW_reviewer_evaluation" \
        --wandb_run     "eval-vllm-${MODEL_SHORT}-score_with_reviews-thinkon" \
        || true

    # --- review_with_reasoning model: standard then cross-mode (same weights) ---
    run_eval "${MODEL_SHORT}-review_with_reasoning" \
        "${BASE_OUTPUT}/eval_results_${MODEL_SHORT}_review_with_reasoning.json" \
        --model         "${BASE_OUTPUT}/${MODEL_SHORT}-review_with_reasoning" \
        --dataset       "${DATASET}" \
        --mode          "review_with_reasoning" \
        --wandb_project "TTCW_reviewer_evaluation" \
        --wandb_run     "eval-vllm-${MODEL_SHORT}-review_with_reasoning" \
        || true

    run_eval "${MODEL_SHORT}-review_with_reasoning-thinkoff" \
        "${BASE_OUTPUT}/eval_results_${MODEL_SHORT}_review_with_reasoning_thinkoff.json" \
        --model         "${BASE_OUTPUT}/${MODEL_SHORT}-review_with_reasoning" \
        --dataset       "${DATASET}" \
        --mode          "review_with_reasoning" \
        --enable_thinking false \
        --wandb_project "TTCW_reviewer_evaluation" \
        --wandb_run     "eval-vllm-${MODEL_SHORT}-review_with_reasoning-thinkoff" \
        || true

    echo "========== Done: ${MODEL_SHORT} =========="
done

echo ""
echo "All evaluations complete."
echo ""
echo "Summary:"
for MODEL_SHORT in "${MODELS[@]}"; do
    for SUFFIX in \
        "score_with_reviews" \
        "score_with_reviews_thinkon" \
        "review_with_reasoning" \
        "review_with_reasoning_thinkoff"; do
        f="${BASE_OUTPUT}/eval_results_${MODEL_SHORT}_${SUFFIX}.json"
        [ -f "$f" ] \
            && echo "  [OK]   ${MODEL_SHORT}-${SUFFIX}" \
            || echo "  [MISS] ${MODEL_SHORT}-${SUFFIX}"
    done
done
