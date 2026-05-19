"""
evaluate_vllm.py
================
Post-training evaluation using vLLM offline inference on the held-out test
split (70%). Computes score_accuracy = exp(-mae) per TTCW metric, BERTScore
F1 for review modes, and the composite eval_score. Results are saved to JSON
and logged to W&B.

Usage:
    python evaluate_vllm.py \
        --model   /path/to/models/ttcw-reviewer/Qwen3-8B-score_with_reviews \
        --dataset /path/to/TTCW_sft_dataset \
        --mode    score_with_reviews \
        --output  /path/to/models/ttcw-reviewer/eval_results_score_with_reviews.json
"""

import argparse
import gc
import json
import os
import re
import sys
import wandb
import evaluate as hf_evaluate

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_from_disk
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
from compute_metrics import (
    TTCW_METRICS, LABEL_TO_KEY, _REASONING_MODES, _REVIEW_MODES,
    _strip_thinking, parse_review_output, parse_score_output,
)

def _extract_thinking(text: str) -> str:
    """Return the content inside the first <think>…</think> block, or empty string."""
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return m.group(1).strip() if m else ""



TRAIN_MODE_TO_COLUMN: dict[str, str] = {
    "score_with_reviews":    "messages_score_with_reviews",
    "review_with_reasoning": "messages_review_with_reasoning",
}


def prepare(ds, mode: str):
    col  = TRAIN_MODE_TO_COLUMN[mode]
    flat = ds.select_columns([col]).flatten()
    return flat.rename_columns({f"{col}.prompt": "prompt", f"{col}.completion": "completion"})


def extract_gt_text(example) -> str:
    return " ".join(
        msg["content"] for msg in example["completion"] if msg["role"] == "assistant"
    )


def run_inference(llm, tokenizer, dataset, sampling_params, enable_thinking: bool,
                  thinking_budget: int | None = None, thinking_via_system: bool = False) -> list[str]:
    if thinking_via_system:
        system_msg = {"role": "system", "content": "detailed thinking on" if enable_thinking else "detailed thinking off"}
        prompts = [
            tokenizer.apply_chat_template(
                [system_msg] + list(ex["prompt"]),
                tokenize=False, add_generation_prompt=True,
            )
            for ex in dataset
        ]
    else:
        template_kwargs: dict = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": enable_thinking}
        if enable_thinking and thinking_budget is not None:
            template_kwargs["thinking_budget"] = thinking_budget
        prompts = [
            tokenizer.apply_chat_template(ex["prompt"], **template_kwargs)
            for ex in dataset
        ]
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]



def compute_results(predictions: list[str], dataset, mode: str, enable_thinking_inference: bool | None = None) -> dict[str, float]:
    has_thinking_gt = mode in _REASONING_MODES   # whether GT contains <think> blocks
    has_reviews     = mode in _REVIEW_MODES
    # None means "match training mode"; explicit bool supports cross-mode tests
    if enable_thinking_inference is None:
        enable_thinking_inference = has_thinking_gt

    gt_texts = [extract_gt_text(ex) for ex in dataset]
    if has_thinking_gt:
        gt_texts    = [_strip_thinking(t) for t in gt_texts]
    if enable_thinking_inference:
        predictions = [_strip_thinking(t) for t in predictions]

    pred_scores = [parse_score_output(t) for t in predictions]
    gt_scores   = [parse_score_output(t) for t in gt_texts]

    results: dict[str, float] = {}

    # Drop predictions that are missing any of the 14 metrics — partial outputs
    # would skew per-metric averages and can't contribute to BERTScore either.
    complete_mask = [all(p[k] is not None for k in TTCW_METRICS) for p in pred_scores]
    n_incomplete = complete_mask.count(False)
    if n_incomplete:
        print(f"Skipping {n_incomplete}/{len(predictions)} predictions with incomplete metrics")
        results["incomplete_predictions"] = n_incomplete
    predictions = [x for x, ok in zip(predictions, complete_mask) if ok]
    gt_texts    = [x for x, ok in zip(gt_texts,    complete_mask) if ok]
    pred_scores = [x for x, ok in zip(pred_scores, complete_mask) if ok]
    gt_scores   = [x for x, ok in zip(gt_scores,   complete_mask) if ok]

    parse_rate = len(predictions) / max(len(complete_mask), 1)
    results["score_parse_rate"] = round(parse_rate, 4)

    # ---- score_accuracy = exp(-mae) ----
    macro_accuracies: list[float] = []
    for key in TTCW_METRICS:
        errors = [
            abs(p[key] - g[key])
            for p, g in zip(pred_scores, gt_scores)
        ]
        if errors:
            mae = float(np.mean(errors))
            acc = float(np.exp(-mae))
            results[f"score_accuracy/{key}"] = round(acc, 4)
            macro_accuracies.append(acc)

    macro_accuracy: float | None = None
    if macro_accuracies:
        arr = np.array(macro_accuracies)
        macro_accuracy = float(np.mean(arr))
        results["score_accuracy/macro"]  = round(macro_accuracy, 4)
        results["score_accuracy/std"]    = round(float(np.std(arr)), 4)
        results["score_accuracy/median"] = round(float(np.median(arr)), 4)
        results["score_accuracy/min"]    = round(float(np.min(arr)), 4)
        results["score_accuracy/max"]    = round(float(np.max(arr)), 4)
        # Per-dimension averages (Fluency/Flexibility/Originality/Elaboration)
        # Use short keys (Fluency1…Elaboration3) for startswith — TTCW_METRICS holds full labels
        _dim_accs: dict[str, list[float]] = {"Fluency": [], "Flexibility": [], "Originality": [], "Elaboration": []}
        for label, acc in zip(TTCW_METRICS, macro_accuracies):
            short_key = LABEL_TO_KEY.get(label, "")
            for dim in _dim_accs:
                if short_key.startswith(dim):
                    _dim_accs[dim].append(acc)
        for dim, vals in _dim_accs.items():
            if vals:
                results[f"score_accuracy/dim_{dim.lower()}"] = round(float(np.mean(vals)), 4)

    # ---- BERTScore (review modes only) ----
    # Run sequentially in the main process on cuda:0 to avoid CUDA context
    # conflicts with vLLM's tensor-parallel workers (spawning subprocesses while
    # vLLM contexts are partially live causes segfaults).
    bertscore_f1: float | None = None
    if has_reviews:
        bs_metric = hf_evaluate.load("bertscore")
        pred_reviews = [parse_review_output(t) for t in predictions]
        gt_reviews   = [parse_review_output(t) for t in gt_texts]

        bs_per_metric: list[float] = []
        bertscore_chunk = 256
        for key in tqdm(TTCW_METRICS, desc="BERTScore"):
            pairs = [
                (p[key], g[key])
                for p, g in zip(pred_reviews, gt_reviews)
                if p[key] and g[key]
            ]
            if not pairs:
                continue
            preds_key = [p for p, _ in pairs]
            refs_key  = [g for _, g in pairs]
            truncated = 0
            for i, t in enumerate(preds_key):
                words = t.split()
                if len(words) > 500:
                    print(f"  [LONG] {key} pred[{i}] = {len(words)} words — truncating to 500")
                    preds_key[i] = " ".join(words[:500])
                    truncated += 1
            if truncated:
                results[f"truncated_reviews/{key}"] = truncated
            all_f1: list[float] = []
            for start in range(0, len(preds_key), bertscore_chunk):
                chunk_result = bs_metric.compute(
                    predictions=preds_key[start : start + bertscore_chunk],
                    references=refs_key[start : start + bertscore_chunk],
                    model_type="microsoft/deberta-xlarge-mnli",
                    device=torch.device("cuda:0"),
                    lang="en",
                    batch_size=4,
                )
                all_f1.extend(chunk_result["f1"])
                torch.cuda.empty_cache()
            f1 = round(float(np.mean(all_f1)), 4)
            results[f"bertscore_f1/{key}"] = f1
            bs_per_metric.append(f1)

        bertscore_f1 = round(float(np.mean(bs_per_metric)), 4) if bs_per_metric else 0.0
        results["review_bertscore_f1"] = bertscore_f1

    # ---- composite eval_score ----
    if macro_accuracy is not None:
        if has_reviews:
            raw = 0.5 * macro_accuracy + 0.5 * (bertscore_f1 or 0.0)
        else:
            raw = macro_accuracy
        results["eval_score"] = round(raw * parse_rate, 4)
    else:
        results["eval_score"] = 0.0

    return results


def _update_comparison_artifact(args, results: dict, run_name: str, enable_thinking: bool) -> None:
    """Append this run's per-metric scores to a persistent artifact table shared across all eval runs."""
    short_keys = [LABEL_TO_KEY.get(m, m) for m in TTCW_METRICS]

    columns = (
        ["run_name", "model", "mode", "enable_thinking",
         "score_accuracy_macro", "score_accuracy_std", "parse_rate", "bertscore_f1", "eval_score"]
        + [f"acc_{k}" for k in short_keys]
        + [f"bs_{k}"  for k in short_keys]
    )

    new_row = (
        [
            run_name,
            os.path.basename(args.model.rstrip("/")),
            args.mode,
            enable_thinking,
            results.get("score_accuracy/macro"),
            results.get("score_accuracy/std"),
            results.get("score_parse_rate"),
            results.get("review_bertscore_f1"),
            results.get("eval_score"),
        ]
        + [results.get(f"score_accuracy/{m}") for m in TTCW_METRICS]
        + [results.get(f"bertscore_f1/{m}")   for m in TTCW_METRICS]
    )

    existing_data: list = []
    try:
        api = wandb.Api()
        artifact = api.artifact(f"{wandb.run.entity}/{args.wandb_project}/eval_comparison:latest")
        tbl = artifact.get("results")
        existing_data = [list(row) for row in tbl.data if list(row)[0] != run_name]
    except Exception:
        pass  # First run or artifact not yet created — start fresh

    existing_data.append(new_row)
    comparison_table = wandb.Table(columns=columns, data=existing_data)
    # Log as a panel so the full accumulated table is visible on the run page
    wandb.log({"eval_comparison": comparison_table})
    art = wandb.Artifact(
        "eval_comparison", type="eval_results",
        description="Accumulated per-metric evaluation results across all model/mode runs",
    )
    art.add(comparison_table, "results")
    wandb.run.log_artifact(art)
    print(f"Updated eval_comparison artifact ({len(existing_data)} rows total)")


def _update_samples_artifact(args, predictions: list[str], dataset, run_name: str, enable_thinking: bool) -> None:
    """Append qualitative samples from this run to a persistent artifact table for cross-run comparison."""
    model_short = os.path.basename(args.model.rstrip("/"))
    has_thinking_gt = args.mode in _REASONING_MODES
    columns = ["run_name", "model", "mode", "enable_thinking", "story_idx", "pred_thinking", "pred_answer", "gt_answer"]

    n = min(args.num_samples, len(predictions))
    new_rows = []
    for idx in range(n):
        gt_raw   = extract_gt_text(dataset[idx])
        pred_raw = predictions[idx]
        gt_answer     = _strip_thinking(gt_raw)     if has_thinking_gt else gt_raw
        pred_thinking = _extract_thinking(pred_raw) if enable_thinking else ""
        pred_answer   = _strip_thinking(pred_raw)   if enable_thinking else pred_raw
        new_rows.append([run_name, model_short, args.mode, enable_thinking, idx, pred_thinking, pred_answer, gt_answer])

    existing_data: list = []
    try:
        api = wandb.Api()
        artifact = api.artifact(f"{wandb.run.entity}/{args.wandb_project}/reasoning_samples:latest")
        tbl = artifact.get("samples")
        existing_data = [list(row) for row in tbl.data if list(row)[0] != run_name]
    except Exception:
        pass

    existing_data.extend(new_rows)
    samples_table = wandb.Table(columns=columns, data=existing_data)
    art = wandb.Artifact(
        "reasoning_samples", type="eval_samples",
        description="Accumulated qualitative reasoning samples across all model/mode runs",
    )
    art.add(samples_table, "samples")
    wandb.run.log_artifact(art)
    print(f"Updated reasoning_samples artifact ({len(existing_data)} rows total)")


def main():
    parser = argparse.ArgumentParser(description="Post-training vLLM evaluation on test split")
    parser.add_argument("--model",          required=True, help="Path to merged trained model")
    parser.add_argument("--dataset",        required=True, help="Path to TTCW HF dataset")
    parser.add_argument("--mode",           required=True, choices=list(TRAIN_MODE_TO_COLUMN))
    parser.add_argument("--output",         required=True, help="Output path for results JSON")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--tp",             type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--thinking_budget",  type=int, default=None, help="Max thinking tokens cap (default: no cap)")
    parser.add_argument("--enable_thinking",  default=None, choices=["true", "false"],
                        help="Override enable_thinking at inference (default: auto from mode)")
    parser.add_argument("--num_samples",      type=int, default=20,
                        help="Number of fixed test stories to log as qualitative samples (0 to skip)")
    parser.add_argument("--debug",            action="store_true",
                        help="Run on 100 samples only and print raw outputs")
    parser.add_argument("--wandb_project",  default="TTCW_reviewer")
    parser.add_argument("--wandb_run",      default="", help="W&B run name for logging results")
    args = parser.parse_args()
    # ---- test split ----
    dataset_dict = load_from_disk(args.dataset)
    raw_test = dataset_dict["test"]
    if args.debug:
        raw_test = raw_test.select(range(300))
    else:
        raw_test = raw_test.select(range(min(500, len(raw_test))))
    test_ds = prepare(raw_test, args.mode)
    print(f"Test size: {len(test_ds)}")

    max_seq_length = 32768
    has_thinking_gt = args.mode in _REASONING_MODES
    # Resolve enable_thinking: CLI override takes precedence, else follow training mode
    if args.enable_thinking is not None:
        enable_thinking = args.enable_thinking == "true"
    else:
        enable_thinking = has_thinking_gt
    max_tokens = max_seq_length - 5120

    # ---- inference (with cache to survive crashes before BERTScore) ----
    predictions_cache = args.output.replace(".json", "_predictions_cache.json")
    if os.path.exists(predictions_cache):
        print(f"Loading cached predictions from {predictions_cache} (skipping vLLM inference)")
        with open(predictions_cache) as f:
            predictions = json.load(f)
    else:
        # ---- model (only loaded when inference is needed) ----
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tp,
            dtype="bfloat16",
            trust_remote_code=True,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            max_model_len=max_seq_length,
            max_num_batched_tokens=max_seq_length,
        )
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        print(f"Running inference on test split ({args.mode}, enable_thinking={enable_thinking})...")
        thinking_via_system = "nemotron" in args.model.lower()
        predictions = run_inference(llm, tokenizer, test_ds, sampling_params, enable_thinking,
                                    thinking_budget=args.thinking_budget if enable_thinking else None,
                                    thinking_via_system=thinking_via_system)
        with open(predictions_cache, "w") as f:
            json.dump(predictions, f)
        print(f"Predictions cached to {predictions_cache}")
        # Release vLLM workers before BERTScore loads onto the GPU.
        # destroy_model_parallel / destroy_distributed_environment cause segfaults
        # on newer vLLM versions — gc.collect is the safe alternative.
        del llm
        gc.collect()
    torch.cuda.empty_cache()

    if args.debug:
        for i, (pred, ex) in enumerate(zip(predictions[:5], test_ds)):
            print(f"\n{'='*60} SAMPLE {i} {'='*60}")
            print(f"--- OUTPUT ({len(pred)} chars) ---\n{pred[:2000]}")

    # ---- metrics ----
    results = compute_results(predictions, test_ds, args.mode, enable_thinking_inference=enable_thinking)
    print(f"Results: {json.dumps(results, indent=2)}")

    # ---- W&B ----
    model_short = os.path.basename(args.model.rstrip("/"))
    thinking_suffix = f"-think{'on' if enable_thinking else 'off'}"
    run_name = args.wandb_run or f"eval-{model_short}-{args.mode}{thinking_suffix}"
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            resume="never",
            config={
                "model":          model_short,
                "mode":           args.mode,
                "enable_thinking": enable_thinking,
                "max_seq_length": max_seq_length,
                "test_size":      len(test_ds),
            },
        )
        wandb.log(results)
        _update_comparison_artifact(args, results, run_name, enable_thinking)
        if args.num_samples > 0:
            _update_samples_artifact(args, predictions, test_ds, run_name, enable_thinking)
        wandb.finish()

    # ---- save ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"mode": args.mode, "test_size": len(test_ds), "results": results}, f, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
