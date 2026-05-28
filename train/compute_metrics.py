from __future__ import annotations

import re
from accelerate import PartialState
import evaluate
import numpy as np
import torch
import wandb

# ---------------------------------------------------------------------------
# TTCW constants
# ---------------------------------------------------------------------------

# Maps short dataset column keys → full label names output by the model
METRIC_LABELS = {
    "Fluency1": "Narrative Pacing (Compression/Stretching)",
    "Fluency2": "Scene vs Exposition Balance",
    "Fluency3": "Language Proficiency & Literary Devices",
    "Fluency4": "Narrative Ending Quality",
    "Fluency5": "Understandability & Coherence",
    "Flexibility1": "Perspective & Voice Flexibility",
    "Flexibility2": "Emotional Flexibility (Interiority/Exteriority)",
    "Flexibility3": "Structural Flexibility (Surprising but Appropriate Turns)",
    "Originality1": "Originality in Theme and Takeaway",
    "Originality2": "Originality in Thought (Cliche Avoidance)",
    "Originality3": "Originality in Form/Structure",
    "Elaboration1": "World-Building and Sensory Believability",
    "Elaboration2": "Character Development Depth",
    "Elaboration3": "Rhetorical Complexity (Surface vs Subtext)",
}

# Primary keys are the full label names (as the model outputs them)
TTCW_METRICS: list[str] = list(METRIC_LABELS.values())

# Reverse: full label name → short dataset column key
LABEL_TO_KEY: dict[str, str] = {v: k for k, v in METRIC_LABELS.items()}

# Loaded lazily on first use so score-only training doesn't pull in DeBERTa
_bertscore = None


def _get_bertscore():
    global _bertscore
    if _bertscore is None:
        _bertscore = evaluate.load("bertscore")
    return _bertscore


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------


def parse_score_output(text: str) -> dict[str, float | None]:
    """
    Parse predicted per-metric scores from model output.

    Handles two formats:
        {Metric Label}: {score}/10               (score_only / score_with_reasoning)
        {Metric Label} | Score: {score}/10       (score_with_reviews / review_with_reasoning)
    """
    scores: dict[str, float | None] = {label: None for label in TTCW_METRICS}
    # Format 1: "Label: score/10"
    pattern = re.compile(r"^(.+?):\s*(\d+(?:\.\d+)?)\s*/\s*10", re.MULTILINE)
    for match in pattern.finditer(text):
        label = match.group(1).strip()
        if label in scores:
            scores[label] = float(match.group(2))
    # Format 2: "Label | Score: score/10" (review modes)
    pattern2 = re.compile(r"^(.+?)\s*\|\s*Score:\s*(\d+(?:\.\d+)?)\s*/\s*10", re.MULTILINE)
    for match in pattern2.finditer(text):
        label = match.group(1).strip()
        if label in scores and scores[label] is None:
            scores[label] = float(match.group(2))
    return scores


def parse_review_output(text: str) -> dict[str, str | None]:
    """
    Parse predicted per-metric review text from score-with-reviews model output.

    Matches section headers:
        {Metric Label} | Score: {score}/10
    and extracts the review text that follows each header.
    """
    reviews: dict[str, str | None] = {label: None for label in TTCW_METRICS}
    header_re = re.compile(
        r"^(.+?)\s*\|\s*Score:\s*\d+(?:\.\d+)?\s*/\s*10\s*$", re.MULTILINE
    )
    headers = list(header_re.finditer(text))
    for i, match in enumerate(headers):
        label = match.group(1).strip()
        if label not in reviews:
            continue
        start = match.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[start:end].strip()
        if body:
            reviews[label] = body
    return reviews


# ---------------------------------------------------------------------------
# Trainer-compatible distributed compute_metrics
# ---------------------------------------------------------------------------


def preprocess_logits_for_metrics(logits: torch.Tensor | tuple, labels: torch.Tensor) -> torch.Tensor:
    """Reduce logits to argmax token IDs before they are stored in CPU RAM.

    Handles two shapes:
    - Standard: (batch, seq_len, vocab) — shift argmax right by 1 so
      token_ids[b][j] is the prediction FOR labels[b][j].
    - Liger LCE: (N_kept, vocab) — only completion-position logits.
      Reconstructs (batch, seq_len) by placing argmax at kept positions
      and 0 at prompt positions.
    """
    if isinstance(logits, tuple):
        logits = logits[0]

    if logits.dim() == 3:
        # logits[:, j, :] predicts position j+1; shift left by 1 so
        # pred_ids[:, j] aligns with labels[:, j].
        pred_ids = logits.argmax(dim=-1)                              # (batch, seq_len)
        pad = torch.zeros(pred_ids.size(0), 1, dtype=torch.long, device=pred_ids.device)
        return torch.cat([pad, pred_ids[:, :-1]], dim=1)             # (batch, seq_len)

    # Liger LCE path: logits are only emitted for completion positions.
    # Liger shifts labels by 1 internally, so kept positions correspond
    # to labels[:, 1:] != -100.
    batch_size, seq_len = labels.shape
    completion_mask = labels[:, 1:].reshape(-1) != -100              # (batch*(seq_len-1),)
    pred_ids = torch.zeros(batch_size * (seq_len - 1), dtype=torch.long, device=logits.device)
    pred_ids[completion_mask] = logits.argmax(dim=-1)
    pred_ids = pred_ids.reshape(batch_size, seq_len - 1)
    pad = torch.zeros(batch_size, 1, dtype=torch.long, device=logits.device)
    return torch.cat([pad, pred_ids], dim=1)                         # (batch, seq_len)


_REVIEW_MODES = {"score_with_reviews", "review_with_reasoning"}
_REASONING_MODES = {"score_with_reasoning", "review_with_reasoning"}


def _strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>…</think> reasoning block so parsers only see the final output.
    If <think> is present but </think> is missing (truncated/runaway generation), return ""
    since no answer content was produced."""
    if "<think>" in text and "</think>" not in text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def make_compute_metrics(
    tokenizer,
    wandb_prefix: str = "eval",
    distributed_state: PartialState | None = None,
    mode: str = "score_only",
):
    """Return a compute_metrics function for SFTTrainer.

    Uses teacher-forced argmax decoding (fast, fully distributed via the
    trainer's built-in gather). Each rank evaluates its shard; predictions
    are gathered by the trainer before this function is called on rank-0.

    Pass wandb_prefix="eval" for mid-training eval steps and
    wandb_prefix="test" for the final held-out test pass.

    mode must match the messages_column used for training so the function
    knows whether to expect reviews (BERTScore) and whether to strip
    <think> blocks before parsing.

    MAE is mapped to (0, 1] via score_accuracy = exp(-mae), making it
    directly comparable to BERTScore F1 (also [0, 1]).
    """
    device = distributed_state.device if distributed_state is not None else torch.device("cpu")
    has_reviews  = mode in _REVIEW_MODES
    has_thinking = mode in _REASONING_MODES

    def compute_metrics(eval_pred):
        token_ids, labels = eval_pred
        # token_ids: (N, seq_len) int ids from preprocess_logits_for_metrics
        # labels:    (N, seq_len) with -100 at prompt/padding positions

        # Decode only completion positions (labels != -100).
        pred_texts: list[str] = []
        gt_texts:   list[str] = []
        for i in range(len(token_ids)):
            mask = labels[i] != -100
            pred_texts.append(tokenizer.decode(token_ids[i][mask], skip_special_tokens=True))
            gt_texts.append(  tokenizer.decode(labels[i][mask],    skip_special_tokens=True))


        # Strip <thinking> blocks before parsing so reasoning traces don't
        # produce false score/review matches.
        if has_thinking:
            pred_texts = [_strip_thinking(t) for t in pred_texts]
            gt_texts   = [_strip_thinking(t) for t in gt_texts]

        pred_scores  = [parse_score_output(t)  for t in pred_texts]
        gt_scores    = [parse_score_output(t)  for t in gt_texts]

        results: dict[str, float] = {}

        # ---- score accuracy: exp(-mae) ∈ (0, 1] ----
        macro_accuracies: list[float] = []
        for key in TTCW_METRICS:
            # Missing predictions are skipped; only paired (pred, gt) scores contribute.
            errors = [
                abs(p[key] - g[key])
                for p, g in zip(pred_scores, gt_scores)
                if p[key] is not None and g[key] is not None
            ]
            if errors:
                mae = float(np.mean(errors))
                acc = float(np.exp(-mae))
                results[f"score_accuracy/{key}"] = round(acc, 4)
                macro_accuracies.append(acc)

        macro_accuracy: float | None = None
        if macro_accuracies:
            macro_accuracy = float(np.mean(macro_accuracies))
            results["score_accuracy/macro"] = round(macro_accuracy, 4)

        parse_rate = sum(
            1 for p in pred_scores if all(p[k] is not None for k in TTCW_METRICS)
        ) / max(len(pred_scores), 1)
        results["score_parse_rate"] = round(parse_rate, 4)

        # ---- BERTScore for review quality (review modes only) ----
        bertscore_f1: float | None = None
        if has_reviews:
            pred_reviews = [parse_review_output(t) for t in pred_texts]
            gt_reviews   = [parse_review_output(t) for t in gt_texts]
            bs_per_metric: list[float] = []
            for key in TTCW_METRICS:
                pairs = [
                    (p[key], g[key])
                    for p, g in zip(pred_reviews, gt_reviews)
                    if p[key] and g[key]
                ]
                if not pairs:
                    continue
                bs_result = _get_bertscore().compute(
                    predictions=[p for p, _ in pairs],
                    references=[g for _, g in pairs],
                    model_type="microsoft/deberta-xlarge-mnli",
                    device=device,
                    lang="en",
                )
                bs_per_metric.append(float(np.mean(bs_result["f1"])))

            # Treat missing reviews as 0 so review modes are penalised
            # when the model fails to produce them.
            bertscore_f1 = round(float(np.mean(bs_per_metric)), 4) if bs_per_metric else 0.0
            results["review_bertscore_f1"] = bertscore_f1

        # ---- composite eval_score ∈ [0, 1] ----
        # Score-only modes: eval_score = score_accuracy * parse_rate
        # Review modes:     eval_score = 0.5 * score_accuracy + 0.5 * bertscore_f1
        #   (missing bertscore treated as 0, so review quality is always penalised)
        if macro_accuracy is not None:
            if has_reviews:
                raw = 0.5 * macro_accuracy + 0.5 * (bertscore_f1 or 0.0)
            else:
                raw = macro_accuracy
            results["eval_score"] = round(raw * parse_rate, 4)
        else:
            results["eval_score"] = 0.0

        if wandb.run is not None:
            wandb.log({f"{wandb_prefix}/{k}": v for k, v in results.items()})

        return results

    return compute_metrics

