import json
import hashlib
import logging
import shutil
import re
from pathlib import Path

from tqdm import tqdm

from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = Path(
    "/path/to/story_evaluation_dataset_synthesized.jsonl"
).resolve()
DEFAULT_OUTPUT_PATH = Path(
    "/path/to/TTCW_sft_dataset"
).resolve()


TTCW_METRICS = [
    "Fluency1",
    "Fluency2",
    "Fluency3",
    "Fluency4",
    "Fluency5",
    "Flexibility1",
    "Flexibility2",
    "Flexibility3",
    "Originality1",
    "Originality2",
    "Originality3",
    "Elaboration1",
    "Elaboration2",
    "Elaboration3",
]
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

# Models excluded from reasoning traces (same set as summarize_reviews.py).
EXCLUDED_MODELS: set[str] = {
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
}

BUILD_CONFIG = {
    "input_path": DEFAULT_INPUT_PATH,
    "output_path": DEFAULT_OUTPUT_PATH,
    "overwrite": True,
    # 2-way split ratio
    "train_ratio": 0.90,
}



def make_messages(user: str, assistant: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": user}],
        "completion": [{"role": "assistant", "content": assistant}],
    }


def build_ttcw_user_prompt(story: str, mode: str = "") -> str:
    """Build the user-turn prompt.  An optional mode trigger word is prepended
    so the fine-tuned model knows which output format is expected."""
    header = f"{mode}\n\n" if mode else ""
    return (
        f"{header}Please evaluate the following story using the TTCW Metrics.\n\n"
        f"Story:\n{story.strip()}"
    )


def build_review_assistant_report(
    metric_reports: list[dict], overall_avg: float
) -> str:
    """Score + LLM-synthesized review per metric."""
    lines = []
    lines.append("Full TTCW Evaluation Report")
    lines.append(f"Overall Average Score: {overall_avg:.2f}/10")
    lines.append("")
    for report in metric_reports:
        lines.append(f"{report['metric']} | Score: {round(report['avg_score'])}/10")
        lines.append(report["synthesized_review"])
        lines.append("")
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# score_only mode — assistant report builder
# ---------------------------------------------------------------------------


def build_score_only_assistant_report(
    metric_reports: list[dict], overall_avg: float
) -> str:
    lines = []
    lines.append("TTCW Score Report")
    lines.append(f"Overall Average Score: {overall_avg:.2f}/10")
    lines.append("")
    for report in metric_reports:
        lines.append(f"{report['metric']}: {report['avg_score']:.2f}/10")
    return "\n".join(lines).strip()


def build_score_with_reasoning_assistant_report(
    metric_reports: list[dict], overall_avg: float, raw_reviews: dict
) -> str:
    """Per-model raw reasoning steps (from the evaluator models) as <thinking>
    chain-of-thought, followed by a score-only summary block.

    Each evaluator model's `reason` field is the step-by-step analysis it
    produced in response to the TTCW prompt (which always asks for reasoning).
    Excluded models (EXCLUDED_MODELS) are omitted, matching summarize_reviews.py.
    """
    reasoning_lines = []
    for report in metric_reports:
        metric_key = report["metric_key"]
        metric_label = report["metric"]
        reasoning_lines.append(f"### {metric_label}")
        model_reasons = raw_reviews.get(metric_key, {})
        valid = [
            (model, data)
            for model, data in model_reasons.items()
            if model not in EXCLUDED_MODELS
            and isinstance(data, dict)
            and isinstance(data.get("reason"), str)
            and data["reason"].strip()
        ]
        for i, (_, data) in enumerate(valid):
            reasoning_lines.append(f"Reviewer {i + 1}:")
            reasoning_lines.append(data["reason"].strip())
            reasoning_lines.append("")
        reasoning_lines.append("")
    reasoning_block = "<think>\n" + "\n".join(reasoning_lines).strip() + "\n</think>"
    score_block = build_score_only_assistant_report(metric_reports, overall_avg)
    return reasoning_block + "\n\n" + score_block


def build_review_with_reasoning_assistant_report(
    metric_reports: list[dict], overall_avg: float, raw_reviews: dict
) -> str:
    """Per-model raw reasoning as <thinking> CoT, followed by the full
    score + synthesized-review report (combines reasoning and review modes)."""
    reasoning_lines = []
    for report in metric_reports:
        metric_key = report["metric_key"]
        metric_label = report["metric"]
        reasoning_lines.append(f"### {metric_label}")
        model_reasons = raw_reviews.get(metric_key, {})
        valid = [
            (model, data)
            for model, data in model_reasons.items()
            if model not in EXCLUDED_MODELS
            and isinstance(data, dict)
            and isinstance(data.get("reason"), str)
            and data["reason"].strip()
        ]
        for i, (_, data) in enumerate(valid):
            reasoning_lines.append(f"Reviewer {i + 1}:")
            reasoning_lines.append(data["reason"].strip())
            reasoning_lines.append("")
        reasoning_lines.append("")
    reasoning_block = "<think>\n" + "\n".join(reasoning_lines).strip() + "\n</think>"
    review_block = build_review_assistant_report(metric_reports, overall_avg)
    return reasoning_block + "\n\n" + review_block


def story_sha1(text: str) -> str:
    """Stable content hash used for deduplication and split assignment."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def split_name(content_hash: str, train_ratio: float) -> str:
    """Assign split deterministically from the story's content hash.
    Using content hash (not line index) guarantees the same story always
    lands in the same split even if the source file is reordered."""
    bucket = int(content_hash[:8], 16) / 0xFFFFFFFF
    return "train" if bucket < train_ratio else "test"


def word_count_no_punct(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def build_raw_reviews(row: dict, metrics: list[str]) -> dict:
    """Collect all model scores/reasons for every metric as metadata."""
    raw: dict = {}
    for metric_name in metrics:
        metric_obj = row.get(metric_name)
        if not isinstance(metric_obj, dict):
            continue
        raw[metric_name] = {
            model: {"score": data["score"], "reason": data["reason"]}
            for model, data in metric_obj.items()
            if model != "overall" and not model.startswith("_")
        }
    return raw


def build_unified_dataset(
    input_path: Path,
    train_ratio: float,
) -> tuple[list[dict], list[dict]]:
    """
    Single-pass builder.  Each eligible story produces ONE row.
    Reviews and scores are read from metric_obj["overall"] written by
    summarize_reviews.py.

    Returns (train_rows, test_rows).
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info("Loading file: %s", input_path)

    sys_score_only = "score only"
    sys_score_with_reasoning = "score only with reasoning"
    sys_review_with_reasoning = "review with reasoning"

    total_records = 0
    total_samples = 0
    skipped_no_story = 0
    skipped_duplicate = 0
    skipped_incomplete_metrics = 0
    seen_hashes: set[str] = set()
    max_raw_row_chars = 0
    max_raw_row_line = -1
    max_story_chars = 0
    max_story_words = 0
    max_story_line = -1
    total_review_chars = 0
    msg_chars_score_only: list[int] = []
    msg_chars_score_with_reasoning: list[int] = []
    msg_chars_score_with_reviews: list[int] = []
    msg_chars_review_with_reasoning: list[int] = []
    train_rows: list[dict] = []
    test_rows: list[dict] = []

    with open(input_path, "r", encoding="utf-8") as fin:
        pbar = tqdm(enumerate(fin, start=1), desc="Processing records", unit="rec")
        for line_idx, raw_line in pbar:
            if not raw_line.strip():
                continue

            total_records += 1
            raw_len = len(raw_line.rstrip("\n"))
            if raw_len > max_raw_row_chars:
                max_raw_row_chars = raw_len
                max_raw_row_line = line_idx

            row = json.loads(raw_line)
            story = row.get("regenerated_story")
            if not isinstance(story, str) or not story.strip():
                skipped_no_story += 1
                continue

            sha1 = story_sha1(story)
            if sha1 in seen_hashes:
                skipped_duplicate += 1
                continue
            seen_hashes.add(sha1)

            story_chars = len(story)
            story_words = word_count_no_punct(story)
            if story_chars > max_story_chars:
                max_story_chars = story_chars
                max_story_words = story_words
                max_story_line = line_idx

            # ---- build metric_reports from synthesized overall key ----
            metric_reports: list[dict] = []
            for metric_name in TTCW_METRICS:
                metric_obj = row.get(metric_name)
                overall = metric_obj.get("overall") if isinstance(metric_obj, dict) else None
                if (
                    not isinstance(overall, dict)
                    or not isinstance(overall.get("avg_score"), (int, float))
                    or not isinstance(overall.get("review"), str)
                    or not overall["review"].strip()
                ):
                    break
                metric_reports.append(
                    {
                        "metric_key": metric_name,
                        "metric": METRIC_LABELS.get(metric_name, metric_name),
                        "avg_score": float(overall["avg_score"]),
                        "synthesized_review": overall["review"].strip(),
                    }
                )

            if len(metric_reports) < len(TTCW_METRICS):
                skipped_incomplete_metrics += 1
                continue

            overall_avg = sum(r["avg_score"] for r in metric_reports) / len(metric_reports)
            total_review_chars += sum(len(r["synthesized_review"]) for r in metric_reports)

            # Build raw_reviews first — needed by score_with_reasoning mode.
            raw_reviews = build_raw_reviews(row, TTCW_METRICS)

            # ---- build assistant texts for all modes ----
            score_only_text = build_score_only_assistant_report(
                metric_reports, overall_avg
            )
            score_with_reasoning_text = build_score_with_reasoning_assistant_report(
                metric_reports, overall_avg, raw_reviews
            )
            review_text = build_review_assistant_report(metric_reports, overall_avg)
            review_with_reasoning_text = build_review_with_reasoning_assistant_report(
                metric_reports, overall_avg, raw_reviews
            )

            # ---- message columns ----
            msgs_score_only = make_messages(build_ttcw_user_prompt(story, sys_score_only), score_only_text)
            msgs_score_with_reasoning = make_messages(build_ttcw_user_prompt(story, sys_score_with_reasoning), score_with_reasoning_text)
            msgs_score_with_reviews = make_messages(build_ttcw_user_prompt(story), review_text)
            msgs_review_with_reasoning = make_messages(build_ttcw_user_prompt(story, sys_review_with_reasoning), review_with_reasoning_text)

            msg_chars_score_only.append(len(json.dumps(msgs_score_only)))
            msg_chars_score_with_reasoning.append(len(json.dumps(msgs_score_with_reasoning)))
            msg_chars_score_with_reviews.append(len(json.dumps(msgs_score_with_reviews)))
            msg_chars_review_with_reasoning.append(len(json.dumps(msgs_review_with_reasoning)))

            # ---- common metadata ----
            metric_avg_scores = {
                r["metric_key"]: round(r["avg_score"], 4) for r in metric_reports
            }
            split = split_name(sha1, train_ratio)

            sample = {
                "prompt": row.get("prompt", ""),
                "story": row.get("story", ""),
                "regenerated_story": story,
                "word_count": row.get("word_count", 0),
                "needs_regeneration": row.get("needs_regeneration", False),
                "generated_model": row.get("generated_model", ""),
                "overall_score": row.get("overall_score", ""),
                **{m: row.get(m) for m in TTCW_METRICS},
                "messages_score_only": msgs_score_only,
                "messages_score_with_reasoning": msgs_score_with_reasoning,
                "messages_score_with_reviews": msgs_score_with_reviews,
                "messages_review_with_reasoning": msgs_review_with_reasoning,
                "source_line": line_idx,
                "overall_avg_score": round(overall_avg, 4),
                "num_metrics": len(metric_reports),
                "metrics_covered": [r["metric_key"] for r in metric_reports],
                "metric_avg_scores": metric_avg_scores,
                "raw_reviews": raw_reviews,
            }

            if split == "train":
                train_rows.append(sample)
            else:
                test_rows.append(sample)
            total_samples += 1
            pbar.set_postfix(train=len(train_rows), test=len(test_rows), skip=skipped_no_story + skipped_duplicate + skipped_incomplete_metrics)

    logger.info("Input records read:    %d", total_records)
    logger.info("SFT samples built:     %d", total_samples)
    logger.info("  train:               %d", len(train_rows))
    logger.info("  test:                %d", len(test_rows))
    logger.info("Skipped (no story):    %d", skipped_no_story)
    logger.info("Skipped (duplicate):   %d", skipped_duplicate)
    logger.info("Skipped (incomplete):  %d", skipped_incomplete_metrics)
    logger.info("Max raw row chars:     %d (line %d)", max_raw_row_chars, max_raw_row_line)
    logger.info("Max story chars/words: %d/%d (line %d)", max_story_chars, max_story_words, max_story_line)
    avg_review_chars = total_review_chars / (total_samples * len(TTCW_METRICS)) if total_samples else 0
    logger.info("Avg metric review len: %.1f chars (%d reviews across %d samples)", avg_review_chars, total_samples * len(TTCW_METRICS), total_samples)

    def p90(lengths: list[int]) -> int:
        if not lengths:
            return 0
        s = sorted(lengths)
        return s[int(len(s) * 0.9)]

    logger.info("P90 message chars (score_only):              %7d", p90(msg_chars_score_only))
    logger.info("P90 message chars (score_with_reasoning):    %7d", p90(msg_chars_score_with_reasoning))
    logger.info("P90 message chars (score_with_reviews):      %7d", p90(msg_chars_score_with_reviews))
    logger.info("P90 message chars (review_with_reasoning):   %7d", p90(msg_chars_review_with_reasoning))
    return train_rows, test_rows


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    input_path = Path(BUILD_CONFIG["input_path"])
    output_path = Path(BUILD_CONFIG["output_path"])
    overwrite = bool(BUILD_CONFIG["overwrite"])
    train_ratio = float(BUILD_CONFIG["train_ratio"])

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Set BUILD_CONFIG['overwrite']=True to replace it."
        )

    logger.info("=== Building unified dataset (score_only + score_with_reviews + combined) ===")
    train_rows, test_rows = build_unified_dataset(
        input_path=input_path,
        train_ratio=train_ratio,
    )

    logger.info("Converting %d train rows to Arrow format ...", len(train_rows))
    train_dataset = Dataset.from_generator(lambda: iter(train_rows))
    del train_rows
    logger.info("Train dataset ready. Converting %d test rows to Arrow format ...", len(test_rows))
    test_dataset = Dataset.from_generator(lambda: iter(test_rows))
    del test_rows
    dataset_dict = DatasetDict(
        {
            "train": train_dataset,
            "test": test_dataset,
        }
    )

    # Clean up the old dataset only after the new one is fully built in memory.
    if output_path.exists():
        logger.info("Removing old dataset at %s ...", output_path)
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))

    logger.info("=== Dataset saved ===")
    logger.info("Train samples:      %d", len(train_dataset))
    logger.info("Test samples:       %d", len(test_dataset))
    logger.info("HF dataset saved to: %s", output_path)


if __name__ == "__main__":
    main()
