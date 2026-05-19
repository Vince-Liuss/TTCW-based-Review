import argparse
import asyncio
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT = (
    SCRIPT_DIR / "../data/story_evaluation_dataset_synthesized.jsonl"
).resolve()
DEFAULT_OUTPUT = (SCRIPT_DIR / "../logs/review_comment_quality_sample.jsonl").resolve()

REVIEW_FIELDS = [
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

# 3 NLG evaluation dimensions (based on SummEval/UniEval framework).
# Each becomes a separate API call per annotation.
CRITERIA = [
    (
        "faithfulness",
        "Does the review only make claims that are consistent with the story's actual content, without introducing details, events, or characterizations not present in the story?",
    ),
    (
        "coherence",
        "Is the review logically organized and internally consistent, with no contradictory statements?",
    ),
    (
        "relevance",
        "Does the review focus on specific aspects of this story rather than making observations that could apply to almost any story?",
    ),
]

SYSTEM_PROMPT = """\
You are evaluating the quality of a review written about a story using NLG evaluation criteria.

Compare the review against the story and answer ONLY whether the review satisfies the given criterion.

Do NOT rewrite the review.
Do NOT judge whether the story is good or bad.
Do NOT reward a review just because it sounds plausible.

Reply with exactly two lines:
Line 1: yes or no
Line 2: one short reason sentence
"""

USER_TEMPLATE = """\
Story excerpt:
{story}

Review:
{text}

Validation task:
Compare the review/comment to the story. Focus on whether the review's claims are actually supported by this story.

Criterion:
{criterion_question}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample dataset rows and ask binary story-match questions per annotation."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of random dataset rows to sample.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target",
        choices=["overall", "comments", "both"],
        default="overall",
        help="overall checks field['overall']['review']; comments checks per-model reason fields.",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model keys to audit for comments. Empty means all.",
    )
    parser.add_argument(
        "--audit-model",
        default="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
        help="Model served by your OpenAI-compatible local API.",
    )
    parser.add_argument("--api-base-url", default="http://0.0.0.0:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--concurrency", type=int, default=8)
    return parser.parse_args()


def story_key(row: dict) -> str:
    story = row.get("regenerated_story") or row.get("story") or ""
    if not isinstance(story, str):
        story = ""
    return hashlib.sha1(story.encode("utf-8")).hexdigest()[:12]


def truncate_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[TRUNCATED]"


def reservoir_sample_jsonl(path: Path, sample_size: int, seed: int) -> tuple[int, list[tuple[int, dict]]]:
    rng = random.Random(seed)
    sample: list[tuple[int, str]] = []  # store raw strings, defer JSON parsing
    total = 0

    with open(path, "r", encoding="utf-8", buffering=1 << 20) as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total += 1
            if len(sample) < sample_size:
                sample.append((line_no, line))
                continue
            replace_at = rng.randint(0, total - 1)
            if replace_at < sample_size:
                sample[replace_at] = (line_no, line)

    sample.sort(key=lambda x: x[0])
    # Parse JSON only for the selected sample_size rows
    return total, [(line_no, json.loads(line)) for line_no, line in sample]


def iter_annotation_items_from_rows(
    sampled_rows: list[tuple[int, dict]],
    target: str,
    requested_models: set[str],
):
    for line_no, row in sampled_rows:
        story = row.get("regenerated_story") or row.get("story") or ""
        if not isinstance(story, str) or not story.strip():
            story = "[NO STORY TEXT AVAILABLE]"

        for field in REVIEW_FIELDS:
            field_obj = row.get(field)
            if not isinstance(field_obj, dict):
                continue

            if target in {"overall", "both"}:
                overall = field_obj.get("overall")
                review = overall.get("review") if isinstance(overall, dict) else None
                if isinstance(review, str) and review.strip():
                    yield {
                        "line": line_no,
                        "story_key": story_key(row),
                        "field": field,
                        "annotation_type": "overall_review",
                        "source_model": "overall",
                        "story": story,
                        "text": review.strip(),
                    }

            if target in {"comments", "both"}:
                for model_name, value in field_obj.items():
                    if model_name == "overall":
                        continue
                    if requested_models and model_name not in requested_models:
                        continue
                    if not isinstance(value, dict):
                        continue
                    reason = value.get("reason")
                    if isinstance(reason, str) and reason.strip():
                        yield {
                            "line": line_no,
                            "story_key": story_key(row),
                            "field": field,
                            "annotation_type": "model_comment",
                            "source_model": model_name,
                            "story": story,
                            "text": reason.strip(),
                        }


def extract_answer(text: str) -> dict:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    text = re.sub(r"</think>", "", text).strip()  # non-think mode emits </think> with no opening tag
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    answer_raw = ""
    reason_raw = ""
    for i, line in enumerate(lines):
        clean = re.sub(r"^answer\s*:\s*", "", line, flags=re.IGNORECASE).strip()
        if re.search(r"\byes\b|\bno\b", clean, flags=re.IGNORECASE):
            answer_raw = clean
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            reason_raw = re.sub(r"^reason\s*:\s*", "", next_line, flags=re.IGNORECASE)
            break
    if not answer_raw:
        answer_raw = lines[0] if lines else ""
        reason_raw = lines[1] if len(lines) > 1 else ""
    return {"answer": answer_raw, "reason": reason_raw}


def normalize_answer(value: object) -> str:
    if not isinstance(value, str):
        return "parse_error"
    value = value.strip().lower()
    if re.search(r"\byes\b", value):
        return "yes"
    if re.search(r"\bno\b", value):
        return "no"
    return "parse_error"


async def audit_one_criterion(
    client: AsyncOpenAI,
    args: argparse.Namespace,
    item: dict,
    criterion_key: str,
    criterion_question: str,
) -> dict:
    user = USER_TEMPLATE.format(
        story=item["story"],
        text=item["text"],
        criterion_question=criterion_question,
    )
    response = await client.chat.completions.create(
        model=args.audit_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=1.0,
        top_p=0.95,
        max_tokens=512,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        stream=False,
    )
    raw_output = response.choices[0].message.content or ""
    audit = extract_answer(raw_output)
    answer = normalize_answer(audit.get("answer"))
    return {
        **{k: v for k, v in item.items() if k not in {"story", "text"}},
        "text_preview": truncate_text(item["text"], 800),
        "criterion": criterion_key,
        "criterion_question": criterion_question,
        "raw_output": raw_output,
        "answer": answer,
        "pass": answer == "yes",
        "reason": audit.get("reason", ""),
    }


async def run_audit(args: argparse.Namespace, tasks: list[tuple[dict, str, str]]) -> list[dict]:
    client = AsyncOpenAI(base_url=args.api_base_url, api_key=args.api_key, timeout=600.0)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def bounded(item: dict, criterion_key: str, criterion_question: str) -> dict:
        async with semaphore:
            try:
                return await audit_one_criterion(
                    client, args, item, criterion_key, criterion_question
                )
            except Exception as exc:
                return {
                    **{k: v for k, v in item.items() if k not in {"story", "text"}},
                    "text_preview": truncate_text(item["text"], 800),
                    "criterion": criterion_key,
                    "criterion_question": criterion_question,
                    "raw_output": "",
                    "answer": "api_error",
                    "pass": False,
                    "reason": str(exc)[:300],
                }

    coros = [bounded(item, ck, cq) for item, ck, cq in tasks]
    return await tqdm.gather(*coros, desc="Asking criterion-level judge questions")


def pass_rate(passed: int, total: int) -> float:
    return round(passed / total, 4) if total else 0.0


def build_summary(
    total_rows: int, sampled_rows: int, results: list[dict], args: argparse.Namespace
) -> dict:
    # Per-criterion pass rates
    criterion_totals: Counter = Counter()
    criterion_passes: Counter = Counter()
    for r in results:
        ck = r.get("criterion", "")
        criterion_totals[ck] += 1
        if r.get("pass"):
            criterion_passes[ck] += 1

    # Per-field pass rates: an annotation passes a field only if ALL criteria pass.
    # Group results by (line, field, source_model).
    item_criteria: dict = defaultdict(dict)
    for r in results:
        field = r.get("field") or r.get("metric", "")
        key = (r["line"], field, r.get("source_model", ""))
        item_criteria[key][r.get("criterion", "")] = r.get("pass", False)

    criterion_keys = [ck for ck, _ in CRITERIA]
    field_item_counts: Counter = Counter()
    field_item_passes: Counter = Counter()
    for (line, field, _), criteria_results in item_criteria.items():
        field_item_counts[field] += 1
        if all(criteria_results.get(ck, False) for ck in criterion_keys):
            field_item_passes[field] += 1

    total_items = len(item_criteria)
    items_all_pass = sum(
        1 for cr in item_criteria.values()
        if all(cr.get(ck, False) for ck in criterion_keys)
    )

    return {
        "input": str(args.input),
        "target": args.target,
        "seed": args.seed,
        "rows_in_file": total_rows,
        "sampled_rows": sampled_rows,
        "total_annotations": total_items,
        "total_criterion_questions": len(results),
        "annotations_all_criteria_pass": items_all_pass,
        "annotation_pass_rate": pass_rate(items_all_pass, total_items),
        "pass_rate_by_criterion": {
            ck: {
                "question": cq,
                "passed": criterion_passes[ck],
                "total": criterion_totals[ck],
                "pass_rate": pass_rate(criterion_passes[ck], criterion_totals[ck]),
            }
            for ck, cq in CRITERIA
        },
        "pass_rate_by_field": {
            field: {
                "annotations_all_pass": field_item_passes[field],
                "total_annotations": field_item_counts[field],
                "pass_rate": pass_rate(field_item_passes[field], field_item_counts[field]),
            }
            for field in REVIEW_FIELDS
            if field_item_counts[field]
        },
    }


def format_summary(summary: dict) -> str:
    lines = []
    lines.append(f"Input: {summary['input']}")
    lines.append(f"Rows in file: {summary['rows_in_file']}")
    lines.append(f"Sampled rows: {summary['sampled_rows']} (seed={summary['seed']})")
    lines.append(f"Total annotations audited: {summary['total_annotations']}")
    lines.append(f"Total criterion questions asked: {summary['total_criterion_questions']}")
    lines.append(
        f"Annotations passing all {len(CRITERIA)} criteria: "
        f"{summary['annotations_all_criteria_pass']}/{summary['total_annotations']} "
        f"({summary['annotation_pass_rate']:.2%})"
    )
    lines.append("")
    lines.append("Pass rate by criterion:")
    for ck, stats in summary["pass_rate_by_criterion"].items():
        lines.append(
            f"  {ck}: {stats['pass_rate']:.2%} ({stats['passed']}/{stats['total']})"
            f"  — {stats['question']}"
        )
    lines.append("")
    lines.append(f"Pass rate by dataset field (all {len(CRITERIA)} criteria pass):")
    for field, stats in summary["pass_rate_by_field"].items():
        lines.append(
            f"  {field}: {stats['pass_rate']:.2%} "
            f"({stats['annotations_all_pass']}/{stats['total_annotations']})"
        )
    return "\n".join(lines)


def load_existing_results(path: Path) -> tuple[list[dict], set[tuple]]:
    if not path.exists():
        return [], set()
    existing = []
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            existing.append(r)
            field = r.get("field") or r.get("metric", "")
            done.add((r["line"], field, r.get("source_model", ""), r.get("criterion", "")))
    return existing, done


async def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be greater than 0")

    requested_models = {m.strip() for m in args.models.split(",") if m.strip()}
    total_rows, sampled_rows = reservoir_sample_jsonl(
        args.input, args.sample_size, args.seed
    )
    items = list(iter_annotation_items_from_rows(
        sampled_rows,
        target=args.target,
        requested_models=requested_models,
    ))
    if not items:
        raise ValueError("No review/comment annotations found to audit.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing_results, done = load_existing_results(args.output)

    pending = [
        (item, ck, cq)
        for item in items
        for ck, cq in CRITERIA
        if (item["line"], item["field"], item.get("source_model", ""), ck) not in done
    ]
    total_tasks = len(items) * len(CRITERIA)
    print(f"Annotations: {len(items)}  Criteria: {len(CRITERIA)}  Total tasks: {total_tasks}")
    print(f"Already done: {len(existing_results)}  Pending: {len(pending)}")

    new_results = await run_audit(args, pending) if pending else []
    with open(args.output, "a", encoding="utf-8") as f:
        for result in new_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    results = existing_results + new_results

    summary = build_summary(total_rows, len(sampled_rows), results, args)
    summary_path = args.output.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    table = format_summary(summary)
    log_path = args.output.with_suffix(".log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")

    print(table)
    print(f"\nDetailed audit written to: {args.output}")
    print(f"Summary written to: {summary_path}")
    print(f"Table log written to: {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
