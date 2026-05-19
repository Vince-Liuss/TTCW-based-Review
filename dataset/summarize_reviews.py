"""
summarize_reviews.py
====================
Preprocessing step: call an LLM to synthesize all per-model reviews for each
TTCW metric into a single, coherent consensus review.

The synthesized text is stored back into the evaluated JSONL under a special
"_synthesis" key inside each metric dict:

    "Fluency1": {
        "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5": {"score": 8, "reason": "..."},
        "openai/gpt-oss-120b":                       {"score": 7, "reason": "..."},
        "Qwen/Qwen3-Next-80B-A3B-Instruct":          {"score": 6, "reason": "..."},
        "overall": {"review": "...", "avg_score": 7.0}
    }

Run this once before dataset_build.py when using review_aggregation="summarized".
The script is resumable: rows/metrics that already have a valid "_synthesis" entry
are skipped.

Usage:
    python summarize_reviews.py
"""

import os

os.environ["OMP_NUM_THREADS"] = "16"

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm as tqdm_sync
from tqdm.asyncio import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

SYNTHESIS_CONFIG = {
    "input_path": (
        SCRIPT_DIR / "../data/story_evaluation_dataset_evaluated.jsonl"
    ).resolve(),
    "output_path": (
        SCRIPT_DIR / "../data/story_evaluation_dataset_synthesized.jsonl"
    ).resolve(),
    "synthesis_model": "zai-org/GLM-4.5-Air",
    "api_base_url": "http://0.0.0.0:8000/v1",
    "api_key": "EMPTY",
    "CONCURRENCY_LIMIT": 600,
    "MAX_RETRIES": 10,
    "DEBUG_MODE": False,
}

# Models whose scores/reviews are excluded from synthesis.
EXCLUDED_MODELS: set[str] = {
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
}

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


SYNTHESIS_SYSTEM_PROMPT = """\
You are a Meta-Reviewer — an expert who synthesizes multiple independent reviewers' assessments of a piece of creative writing into a single authoritative meta-review.

You will be given several numbered reviewer assessments. Treat them as the ONLY source of truth; do not invent claims that are not supported by at least one reviewer.

Your task (do internally, do not print steps):
1) For each reviewer, extract concrete STRENGTHS and WEAKNESSES (keep the reviewer’s meaning).
2) Merge across reviewers: prioritize points mentioned by multiple reviewers; include single-reviewer points only if they are important.
3) Note any meaningful disagreements (explicit contradictions or clear divergence in ratings/claims).

Write ONE unified meta-review paragraph(s) that:
- Covers the main shared strengths.
- Covers the main shared weaknesses / revision priorities.
- Briefly acknowledges key disagreements (without attributing to individuals).
- Uses a direct, professional voice.

Length policy:
- Do NOT force a fixed sentence count.
- Be as long as needed to cover all high-salience points, but avoid repetition.
- Prefer compact sentences; do not add filler.

Formatting (STRICT):
- Output ONLY the meta-review text.
- No preamble, no headings, no labels, no bullets, no tables, no numbering.
- No attribution like “Reviewer 1 said…”.

If reviewers provide scores, you may mention overall consensus direction (e.g., “overall strong but inconsistent in X”), but do not compute or output a numeric score unless explicitly asked.
"""

SYNTHESIS_USER_TEMPLATE = """\
Individual reviewer assessments:
{reviews}

Synthesize the above into one overall review by extracting each reviewer's key \
strengths and weaknesses, then combining them into a coherent consensus review."""


# ---------------------------------------------------------------------------
# Data helpers (shared with Story_evaluator_api.py pattern)
# ---------------------------------------------------------------------------


def _record_key(record: dict) -> str | None:
    if not isinstance(record, dict):
        return None
    # Prefer regenerated_story; fall back to original story for rows not yet regenerated.
    story = record.get("regenerated_story") or record.get("story")
    if isinstance(story, str) and story.strip():
        digest = hashlib.sha1(story.encode("utf-8")).hexdigest()
        return f"story_sha1:{digest}"
    return None


def load_records(path: Path, max_rows: int | None = None) -> list[dict]:
    if not path.exists():
        return []
    limit_msg = f" (max {max_rows} rows)" if max_rows else ""
    logger.info(f"Loading {path.name}{limit_msg}...")
    records: list[dict] = []
    pbar = tqdm_sync(desc=f"load:{path.name}", unit="row", leave=False)
    with open(path, "rb", buffering=4 * 1024 * 1024) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                pbar.update(1)
                if max_rows and len(records) >= max_rows:
                    break
    pbar.close()
    logger.info(f"Loaded {len(records)} rows.")
    return records


def write_records(records: list[dict], path: Path, workers: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{int(time.time()*1000)}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Synthesis log (append-only, one line per completed synthesis)
# Format: {"k": <record_key>, "m": <metric>, "review": "...", "avg_score": X}
# ---------------------------------------------------------------------------


def _synthesis_log_path(output_path: Path) -> Path:
    return output_path.with_suffix(".synthesis_log.jsonl")


def _apply_synthesis_log(records: list[dict], log_path: Path) -> int:
    """Replay completed syntheses from log into in-memory records. Returns count applied."""
    if not log_path.exists():
        return 0
    key_to_idx = {_record_key(r): i for i, r in enumerate(records) if _record_key(r)}
    applied = 0
    corrupt = 0
    with open(log_path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                corrupt += 1
                logger.warning(f"Log replay: corrupt JSON at line {lineno} — skipping.")
                continue
            k = entry.get("k")
            metric = entry.get("m")
            review = entry.get("review")
            avg_score = entry.get("avg_score")
            # All fields must be present and non-empty.
            if not k or not metric or not isinstance(review, str) or not review.strip():
                corrupt += 1
                logger.warning(
                    f"Log replay: missing/invalid fields at line {lineno} — skipping."
                )
                continue
            if avg_score is None or not isinstance(avg_score, (int, float)):
                corrupt += 1
                logger.warning(
                    f"Log replay: invalid avg_score at line {lineno} — skipping."
                )
                continue
            idx = key_to_idx.get(k)
            if idx is None:
                continue
            _write_synthesis(records[idx], metric, review, float(avg_score))
            applied += 1
    if corrupt:
        logger.warning(f"Log replay: {corrupt} corrupt line(s) skipped.")
    return applied


def _write_synthesis(record: dict, metric: str, review: str, avg_score: float) -> None:
    metric_obj = record.get(metric)
    if not isinstance(metric_obj, dict):
        metric_obj = {}
        record[metric] = metric_obj
    metric_obj["overall"] = {"review": review, "avg_score": avg_score}


def _has_valid_scores(record: dict, metrics: list[str]) -> bool:
    """Return True iff every metric has a valid score/reason from every non-excluded
    model that is present in the record.  The expected count is derived dynamically
    from the model keys found in the metric dict, so adding or removing models from
    EXCLUDED_MODELS requires no further edits here."""
    if not record.get("regenerated_story"):
        return False
    for metric in metrics:
        metric_obj = record.get(metric)
        if not isinstance(metric_obj, dict):
            return False
        # All non-excluded model keys present for this metric.
        all_non_excluded = [
            k
            for k in metric_obj
            if k != "overall" and not k.startswith("_") and k not in EXCLUDED_MODELS
        ]
        # Subset of those with a valid numeric score and non-empty reason.
        valid = [
            k
            for k in all_non_excluded
            if isinstance(metric_obj[k], dict)
            and isinstance(metric_obj[k].get("score"), int)
            and 0 <= metric_obj[k]["score"] <= 10
            and isinstance(metric_obj[k].get("reason"), str)
            and metric_obj[k]["reason"].strip()
        ]
        # Every non-excluded model must have a valid entry (and there must be at least one).
        if not all_non_excluded or len(valid) != len(all_non_excluded):
            return False
    return True


def get_pending(records: list[dict], metrics: list[str]) -> list[tuple[int, str]]:
    """Return (row_idx, metric_name) pairs that still need synthesis.

    Assumes records have already been filtered by _has_valid_scores() at load time.
    """
    pending = []
    for idx, rec in enumerate(records):
        for metric in metrics:
            metric_obj = rec.get(metric)
            synth = metric_obj.get("overall")  # type: ignore[union-attr]
            if (
                isinstance(synth, dict)
                and isinstance(synth.get("review"), str)
                and synth["review"].strip()
            ):
                continue  # already synthesized
            pending.append((idx, metric))
    return pending


# ---------------------------------------------------------------------------
# LLM synthesis
# ---------------------------------------------------------------------------


def _build_synthesis_prompt(record: dict, metric: str) -> str | None:
    metric_obj = record.get(metric)
    if not isinstance(metric_obj, dict):
        return None
    model_reviews = [
        (k, v)
        for k, v in metric_obj.items()
        if k != "overall"
        and not k.startswith("_")
        and k not in EXCLUDED_MODELS
        and isinstance(v, dict)
        and isinstance(v.get("score"), int)
        and 0 <= v["score"] <= 10
        and isinstance(v.get("reason"), str)
        and v["reason"].strip()
    ]
    if len(model_reviews) < 2:
        return None

    review_lines = "\n\n".join(
        f"Reviewer {i + 1}:\n{data['reason'].strip()}"
        for i, (_, data) in enumerate(model_reviews)
    )
    return SYNTHESIS_USER_TEMPLATE.format(reviews=review_lines)


def _avg_score_for_metric(record: dict, metric: str) -> float:
    metric_obj = record.get(metric, {})
    scores = [
        v["score"]
        for k, v in metric_obj.items()
        if k != "overall"
        and not k.startswith("_")
        and k not in EXCLUDED_MODELS
        and isinstance(v, dict)
        and isinstance(v.get("score"), int)
        and 0 <= v["score"] <= 10
    ]
    return sum(scores) / len(scores) if scores else 0.0


def parse_synthesis_response(text: str) -> str:
    """Strip think-tags and leading/trailing whitespace."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


# ---------------------------------------------------------------------------
# Async engine — semaphore-based, metric-by-metric
# ---------------------------------------------------------------------------


@dataclass
class SynthesisEngine:
    records: list[dict]
    cfg: dict
    client: AsyncOpenAI
    log_path: Path
    debug_mode: bool
    output_path: Path

    async def call_llm(self, prompt: str) -> str | None:
        for attempt in range(self.cfg["MAX_RETRIES"] + 1):
            try:
                res = await self.client.chat.completions.create(
                    model=self.cfg["synthesis_model"],
                    messages=[
                        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=4096,
                    stream=False,
                )
                text = res.choices[0].message.content
                if text:
                    return parse_synthesis_response(text)
            except Exception as e:
                if attempt >= self.cfg["MAX_RETRIES"]:
                    logger.error(f"Synthesis API error after retries: {e}")
                    return None
                await asyncio.sleep(1.0 * (attempt + 1))
        return None

    async def run_metric(self, metric: str, pending_indices: list[int]) -> int:
        """
        Synthesize all pending rows for one metric.

        Architecture mirrors Story_evaluator_api.py:
          - N worker coroutines drain eval_queue, call the LLM, and put
            (row_idx, review, avg_score) into write_queue.  No file I/O.
          - A single writer coroutine drains write_queue, updates the in-memory
            record, and appends one log line.  Single writer = no races, no locking.

        Shutdown order (critical — prevents early-exit race):
          1. eval_queue.join()  — all items dequeued *and* task_done called
             (task_done is in the worker finally block, after result is enqueued)
          2. Cancel + await workers
          3. write_queue.join() — all results written
          4. Send None sentinel to stop the writer task
        """
        if not pending_indices:
            return 0

        pbar = tqdm(total=len(pending_indices), desc=f"synthesize:{metric}", unit="row")
        eval_queue: asyncio.Queue[int] = asyncio.Queue()
        write_queue: asyncio.Queue[tuple[int, str | None, float] | None] = (
            asyncio.Queue()
        )
        for idx in pending_indices:
            eval_queue.put_nowait(idx)

        completed = 0

        async def writer_loop() -> None:
            nonlocal completed
            log_f = (
                None if self.debug_mode else open(self.log_path, "a", encoding="utf-8")
            )
            try:
                while True:
                    item = await write_queue.get()
                    if item is None:  # shutdown sentinel
                        write_queue.task_done()
                        break
                    row_idx, review, avg_score = item
                    pbar.update(1)
                    if review:
                        _write_synthesis(
                            self.records[row_idx], metric, review, avg_score
                        )
                        completed += 1
                        if self.debug_mode:
                            key = _record_key(self.records[row_idx])
                            logger.info(
                                f"[DEBUG] key={key} metric={metric} "
                                f"avg_score={avg_score:.1f}\nreview: {review}"
                            )
                        else:
                            key = _record_key(self.records[row_idx])
                            if key and log_f is not None:
                                log_f.write(
                                    json.dumps(
                                        {
                                            "k": key,
                                            "m": metric,
                                            "review": review,
                                            "avg_score": avg_score,
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                                log_f.flush()
                    write_queue.task_done()
            finally:
                if log_f is not None:
                    log_f.close()

        async def worker_loop() -> None:
            while True:
                try:
                    row_idx = await eval_queue.get()
                except asyncio.CancelledError:
                    break
                try:
                    prompt = _build_synthesis_prompt(self.records[row_idx], metric)
                    review = await self.call_llm(prompt) if prompt is not None else None
                    avg_score = (
                        _avg_score_for_metric(self.records[row_idx], metric)
                        if review
                        else 0.0
                    )
                    await write_queue.put((row_idx, review, avg_score))
                finally:
                    eval_queue.task_done()

        writer_task = asyncio.create_task(writer_loop())
        workers = [
            asyncio.create_task(worker_loop())
            for _ in range(self.cfg["CONCURRENCY_LIMIT"])
        ]

        # Wait for all API calls to finish and their results to enter write_queue.
        await eval_queue.join()
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        # Drain write_queue fully, then send the sentinel to stop the writer.
        await write_queue.join()
        await write_queue.put(None)
        await writer_task

        pbar.close()
        logger.info(f"Metric '{metric}': {completed}/{len(pending_indices)} succeeded.")
        return completed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    cfg = SYNTHESIS_CONFIG
    input_path = Path(cfg["input_path"])
    output_path = Path(cfg["output_path"]) if cfg["output_path"] else input_path
    log_path = _synthesis_log_path(output_path)
    debug_mode = bool(cfg["DEBUG_MODE"])

    DEBUG_SAMPLE = 20  # rows to load in debug mode

    if not debug_mode and output_path.exists():
        records = load_records(output_path, max_rows=None)
        if records:
            logger.info(
                f"Resuming from checkpoint file: {output_path.name} ({len(records)} rows)"
            )
        else:
            logger.warning(
                f"Checkpoint file {output_path.name} is empty — falling back to input."
            )
            records = load_records(input_path, max_rows=None)
    else:
        records = load_records(
            input_path, max_rows=DEBUG_SAMPLE if debug_mode else None
        )

    if not records:
        logger.error(f"No records found at {input_path}")
        return

    # Drop rows that don't have all 3 valid model scores for every metric.
    # Such rows can never be synthesized and must not be written to the output.
    before = len(records)
    records = [r for r in records if _has_valid_scores(r, TTCW_METRICS)]
    dropped = before - len(records)
    if dropped:
        logger.info(
            f"Dropped {dropped} rows with incomplete/invalid scores "
            f"({len(records)} rows eligible)."
        )
        # Write back immediately so future resumes load the already-filtered set.
        if not debug_mode:
            write_records(records, output_path)
            logger.info(
                f"💾 Checkpoint updated: removed {dropped} ineligible rows → {len(records)} rows."
            )

    # Replay any partial log for the metric that was interrupted mid-run.
    if not debug_mode and log_path.exists():
        applied = _apply_synthesis_log(records, log_path)
        logger.info(f"Replayed {applied} syntheses from {log_path.name}")

    pending = get_pending(records, TTCW_METRICS)
    if debug_mode:
        first_metric = TTCW_METRICS[0]
        pending = [(i, m) for i, m in pending if m == first_metric]
        logger.info(
            f"DEBUG_MODE: limited to metric '{first_metric}' ({len(pending)} items)"
        )
    else:
        logger.info(f"Pending syntheses: {len(pending)}")

    if not pending:
        logger.info("Nothing to do. All metrics already synthesized.")
    else:
        client = AsyncOpenAI(
            base_url=cfg["api_base_url"],
            api_key=cfg["api_key"],
            timeout=600.0,
        )
        engine = SynthesisEngine(
            records=records,
            cfg=cfg,
            client=client,
            log_path=log_path,
            debug_mode=debug_mode,
            output_path=output_path,
        )

        # Build a per-metric index from the flat pending list.
        pending_by_metric: dict[str, list[int]] = defaultdict(list)
        for row_idx, metric in pending:
            pending_by_metric[metric].append(row_idx)

        # Process one metric at a time for stability; checkpoint after each one.
        for metric in TTCW_METRICS:
            indices = pending_by_metric.get(metric, [])
            if not indices:
                logger.info(f"Metric '{metric}': already complete, skipping.")
                continue

            await engine.run_metric(metric, indices)

            if not debug_mode:
                write_records(records, output_path)
                logger.info(f"💾 Checkpoint saved after metric: {metric}")
                if log_path.exists():
                    log_path.unlink()
                    logger.info(f"🗑️  Cleared log after checkpoint: {log_path.name}")

    if not debug_mode:
        # Final pass: keep only rows with valid scores (guaranteed by load-time
        def _row_is_complete(r: dict) -> bool:
            if not _has_valid_scores(r, TTCW_METRICS):
                return False
            for m in TTCW_METRICS:
                synth = r[m].get("overall")  # type: ignore[index]
                if (
                    not isinstance(synth, dict)
                    or not isinstance(synth.get("review"), str)
                    or not synth["review"].strip()
                ):
                    return False
            return True

        complete_records = [r for r in records if _row_is_complete(r)]
        skipped = len(records) - len(complete_records)
        if skipped:
            logger.info(
                f"⚠️  {skipped} rows still incomplete — excluded from final output."
            )
        write_records(complete_records, output_path)
        logger.info(f"✅ Written {len(complete_records)} rows to {output_path}")
    else:
        logger.info("DEBUG_MODE: no files written.")


if __name__ == "__main__":
    asyncio.run(main())
