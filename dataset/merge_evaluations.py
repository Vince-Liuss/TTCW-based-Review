"""
merge_evaluations.py
--------------------
Standalone script to merge per-model split files produced by
Story_evaluator_api.py into a single combined JSONL file.

Usage:
    python merge_evaluations.py

Keep EVALUATION_MODELS and OUTPUT_PATH in sync with Story_evaluator_api.py.
"""

import os
import re
import json
import time
import hashlib
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm as tqdm_sync

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — keep in sync with Story_evaluator_api.py
# ---------------------------------------------------------------------------
EVALUATION_MODELS: list[str] = [
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
    "openai/gpt-oss-120b",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
]
OUTPUT_PATH = Path("../data/story_evaluation_dataset_evaluated.jsonl")
NUM_WORKERS = 16

# Field ordering for the output file.
BASE_FIELDS: list[str] = [
    "prompt",
    "story",
    "regenerated_story",
    "word_count",
    "needs_regeneration",
    "generated_model",
]
METRICS: list[str] = [
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_records(path: Path, quiet: bool = False) -> list[dict]:
    if not path.exists():
        return []
    if not quiet:
        logger.info(f"Loading {path}...")
    records: list[dict] = []
    start = time.time()
    pbar = tqdm_sync(desc=f"load:{path.name}", unit="row", leave=False)
    try:
        with open(path, "rb", buffering=4 * 1024 * 1024) as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(json.loads(line))
                pbar.update(1)
    finally:
        pbar.close()
    if not quiet:
        logger.info(
            f"Loaded {len(records)} rows from {path.name} in {time.time() - start:.1f}s"
        )
    return records


def _safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.split("/")[-1])


def model_split_path(output_path: Path, model: str) -> Path:
    safe = _safe_model_name(model)
    return output_path.with_name(f"{output_path.stem}.{safe}{output_path.suffix}")


def _record_key(record: dict) -> str | None:
    if not isinstance(record, dict):
        return None
    story = record.get("regenerated_story") or record.get("story")
    if isinstance(story, str) and story.strip():
        return "story_sha1:" + hashlib.sha1(story.encode()).hexdigest()
    return None


def _serialize_chunk_jsonl(chunk: list[dict]) -> list[str]:
    return [json.dumps(record, ensure_ascii=False) + "\n" for record in chunk]


def _write_records_atomic_parallel(records: list[dict], path: Path, workers: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(
        f"{path.name}.{os.getpid()}.{int(time.time() * 1000)}.tmp"
    )

    if workers <= 1 or len(records) < 10000:
        with open(temp_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temp_path, path)
        return

    chunk_size = max(1, len(records) // workers)
    chunks = [records[i : i + chunk_size] for i in range(0, len(records), chunk_size)]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        chunk_lines = list(pool.map(_serialize_chunk_jsonl, chunks))

    with open(temp_path, "w", encoding="utf-8") as f:
        for lines in chunk_lines:
            f.writelines(lines)

    os.replace(temp_path, path)


# ---------------------------------------------------------------------------
# Core merge
# ---------------------------------------------------------------------------
def merge_split_files(
    models: list[str],
    output_path: Path,
    num_workers: int = 1,
) -> None:
    """
    Merge all per-model split files into a single combined JSONL.
    Metrics are discovered automatically from the split file data.
    """
    logger.info("🔀 Merging model split files...")

    per_model: dict[str, dict[str, dict]] = {}
    row_count: int | None = None
    ordered_keys: list[str] = []

    for model in models:
        split_path = model_split_path(output_path, model)
        if not split_path.exists():
            logger.warning(f"  ⚠️  Split file missing, skipping: {split_path.name}")
            continue

        records = load_records(split_path, quiet=True)
        if row_count is None:
            row_count = len(records)
            for row in records:
                k = _record_key(row)
                if k:
                    ordered_keys.append(k)
        elif len(records) != row_count:
            logger.warning(
                f"  ⚠️  Row count mismatch ({len(records)} vs {row_count}) "
                f"for {split_path.name} — skipping."
            )
            continue

        per_model[model] = {_record_key(r): r for r in records if _record_key(r)}
        logger.info(f"  Loaded {len(records)} rows from {split_path.name}")

    if not per_model:
        logger.error("No split files found; nothing to merge.")
        return

    logger.info(f"Using metrics ({len(METRICS)}): {METRICS}")

    first_model = next(iter(per_model))
    merged: list[dict] = []
    for key in ordered_keys:
        base_row = per_model[first_model].get(key)
        if base_row is None:
            continue

        # Build output with guaranteed field order: base fields → metric fields.
        merged_row: dict = {}
        for field in BASE_FIELDS:
            merged_row[field] = base_row.get(field, "")

        for metric in METRICS:
            combined: dict = {}
            for model, index in per_model.items():
                row = index.get(key)
                if row is None:
                    continue
                entry = (row.get(metric) or {}).get(model)
                if isinstance(entry, dict) and entry.get("score") is not None:
                    combined[model] = entry
            merged_row[metric] = combined  # always present; empty dict if unevaluated

        merged.append(merged_row)

    _write_records_atomic_parallel(merged, output_path, workers=num_workers)
    logger.info(
        f"✅ Merged {len(merged)} rows × {len(per_model)} models → {output_path}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    merge_split_files(EVALUATION_MODELS, OUTPUT_PATH, NUM_WORKERS)


if __name__ == "__main__":
    main()
