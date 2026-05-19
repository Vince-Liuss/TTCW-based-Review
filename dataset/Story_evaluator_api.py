import os

os.environ["OMP_NUM_THREADS"] = "16"

import re
import json
import copy
import time
import hashlib
import logging
import asyncio
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from tqdm import tqdm as tqdm_sync
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI

from prompts import system_prompt as get_system, get_all_prompts, GPT_oss_system_prompt

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


CONFIG = {
    "evaluation_models": [
        "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
        "openai/gpt-oss-120b",
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
    ],
    "dataset_path": Path("../data/story_evaluation_dataset.jsonl"),
    "output_path": Path("../data/story_evaluation_dataset_evaluated.jsonl"),
    "api_base_url": "http://0.0.0.0:8000/v1",
    "api_key": "EMPTY",
    "CONCURRENCY_LIMIT": 256,
    "MODEL_CONCURRENCY": {
        "openai/gpt-oss-120b": 512,
    },
    "SAVE_INTERVAL": 5000,
    "DEBUG_MODE": False,
    "NUM_WORKERS": 16,
}

NEMOTRON_MODEL_NAME = "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5"
MAX_RETRIES = 10
SCORE_CONTEXT_LIMIT = -1
SCORE_PARSE_FAILURE = -2
SCORE_PROMPT_ERROR = -3


@dataclass(frozen=True)
class EvalConfig:
    evaluation_models: list[str]
    dataset_path: Path
    output_path: Path
    api_base_url: str
    api_key: str
    concurrency_limit: int
    model_concurrency: dict[str, int]
    save_interval: int
    debug_mode: bool
    num_workers: int

    def concurrency_for(self, model: str) -> int:
        """Return the concurrency limit for a specific model."""
        return self.model_concurrency.get(model, self.concurrency_limit)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "EvalConfig":
        return cls(
            evaluation_models=list(cfg["evaluation_models"]),
            dataset_path=Path(cfg["dataset_path"]),
            output_path=Path(cfg["output_path"]),
            api_base_url=str(cfg["api_base_url"]),
            api_key=str(cfg["api_key"]),
            concurrency_limit=int(cfg["CONCURRENCY_LIMIT"]),
            model_concurrency=dict(cfg.get("MODEL_CONCURRENCY", {})),
            save_interval=max(1, int(cfg["SAVE_INTERVAL"])),
            debug_mode=bool(cfg["DEBUG_MODE"]),
            num_workers=max(1, int(cfg.get("NUM_WORKERS", 1))),
        )


def _json_loads_fast(line: bytes) -> dict:
    return json.loads(line)


def load_records(path: Path, quiet: bool = False) -> list[dict]:
    if not path.exists():
        return []
    if not quiet:
        logger.info(f"Loading dataset from {path}...")

    records: list[dict] = []
    start = time.time()
    pbar = tqdm_sync(desc=f"load:{path.name}", unit="row", leave=False)
    try:
        with open(path, "rb", buffering=4 * 1024 * 1024) as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(_json_loads_fast(line))
                pbar.update(1)
    finally:
        pbar.close()

    if not quiet:
        logger.info(
            f"Loaded {len(records)} rows from {path.name} in {time.time() - start:.1f}s using json."
        )
    return records


def _safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.split("/")[-1])


def model_output_path(base_output_path: Path, model: str) -> Path:
    safe_model = _safe_model_name(model)
    return base_output_path.with_name(
        f"{base_output_path.stem}.{safe_model}{base_output_path.suffix}"
    )


def _record_key(record: dict) -> str | None:
    if not isinstance(record, dict):
        return None
    # Prefer regenerated_story; fall back to original story for rows not yet regenerated.
    story = record.get("regenerated_story") or record.get("story")
    if isinstance(story, str) and story.strip():
        digest = hashlib.sha1(story.encode("utf-8")).hexdigest()
        return f"story_sha1:{digest}"
    return None


def _results_log_path(model_path: Path) -> Path:
    """Append-only per-result log written during evaluation: …ModelName.log.jsonl"""
    return model_path.with_suffix(".log.jsonl")


def _apply_results_log(
    records: list[dict], log_path: Path, model: str
) -> dict[str, int]:
    """
    Replay a results log into the in-memory records list.

    Each log line is a compact JSON object:
      {"k": <record_key>, "m": <metric>, "s": <score>, "r": <reason>}

    Returns a stats dict with keys: applied, skipped_unknown_key, corrupt.
    """
    stats = {"applied": 0, "skipped_unknown_key": 0, "corrupt": 0}
    if not log_path.exists():
        return stats

    key_to_idx: dict[str, int] = {}
    for idx, record in enumerate(records):
        k = _record_key(record)
        if k:
            key_to_idx[k] = idx

    with open(log_path, "r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                stats["corrupt"] += 1
                logger.warning(
                    f"Log replay: corrupt JSON at line {lineno} in {log_path.name} — skipping."
                )
                continue
            key = entry.get("k")
            metric = entry.get("m")
            score = entry.get("s")
            reason = entry.get("r", "")
            if key is None or metric is None or score is None:
                stats["corrupt"] += 1
                logger.warning(
                    f"Log replay: missing field(s) at line {lineno} in {log_path.name} — skipping."
                )
                continue
            idx = key_to_idx.get(key)
            if idx is None:
                stats["skipped_unknown_key"] += 1
                continue
            _set_metric_result(records[idx], metric, model, score, reason)
            stats["applied"] += 1

    return stats


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


def _set_metric_result(record: dict, metric: str, model: str, score: int, reason: str):
    metric_obj = record.get(metric)
    if not isinstance(metric_obj, dict):
        metric_obj = {}
        record[metric] = metric_obj
    metric_obj[model] = {"score": score, "reason": reason}


def parse_response(raw_text: str) -> dict[str, Any]:
    if not raw_text:
        return {"score": None, "reason": None}

    clean_text = (
        raw_text.split("</think>")[-1].strip() if "</think>" in raw_text else raw_text
    )

    match = re.search(
        r"\*\*Reasons:\*\*\s*([\s\S]*?)\s*\*\*Score:\*\*\s*(\d{1,2})",
        clean_text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        match = re.search(
            r"Reasons:\s*([\s\S]*?)\s*Score:\s*(\d{1,2})",
            clean_text,
            re.IGNORECASE | re.DOTALL,
        )

    if match:
        try:
            return {"score": int(match.group(2)), "reason": match.group(1).strip()}
        except ValueError:
            pass

    return {"score": None, "reason": clean_text}


def build_model_records_from_source(
    base_records: list[dict],
    source_records: list[dict] | None,
    model: str,
    metrics: list[str],
) -> list[dict]:
    source_by_key: dict[str, dict] = {}
    if source_records:
        for row in source_records:
            key = _record_key(row)
            if key and key not in source_by_key and isinstance(row, dict):
                source_by_key[key] = row

    seeded_from_existing = 0
    model_records: list[dict] = []

    for row in base_records:
        new_row = dict(row)
        key = _record_key(row)
        source_row = source_by_key.get(key) if key else None
        if source_row is None:
            source_row = row
        else:
            seeded_from_existing += 1

        for metric in metrics:
            metric_data = source_row.get(metric)
            if isinstance(metric_data, dict) and isinstance(
                metric_data.get(model), dict
            ):
                new_row[metric] = {model: copy.deepcopy(metric_data[model])}
            else:
                new_row.pop(metric, None)

        model_records.append(new_row)

    if source_records:
        logger.info(
            f"Seeded {seeded_from_existing}/{len(base_records)} rows for model={model} from existing combined results."
        )

    return model_records


def ensure_model_split_records(
    cfg: EvalConfig,
    base_records: list[dict],
    seed_records: list[dict] | None,
    model: str,
    metrics: list[str],
    model_path: Path,
) -> list[dict]:
    """
    Return the working record list for this model's split file.

    Priority:
      1. Existing split file whose row count matches the canonical base → use as-is.
      2. Existing split file with wrong row count (base was extended) → rebuild,
         recovering already-scored rows from the stale split via seed_records.
      3. No split file → build fresh from base, optionally seeding from seed_records.

    seed_records should always be this model's own previous split file (loaded by the
    caller).  It must NEVER be the combined/merged output file.
    """
    if model_path.exists():
        logger.info(f"Step 1/3: Load existing model split file: {model_path}")
        records = load_records(model_path, quiet=True)
        if len(records) == len(base_records):
            logger.info(
                f"Step 1/3: Row count matches ({len(records)}), resuming from split file."
            )
            return records
        logger.warning(
            f"Model split row count {len(records)} != base {len(base_records)}. Rebuilding."
        )
        # Use the stale split itself as seed so completed scores are preserved.
        if seed_records is None:
            seed_records = records

    logger.info(f"Step 1/3: Building model split file: {model_path}")
    records = build_model_records_from_source(
        base_records, seed_records, model, metrics
    )
    if not cfg.debug_mode:
        _write_records_atomic_parallel(records, model_path, workers=cfg.num_workers)
    return records


def get_pending_indices_map(
    records: list[dict], metrics: list[str], model: str
) -> dict[str, list[int]]:
    pending = {metric: [] for metric in metrics}
    logger.info(f"Scanning dataset once for pending map: {model}")

    for idx, record in enumerate(records):
        if not record.get("regenerated_story"):
            continue
        for metric in metrics:
            metric_data = record.get(metric)
            if isinstance(metric_data, dict):
                model_data = metric_data.get(model)
                if (
                    isinstance(model_data, dict)
                    and isinstance(model_data.get("score"), int)
                    and model_data["score"] >= 0
                ):
                    continue
            pending[metric].append(idx)

    total_pending = sum(len(v) for v in pending.values())
    logger.info(
        f"Pending map ready for {model}. Total pending metric-items: {total_pending}"
    )
    return pending


class StoryEvaluator:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.file_save_lock = asyncio.Lock()
        self.client = AsyncOpenAI(
            base_url=cfg.api_base_url,
            api_key=cfg.api_key,
            timeout=600.0,
        )
        self.system_msg = get_system()
        self.gpt_oss_system_msg = GPT_oss_system_prompt()

    async def save_records_async(
        self, path: Path, records: list[dict], reason: str = ""
    ):
        if self.cfg.debug_mode:
            return
        async with self.file_save_lock:
            await asyncio.to_thread(
                _write_records_atomic_parallel,
                records,
                path,
                self.cfg.num_workers,
            )
        if reason:
            logger.info(f"💾 Saved {path.name} ({reason})")

    async def call_api_one_shot(
        self, model: str, user_prompt: str, model_system_msg: str
    ) -> str | None:
        is_gpt_oss = "gpt-oss" in model
        is_nemotron = model == NEMOTRON_MODEL_NAME
        if is_gpt_oss:
            messages = [
                {"role": "system", "content": self.gpt_oss_system_msg},
                {"role": "user", "content": user_prompt},
            ]
            extra_body = {"include_reasoning": False}
            max_tokens = 8192
            temperature = 0.0
        elif is_nemotron:
            messages = [
                {"role": "system", "content": model_system_msg},
                {"role": "user", "content": user_prompt},
            ]
            extra_body = {"top_p": 0.95}
            max_tokens = 4096
            temperature = 0.0
        else:
            messages = [
                {"role": "system", "content": model_system_msg},
                {"role": "user", "content": user_prompt},
            ]
            extra_body = None
            max_tokens = 4096
            temperature = 0.0

        try:
            res = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
                stream=False,
            )
            return res.choices[0].message.content
        except Exception as e:
            err_text = str(e)
            if (
                "maximum context length" in err_text
                or "'max_tokens' or 'max_completion_tokens' is too large" in err_text
            ):
                return "ERROR_CONTEXT_LIMIT"

            if "timeout" not in err_text.lower():
                logger.error(f"⚠️ API Call Error: {e}")
            return None

    async def evaluate_index(
        self, records: list[dict], idx: int, metric: str, model: str
    ) -> tuple[int, str]:
        story = records[idx].get("regenerated_story", "")
        if not story:
            return SCORE_PROMPT_ERROR, "Missing regenerated_story"

        prompt_text = get_all_prompts(story).get(metric)
        if not prompt_text or not isinstance(prompt_text, str):
            return SCORE_PROMPT_ERROR, "Logic Error: Prompt was None or Empty"

        model_system_msg = self.system_msg

        for attempt in range(MAX_RETRIES + 1):
            raw_output = await self.call_api_one_shot(
                model, prompt_text, model_system_msg
            )

            if raw_output == "ERROR_CONTEXT_LIMIT":
                return SCORE_CONTEXT_LIMIT, "Context Limit Exceeded"

            if raw_output is None:
                if attempt >= MAX_RETRIES:
                    return SCORE_CONTEXT_LIMIT, "Exception: API returned empty response"
                continue

            parsed = parse_response(raw_output)
            if parsed.get("score") is not None:
                return int(parsed["score"]), str(parsed.get("reason") or "")

            if attempt >= MAX_RETRIES:
                return SCORE_PARSE_FAILURE, "Parse Failure after retries"

        return SCORE_CONTEXT_LIMIT, "Exception: unexpected retry flow"

    async def run_metric(
        self,
        records: list[dict],
        model: str,
        metric: str,
        pending_indices: list[int],
        log_path: Path,
    ):
        """
        Evaluate all pending_indices for one metric.

        Architecture:
          - N worker coroutines call the API and put (idx, score, reason) into
            a write_queue.  They do NO file I/O.
          - A single writer coroutine drains write_queue, updates the in-memory
            record, and immediately appends one compact JSON line to log_path.
            Because there is only one writer there are no races and no locking.
        """
        if not pending_indices:
            return

        pbar = tqdm(
            total=len(pending_indices), desc=f"{_safe_model_name(model)}-{metric}"
        )
        eval_queue: asyncio.Queue[int] = asyncio.Queue()
        write_queue: asyncio.Queue[tuple | None] = asyncio.Queue()
        for idx in pending_indices:
            eval_queue.put_nowait(idx)

        async def writer_loop():
            # Keep the log file open for the duration of this metric.
            with open(log_path, "a", encoding="utf-8") as log_f:
                while True:
                    item = await write_queue.get()
                    if item is None:  # shutdown sentinel
                        write_queue.task_done()
                        break
                    w_idx, score, reason = item
                    _set_metric_result(records[w_idx], metric, model, score, reason)
                    pbar.update(1)
                    if not self.cfg.debug_mode:
                        key = _record_key(records[w_idx])
                        if key:
                            log_f.write(
                                json.dumps(
                                    {"k": key, "m": metric, "s": score, "r": reason},
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            log_f.flush()  # kernel buffer — fast, survives Python crash
                    write_queue.task_done()

        async def worker_loop():
            while True:
                try:
                    idx = await eval_queue.get()
                except asyncio.CancelledError:
                    break
                try:
                    score, reason = await self.evaluate_index(
                        records, idx, metric, model
                    )
                    await write_queue.put((idx, score, reason))
                finally:
                    eval_queue.task_done()

        writer_task = asyncio.create_task(writer_loop())
        workers = [
            asyncio.create_task(worker_loop())
            for _ in range(self.cfg.concurrency_for(model))
        ]

        # Wait for all API calls to complete and all results to enter write_queue.
        await eval_queue.join()
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        # Drain write_queue, then send the sentinel to stop the writer.
        await write_queue.join()
        await write_queue.put(None)
        await writer_task

        pbar.close()

    async def run_model(
        self,
        base_records: list[dict],
        seed_records: list[dict] | None,
        model: str,
        metrics: list[str],
    ):
        model_path = model_output_path(self.cfg.output_path, model)
        log_path = _results_log_path(model_path)
        logger.info(f"🚀 Processing model: {model}")
        logger.info(f"📁 Model result file: {model_path}")
        logger.info(f"📝 Results log:       {log_path}")

        records = ensure_model_split_records(
            self.cfg,
            base_records,
            seed_records,
            model,
            metrics,
            model_path,
        )

        # Replay any results written to the log during a previous interrupted run.
        # The log only ever contains results for the metric that was interrupted;
        # completed metrics are always checkpointed into the split file first.
        if log_path.exists():
            stats = _apply_results_log(records, log_path, model)
            logger.info(
                f"Replayed log {log_path.name}: "
                f"applied={stats['applied']}, "
                f"skipped_unknown_key={stats['skipped_unknown_key']}, "
                f"corrupt={stats['corrupt']}"
            )

        logger.info("Step 2/3: Check pending tasks...")
        pending_map = get_pending_indices_map(records, metrics, model)
        if not any(pending_map[m] for m in metrics):
            logger.info(f"✅ Model already complete, skipping: {model}")
            if log_path.exists():
                log_path.unlink()
                logger.info(f"🗑️  Removed stale log: {log_path.name}")
            return

        logger.info("Step 3/3: Start processing pending tasks...")
        for metric in metrics:
            pending_indices = pending_map.get(metric, [])
            await self.run_metric(records, model, metric, pending_indices, log_path)

            # Checkpoint: persist all progress so far into the split file, then clear the log.  The next metric starts with a clean empty log.
            if not self.cfg.debug_mode:
                await self.save_records_async(
                    model_path,
                    records,
                    reason=f"checkpoint after metric: {metric}",
                )
                if log_path.exists():
                    log_path.unlink()
                    logger.info(f"🗑️  Cleared log after checkpoint: {log_path.name}")

        logger.info(f"✅ Finished model: {model} -> {model_path}")


async def wait_for_next_model(next_model: str) -> bool:
    while True:
        user_input = await asyncio.to_thread(
            input,
            f"\nSwitch vLLM to '{next_model}', then type 'c' to continue or 'q' to stop: ",
        )
        user_input = user_input.strip().lower()
        if user_input in {"c", "continue", ""}:
            return True
        if user_input in {"q", "quit", "stop"}:
            return False
        logger.info("Please type 'c' to continue or 'q' to stop.")


async def main():
    cfg = EvalConfig.from_dict(CONFIG)

    # ------------------------------------------------------------------ #
    # Step 1: Determine canonical base records.                            #
    # Priority: raw dataset > any existing model split > error.            #
    # The combined output file is NEVER used as input here.                #
    # ------------------------------------------------------------------ #
    canonical_base_records: list[dict] = []
    canonical_base_label = ""

    raw_records = load_records(cfg.dataset_path, quiet=True)
    if raw_records:
        logger.info(f"Loaded raw dataset rows: {len(raw_records)}")
        canonical_base_records = raw_records
        canonical_base_label = f"raw dataset ({cfg.dataset_path})"
    else:
        logger.warning(
            f"Raw dataset not found at {cfg.dataset_path}. "
            "Falling back to first available model split file."
        )
        for model in cfg.evaluation_models:
            split_path = model_output_path(cfg.output_path, model)
            if split_path.exists():
                split_records = load_records(split_path, quiet=True)
                if split_records:
                    canonical_base_records = split_records
                    canonical_base_label = f"model split ({split_path})"
                    break

    if not canonical_base_records:
        logger.error(
            "No canonical base found. Provide the raw dataset "
            f"at {cfg.dataset_path} or ensure at least one model split file exists."
        )
        return

    logger.info(
        f"Canonical base: {canonical_base_label} ({len(canonical_base_records)} rows)"
    )

    # ------------------------------------------------------------------ #
    # Step 2: Discover metrics from a sample story.                        #
    # ------------------------------------------------------------------ #
    sample_story = next(
        (
            r["regenerated_story"]
            for r in canonical_base_records
            if r.get("regenerated_story")
        ),
        None,
    )
    if not sample_story:
        logger.error("No stories found in canonical base.")
        return

    metrics = list(get_all_prompts(sample_story).keys())
    logger.info(f"Metrics ({len(metrics)}): {metrics}")

    # ------------------------------------------------------------------ #
    # Step 3: Evaluate each model.                                         #
    # Seed source per model = that model's own split file only.            #
    # The combined output file is never read here.                         #
    # ------------------------------------------------------------------ #
    evaluator = StoryEvaluator(cfg)
    for model_idx, model in enumerate(cfg.evaluation_models):
        split_path = model_output_path(cfg.output_path, model)

        # Load only this model's own split file as seed (to recover completed
        # scores if a rebuild is needed). Never fall back to the combined output.
        seed_records: list[dict] | None = None
        if split_path.exists():
            loaded = load_records(split_path, quiet=True)
            if loaded:
                seed_records = loaded
                logger.info(
                    f"Seed source for {model}: {split_path.name} ({len(loaded)} rows)"
                )
        if seed_records is None:
            logger.info(f"Seed source for {model}: none (fresh start)")

        await evaluator.run_model(
            canonical_base_records,
            seed_records,
            model,
            metrics,
        )

        if model_idx < len(cfg.evaluation_models) - 1:
            next_model = cfg.evaluation_models[model_idx + 1]
            should_continue = await wait_for_next_model(next_model)
            if not should_continue:
                logger.info("🛑 Stopped by user before next model.")
                break

    logger.info(
        "✅ All models evaluated. Run merge_evaluations.py to merge split files."
    )


if __name__ == "__main__":
    asyncio.run(main())
