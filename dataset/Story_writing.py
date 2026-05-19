import os
import json
import logging
import asyncio
import random
from pathlib import Path
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from datasets import load_dataset
from openai import AsyncOpenAI
import spacy

# --- Basic Setup ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

NLP = spacy.blank("en")

# --- Centralized Configuration ---
CONFIG = {
    "model_name": "google/gemma-3-27b-it",
    "dataset_name": "euclaise/writingprompts",
    "output_path": Path("../data/new_story_dataset.jsonl"),
    "api_base_url": "http://0.0.0.0:8000/v1",
    "api_key": "EMPTY",
    "CONCURRENCY_LIMIT": 256,
    "SAVE_INTERVAL": 500,
    "MAX_RETRIES": 5,
    "word_count_min": 2000,
    "word_count_max": 8000,
}


# --- Core Functions ---
def count_words(text: str) -> int:
    """Counts words using spaCy, ignoring punctuation and spaces."""
    if not text:
        return 0
    doc = NLP.make_doc(text)
    return sum(1 for token in doc if not token.is_punct and not token.is_space)


def save_records_to_jsonl(records: list[dict], path: Path):
    """
    Atomically saves a list of records to a JSONL file.
    Writes to a temporary file first, then renames it to prevent corruption.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Atomic operation: rename the completed temp file to the final path.
    os.replace(temp_path, path)


def load_records_from_jsonl(path: Path) -> list[dict]:
    """Loads records from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def create_initial_records(dataset) -> list[dict]:
    """Filters the raw dataset and structures it into a list of records."""
    records = []
    for item in tqdm(dataset, desc="Structuring initial dataset"):
        word_count = count_words(item["story"])
        if not (0 < word_count <= CONFIG["word_count_max"]):
            continue

        needs_regen = word_count < CONFIG["word_count_min"]

        records.append(
            {
                "prompt": item["prompt"],
                "story": item["story"],
                "regenerated_story": "" if needs_regen else item["story"],
                "word_count": word_count,
                "needs_regeneration": needs_regen,
                "generated_model": "",
                "Fluency1": "",
                "Fluency2": "",
                "Fluency3": "",
                "Fluency4": "",
                "Fluency5": "",
                "Flexibility1": "",
                "Flexibility2": "",
                "Flexibility3": "",
                "Originality1": "",
                "Originality2": "",
                "Originality3": "",
                "Elaboration1": "",
                "Elaboration2": "",
                "Elaboration3": "",
                "overall_score": "",
            }
        )

    total = len(records)
    regen_count = sum(r["needs_regeneration"] for r in records)
    logger.info(f"Structured dataset with {total} items created.")
    logger.info(
        f"Items requiring regeneration (<{CONFIG['word_count_min']} words): {regen_count}"
    )
    return records


def build_story_messages(prompt: str, story: str) -> list:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert storyteller who writes engaging, imaginative, and high-quality long-form fiction. "
                "Craft original stories that are creative, coherent, and emotionally compelling. "
                "Your stories should feature well-developed characters, vivid world-building, and a strong plot "
                "with a clear beginning, middle, and end. Avoid clichés, repetition, or derivative writing, "
                "and maintain narrative flow throughout. "
                "**You must write the entire story exclusively in English.**"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Write a complete story between 4,000 and 8,000 words based on the following prompt:\n\n"
                f"{prompt}\n\n"
                f"Use the following sample stories only as inspiration for tone, pacing, and style "
                f"(do not copy or reuse their content):\n\n{story}"
            ),
        },
    ]


async def generate_one_story(
    client: AsyncOpenAI, semaphore: asyncio.Semaphore, idx: int, record: dict
) -> tuple[int, str | None]:
    """Generate one story via vLLM server with retry."""
    messages = build_story_messages(record["prompt"], record["story"])
    max_retries = int(CONFIG["MAX_RETRIES"])

    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=CONFIG["model_name"],
                    messages=messages,
                    temperature=1.0,
                    top_p=0.95,
                    extra_body={
                        "top_k": 64,
                        "min_p": 0.0,
                        "repetition_penalty": 1.0,
                        "max_tokens": 13312,
                    },
                )
            text = (response.choices[0].message.content or "").strip()
            if text:
                return idx, text
            raise ValueError("Empty response content")
        except Exception as exc:
            if attempt == max_retries - 1:
                logger.error(f"Generation failed for idx={idx}: {exc}")
                return idx, None
            backoff = min(2**attempt, 20) + random.random()
            await asyncio.sleep(backoff)

    return idx, None


async def process_generation_async(all_records: list[dict], output_path: Path):
    indices_to_process = [
        i
        for i, r in enumerate(all_records)
        if r.get("needs_regeneration") and not r.get("generated_model")
    ]

    if not indices_to_process:
        logger.info("All stories have been processed. Job complete. ✅")
        return

    total_to_process = len(indices_to_process)
    logger.info(f"Found {total_to_process} stories remaining to regenerate.")
    logger.info(
        f"Using vLLM server mode at {CONFIG['api_base_url']} with async workers={CONFIG['CONCURRENCY_LIMIT']}"
    )

    client = AsyncOpenAI(base_url=CONFIG["api_base_url"], api_key=CONFIG["api_key"])
    semaphore = asyncio.Semaphore(int(CONFIG["CONCURRENCY_LIMIT"]))
    tasks = [
        asyncio.create_task(
            generate_one_story(client, semaphore, idx, all_records[idx])
        )
        for idx in indices_to_process
    ]

    processed_count = 0
    success_count = 0
    with async_tqdm(
        total=total_to_process, desc="Generating Stories (server async)"
    ) as pbar:
        for future in asyncio.as_completed(tasks):
            idx, generated_text = await future
            if generated_text:
                all_records[idx]["regenerated_story"] = generated_text
                all_records[idx]["word_count"] = count_words(generated_text)
                all_records[idx]["generated_model"] = CONFIG["model_name"]
                success_count += 1

            processed_count += 1
            pbar.update(1)

            if processed_count % int(CONFIG["SAVE_INTERVAL"]) == 0:
                save_records_to_jsonl(all_records, output_path)
                logger.info(
                    f"Checkpoint saved. Processed {processed_count}/{total_to_process}, successful={success_count}."
                )

    save_records_to_jsonl(all_records, output_path)
    logger.info(
        f"Generation finished. Successful={success_count}/{total_to_process}. Final dataset at: {output_path}"
    )


async def main():
    """Main execution function for the data processing pipeline."""
    output_path = CONFIG["output_path"]
    # Step 1: Load existing dataset or create a new one.
    if output_path.exists():
        logger.info(f"Resuming from existing dataset: {output_path}")
        all_records = load_records_from_jsonl(output_path)
    else:
        logger.info("Processing dataset for the first time...")
        raw_dataset = load_dataset(CONFIG["dataset_name"], split="train")
        # To test with a small subset: raw_dataset = raw_dataset.select(range(1000))
        all_records = create_initial_records(raw_dataset)
        save_records_to_jsonl(all_records, output_path)
        logger.info(f"Initial dataset created and saved as first checkpoint.")

    # Step 2: Run async generation via vLLM server.
    await process_generation_async(all_records, output_path)


if __name__ == "__main__":
    asyncio.run(main())
