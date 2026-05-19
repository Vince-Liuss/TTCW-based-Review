# TTCW-based-Review

**[Paper coming soon]**

A framework for building LLM-based reviewers of creative writing using the TTCW (Thinking Through Creative Writing) evaluation criteria. The pipeline covers dataset construction, model training, and evaluation across 14 structured writing quality metrics.

---

## TTCW Metrics (14)

| Dimension | Metrics |
|-----------|---------|
| Fluency (5) | Narrative Pacing, Scene vs Exposition Balance, Language Proficiency, Narrative Ending Quality, Understandability |
| Flexibility (3) | Perspective and Voice, Emotional Flexibility, Structural Flexibility |
| Originality (3) | Theme/Takeaway, Thought/Cliche Avoidance, Form/Structure |
| Elaboration (3) | World-Building, Character Development, Rhetorical Complexity |

---

## Setup

Requires **Python 3.12** and **CUDA 12.8**.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync --python 3.12
```

> `pyproject.toml` pulls PyTorch from the CUDA 12.8 index (`cu128`). Ensure your system has CUDA 12.8 drivers installed before running `uv sync`.

Requires 4 CUDA GPUs. Update the data and model paths at the top of each script before running.

### vLLM Server

The dataset pipeline scripts do **not** load models directly — they call an OpenAI-compatible API served by vLLM. You must start the server with the appropriate model before each step:

```bash
vllm serve <model-name> \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

Each pipeline step requires a different model:

| Step | Script | Required model type |
|------|--------|-------------------|
| Story generation | `Story_writing.py` | A capable instruction-following LLM (e.g. `google/gemma-3-27b-it`) |
| Story evaluation | `Story_evaluator_api.py` | Judge LLMs — script runs each model listed in `CONFIG["evaluation_models"]` sequentially, restart the server for each |
| Review synthesis | `summarize_reviews.py` | A capable instruction-following LLM (e.g. `zai-org/GLM-4.5-Air`) |
| Dataset validation | `quality_validate_reviews_comments.py` | Any capable LLM for binary quality judgement |

The model name passed to `vllm serve` must exactly match the name configured in each script's `CONFIG` dict.

---

## Dataset Pipeline

```bash
# 1) Generate stories from writing prompts
cd dataset && python Story_writing.py

# 2) Evaluate stories with multiple LLM judges (resumable)
cd dataset && python Story_evaluator_api.py

# 3) Merge per-model evaluation splits
cd dataset && python merge_evaluations.py

# 4) Synthesize per-metric consensus reviews
cd dataset && python summarize_reviews.py

# 5) Build HF DatasetDict for SFT training
cd dataset && python dataset_build.py
```

### Dataset Quality Validation

Validate synthesized review quality using NLG evaluation dimensions (faithfulness, coherence, relevance):

```bash
# Start vLLM server first (see above)

python dataset/quality_validate_reviews_comments.py \
  --audit-model <model-name> \
  --input /path/to/story_evaluation_dataset_synthesized.jsonl \
  --sample-size 50 \
  --concurrency 8
```

Results written to `logs/`:
- `review_comment_quality_sample.jsonl` — per-item audit results
- `review_comment_quality_sample.summary.json` — summary statistics
- `review_comment_quality_sample.log.txt` — human-readable pass-rate table

### Dataset Analysis

Generate publication-quality figures (score distributions, inter-metric correlation heatmaps, discrimination scores):

```bash
cd dataset && python analysis.py
```

---

## Training

### Training Modes

| Mode | Description | Max Seq Length |
|------|-------------|----------------|
| `score_only` | Predict numeric scores only | 8192 |
| `score_with_reasoning` | Predict scores with step-by-step reasoning | 24576 |
| `score_with_reviews` | Predict scores with full written reviews | 16384 |
| `review_with_reasoning` | Generate reviews with reasoning traces | 32768 |

### Single Run

```bash
cd train
accelerate launch --deepspeed_config_file ../config/ds_config.json train.py \
  --dataset /path/to/TTCW_sft_dataset \
  --model Qwen/Qwen3-8B \
  --messages_column score_with_reviews \
  --epochs 1 \
  --max_seq_length 16384 \
  --batch_size 4 \
  --accumulation_steps 8 \
  --output_dir /path/to/output
```

### Multi-Model Training

```bash
# All configured models and modes
bash scripts/train_ttcw_all_modes.sh

# Nemotron-8B
bash scripts/train_ttcw_nemotron.sh
```

---

## Evaluation

```bash
# Full automated eval sweep across all saved checkpoints
bash scripts/evaluate_ttcw_auto.sh

# Use --fresh to clear previous outputs and rerun from scratch
bash scripts/evaluate_ttcw_auto.sh --fresh
```

---

## Repository Structure

```
├── dataset/
│   ├── prompts.py                          # TTCW metric evaluation prompts
│   ├── Story_writing.py                    # Story generation from writing prompts
│   ├── Story_evaluator_api.py              # Async multi-judge LLM evaluation (resumable)
│   ├── merge_evaluations.py                # Merge per-model evaluation splits
│   ├── summarize_reviews.py                # Synthesize consensus reviews
│   ├── dataset_build.py                    # Build HF SFT dataset
│   ├── quality_validate_reviews_comments.py # NLG-based dataset quality validation
│   └── analysis.py                         # Score distribution and correlation analysis
├── train/
│   ├── train.py                            # SFT + LoRA training (DeepSpeed ZeRO-3)
│   ├── merge_lora.py                       # Standalone LoRA merge
│   └── compute_metrics.py                  # Shared metric utilities (used by eval)
├── eval/
│   └── evaluate_vllm.py                    # Post-training evaluation with vLLM
├── scripts/
│   ├── train_ttcw_all_modes.sh             # Multi-model training orchestration
│   ├── train_ttcw_nemotron.sh              # Nemotron-8B training
│   └── evaluate_ttcw_auto.sh              # Batch eval across checkpoints
├── config/
│   └── ds_config.json                      # DeepSpeed ZeRO-3 configuration
├── pyproject.toml                          # Python dependencies (uv)
└── README.md
```

---

## Citation

```bibtex
@article{coming_soon,
  title   = {},
  author  = {},
  year    = {2025},
}
```

---

## License

See [LICENSE](LICENSE).
