import logging
import os
from dataclasses import dataclass, field
import torch
import wandb
import subprocess
import sys
from accelerate import PartialState
from datasets import load_from_disk, concatenate_datasets
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, HfArgumentParser
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed internal constants (not user-facing)
# ---------------------------------------------------------------------------

LR_SCHEDULER_TYPE          = "cosine"
WARMUP_RATIO               = 0.05
LORA_TARGET_MODS  = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]

# Maps train_mode → dataset message column
TRAIN_MODE_TO_COLUMN: dict[str, str] = {
    "score_only":           "messages_score_only",
    "score_with_reasoning": "messages_score_with_reasoning",
    "score_with_reviews":   "messages_score_with_reviews",
    "review_with_reasoning":"messages_review_with_reasoning",
}
_REASONING_MODES: set[str] = {"score_with_reasoning", "review_with_reasoning"}


# ---------------------------------------------------------------------------
# User-facing arguments
# ---------------------------------------------------------------------------

@dataclass
class ScriptArguments:
    """Arguments for the TTCW reviewer model training script."""

    # Data & Model
    dataset: str = field(
        default="/path/to/TTCW_sft_dataset",
        metadata={"help": "Local path to a TTCW dataset saved with save_to_disk()"},
    )
    messages_column: str = field(
        default="score_only",
        metadata={"help": (
            "Fine-tuning mode. One of:\n"
            "  score_only            — predict scores only (no reasoning)\n"
            "  score_with_reasoning  — predict scores with <thinking> CoT\n"
            "  score_with_reviews    — predict scores + synthesized reviews\n"
            "  review_with_reasoning — predict scores + reviews with <thinking> CoT\n"
            "  mixed                 — train on all four tasks combined (shuffled)"
        )},
    )
    model: str = field(
        default="Qwen/Qwen3-8B",
        metadata={"help": "Model name or path on the HuggingFace Hub or local directory"},
    )
    output_dir: str = field(
        default="/path/to/models/ttcw-reviewer",
        metadata={"help": "Base directory for saving models; each run is saved to output_dir/run_name/"},
    )
    # Training
    epochs: int = field(default=1, metadata={"help": "Number of training epochs"})
    batch_size: int = field(default=2, metadata={"help": "Per-device train batch size"})
    accumulation_steps: int = field(
        default=4, metadata={"help": "Gradient accumulation steps"}
    )
    learning_rate: float = field(default=2e-4, metadata={"help": "Learning rate"})
    seed: int = field(default=42, metadata={"help": "Random seed"})
    # LoRA  
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha scaling factor"})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout"})
    # Sequence length
    max_seq_length: int = field(
        default=16384,
        metadata={"help": "Maximum tokenised sequence length. Reasoning modes typically need 32768+."},
    )
    # W&B
    use_wandb: bool = field(default=True, metadata={"help": "Enable W&B logging"})
    wandb_project: str = field(
        default="TTCW_reviewer", metadata={"help": "W&B project name"}
    )
    wandb_run_name: str = field(
        default="",
        metadata={"help": "W&B run name; leave empty to auto-generate as '{model}-{timestamp}'"},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_run_name(model: str, messages_column: str) -> str:
    """Generate a run name encoding model, training mode, and timestamp."""
    model_short = model.rstrip("/").split("/")[-1]
    return f"{model_short}-{messages_column}"


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def training(args: ScriptArguments) -> None:
    run_name = args.wandb_run_name or _auto_run_name(args.model, args.messages_column)
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Run name:   %s", run_name)
    logger.info("Output dir: %s", output_dir)

    distributed_state = PartialState()

    if distributed_state.is_main_process and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )

    # ---- dataset ----
    logger.info("Loading dataset from %s", args.dataset)
    dataset_dict = load_from_disk(args.dataset)
    train_dataset = dataset_dict["train"]

    logger.info("Train size: %d", len(train_dataset))

    # ---- model + tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoLigerKernelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        rope=True,
        rms_norm=True,
        swiglu=True,
        fused_linear_cross_entropy=True,
        use_cache=False,
    )

    # ---- LoRA ----
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODS,
        bias="none",
        use_rslora=True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- select message columns ----
    # Each column is a struct {prompt: [...], completion: [...]}.
    logger.info("Train mode: %s", args.messages_column)

    thinking_system = (
        "detailed thinking on"
        if args.messages_column in _REASONING_MODES
        else "detailed thinking off"
    )

    def _add_thinking_system(prompt: list) -> list:
        return [{"role": "system", "content": thinking_system}] + list(prompt)

    def prepare(ds):
        col = TRAIN_MODE_TO_COLUMN[args.messages_column]
        flat = ds.select_columns([col]).flatten()
        flat = flat.rename_columns({f"{col}.prompt": "prompt", f"{col}.completion": "completion"})
        return flat.map(
            lambda x: {"prompt": _add_thinking_system(x["prompt"])},
            num_proc=64,
            desc="Adding thinking system prompt",
        )

    train_dataset = prepare(train_dataset)

    # ---- SFT config ----
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        optim="adamw_torch_fused",
        warmup_ratio=WARMUP_RATIO,
        bf16=True,
        tf32=True,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        seed=args.seed,
        run_name=run_name,
        max_length=None,
        padding_free=True,
        packing=False,
        report_to="wandb" if (distributed_state.is_main_process and args.use_wandb) else "none",
        dataloader_num_workers=64,
        dataset_num_proc=64,
        dataloader_prefetch_factor=4,
        dataloader_pin_memory=True,
        completion_only_loss=True,
        torch_empty_cache_steps=4,
        use_liger_kernel=True,
        liger_kernel_config={
            "rope": True,
            "rms_norm": True,
            "swiglu": True,
            "fused_linear_cross_entropy": True,
        },
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # max_steps=10,
    )

    # ---- trainer ----
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=sft_config,
    )

    logger.info("Starting training (%d epoch(s))...", args.epochs)
    trainer.train()

    # Save adapter via trainer (handles ZeRO-3 parameter gathering correctly).
    # Direct merge_and_unload() fails under ZeRO-3 because weights are sharded.
    adapter_dir = output_dir + "_adapter_tmp"
    logger.info("Saving LoRA adapter to %s (rank %d)...", adapter_dir, distributed_state.process_index)
    trainer.save_model(adapter_dir)
    trainer.accelerator.wait_for_everyone()

    # Merge offline via subprocess on rank 0 — subprocess has no DeepSpeed context,
    # which avoids the "ZeRO-3 incompatible with device_map" error.
    if distributed_state.is_main_process:
        tokenizer.save_pretrained(adapter_dir)
        logger.info("Merging LoRA weights via subprocess...")
        merge_script = os.path.join(os.path.dirname(__file__), "merge_lora.py")
        subprocess.run(
            [sys.executable, merge_script,
             "--base_model", args.model,
             "--adapter_dir", adapter_dir,
             "--output_dir", output_dir],
            check=True,
        )
        logger.info("Merged model saved to %s", output_dir)

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    (args,) = parser.parse_args_into_dataclasses()
    training(args)
