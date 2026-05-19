"""Standalone LoRA merge script — runs outside DeepSpeed context.

Called by train.py after training via subprocess so that the active
DeepSpeed ZeRO-3 plugin does not interfere with from_pretrained().
"""
import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", required=True)
parser.add_argument("--adapter_dir", required=True)
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()

print(f"Loading base model {args.base_model} on CPU...")
base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map="cpu")
print("Merging LoRA adapter...")
merged = PeftModel.from_pretrained(base, args.adapter_dir).merge_and_unload()
merged.save_pretrained(args.output_dir, safe_serialization=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.base_model)
tokenizer.save_pretrained(args.output_dir)
print(f"Merged model saved to {args.output_dir}")
