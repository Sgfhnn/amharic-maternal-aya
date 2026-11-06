"""Merge a LoRA adapter into its base model and save a full fp16 snapshot (used to stack run-two SFT -> v3 SFT -> DPO).

Usage: python merge_adapter.py BASE_DIR ADAPTER_DIR OUT_DIR [--fp32]
"""
import glob, os, shutil, sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

base, adapter, out = sys.argv[1:4]
dt = torch.float32 if "--fp32" in sys.argv or os.environ.get("SFT_TEST") == "1" else torch.float16
m = AutoModelForCausalLM.from_pretrained(base, dtype=dt, device_map="cpu")
m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
os.makedirs(out, exist_ok=True)
m.save_pretrained(out, safe_serialization=True)
for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"]:
    for src in (f"{base}/{f}",):  # base tokenizer files (stock chat template, no generation tags)
        if os.path.exists(src):
            shutil.copy(src, f"{out}/{f}"); break
print("merged", adapter, "into", base, "->", out, dt)
