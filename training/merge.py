"""Merge a local LoRA adapter into a new local HF model folder for conversion."""
import argparse
import os
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
parser = argparse.ArgumentParser()
parser.add_argument("--base", required=True)
parser.add_argument("--adapter", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
base = Path(args.base).resolve(strict=True)
adapter = Path(args.adapter).resolve(strict=True)
output = Path(args.output).resolve()
if output.exists():
    raise SystemExit("Output already exists. Choose a new output folder; existing models are not overwritten.")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
model = AutoModelForCausalLM.from_pretrained(str(base), local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True)
model = model.merge_and_unload()
model.save_pretrained(str(output), safe_serialization=True)
AutoTokenizer.from_pretrained(str(base), local_files_only=True, trust_remote_code=False).save_pretrained(str(output))
print(f"Merged local model saved to {output}. Convert with a compatible llama.cpp converter, then import into Lemonade.")
