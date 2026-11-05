"""Offline LoRA training. Never downloads models or executes model-supplied code."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", WANDB_DISABLED="true")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.steps <= 500:
        parser.error("steps must be 1–500")
    model_path = Path(args.model).resolve(strict=True)
    if not model_path.is_dir():
        parser.error("model must be a local Hugging Face model folder, not GGUF")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise SystemExit(f"Missing training dependency: {exc}. Install training/requirements.txt in a dedicated Python 3.11/3.12 environment and set TEACHER_PYTHON to its interpreter.")

    random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(6, os.cpu_count() or 1))
    print(f"OFFLINE TRAINING | device={device} | model={model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise SystemExit("This classroom requires an instruction model with a tokenizer chat template.")
    raw = Path(args.data).read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("Training set is empty.")
    encoded = []
    for row in rows:
        if not row.get("metadata", {}).get("taskId", "").startswith("train-"):
            raise SystemExit("Refusing a non-training example. Export a teaching pack from the app.")
        messages = row["messages"]
        full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
        if full[:len(prefix)] != prefix or len(prefix) >= len(full):
            raise SystemExit("Chat template prefix is incompatible with assistant-only loss masking.")
        if len(full) > 2048:
            raise SystemExit("An example exceeds 2048 tokens. Shorten it; examples are never silently truncated.")
        encoded.append((full, [-100] * len(prefix) + full[len(prefix):]))
    print(f"Validated {len(encoded)} examples. Loss is computed only on assistant responses.", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32, attn_implementation="eager")
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM", bias="none"))
    model.to(device)
    model.config.use_cache = False
    model.train()
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.0002)
    started = time.perf_counter()
    metrics = []
    order = list(range(len(encoded)))
    for step in range(args.steps):
        if step % len(order) == 0:
            random.shuffle(order)
        tokens, labels = encoded[order[step % len(order)]]
        ids = torch.tensor([tokens], device=device)
        target = torch.tensor([labels], device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=target).loss
        if not torch.isfinite(loss):
            raise SystemExit("Non-finite loss; training stopped without publishing an adapter.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        item = {"step": step + 1, "loss": float(loss.detach().cpu()), "elapsedSeconds": time.perf_counter() - started}
        metrics.append(item)
        print(f"STEP {step + 1}/{args.steps} | loss={item['loss']:.5f} | elapsed={item['elapsedSeconds']:.1f}s", flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output / "adapter"), safe_serialization=True)
    tokenizer.save_pretrained(str(output / "adapter"))
    manifest = {"status": "trained-not-evaluated", "base_model": str(model_path), "dataset_sha256": hashlib.sha256(raw).hexdigest(), "examples": len(rows), "steps": args.steps, "seed": 42, "device": device, "hyperparameters": {"rank": 8, "alpha": 16, "dropout": 0.05, "lr": 0.0002, "batch_size": 1, "dtype": "float32", "loss": "assistant-only"}, "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft"]}, "metrics": metrics}
    (output / "training.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"SAVED {output / 'adapter'}\nTraining complete. Improvement is unproven until an independent exam is run.", flush=True)


if __name__ == "__main__":
    main()
