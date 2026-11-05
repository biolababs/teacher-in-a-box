# Teacher in a Box

A local AI classroom for teaching a small language model a robot command language. Diagnose → generate targeted lessons → review → fine-tune locally → independently evaluate.

This is an early working prototype. **No model has been trained or shown to improve just by installing this app.** Reference playback uses a deterministic solver and is labeled separately from model results. The final exam has 12 fixed scenes: useful for a demo, insufficient for a broad generalization claim.

## Run locally

Requires Node.js 22+. The app has **zero npm dependencies**.

```powershell
npm start
```

Open http://127.0.0.1:4173. Lemonade defaults to http://127.0.0.1:13305/api/v1. Change it using the connection button. The app only connects to HTTP loopback addresses, rejects redirects, binds to 127.0.0.1, blocks cross-origin writes, uses local fonts/assets, and has no cloud inference or analytics integration.

The simulator and reference curriculum work without a model. Download an instruction model in Lemonade to enable real inference. Model files and training dependencies require initial acquisition; once available, the classroom can work without internet. Use a local Lemonade model; do not configure cloud-backed model aliases. This app cannot enforce the behavior of a separately configured inference server.

## Classroom workflow

1. **Playground:** run an explicitly labeled reference plan or enter JSON actions. Choose a local student and run nine diagnostic tasks. Compare instruction-only and few-shot baselines.
2. **Lesson studio:** a local teacher proposes plans for six training scenes in the weakest diagnosed skill. The simulator rejects invalid plans. Alternatively add the solver-generated reference curriculum. Inspect each board/plan and check approval.
3. **Training lab:** export only approved, valid training examples. An export includes a dataset, provenance, and a hash. Point the local trainer at a Hugging Face model directory. GGUF files cannot serve as training inputs.
4. **Report cards:** after importing the trained model into Lemonade, run the same held-out exam against the original and trained versions, with identical prompt mode. Include the few-shot baseline. Raw exports include failures, latency, model ID, prompt settings, and suite hash.

Thirty training scenes, nine diagnostic scenes and twelve exam scenes have disjoint layouts. Exam scenes never enter teacher prompts or training exports. Diagnostics guide curriculum selection; they are not an unbiased final score. Repeatedly optimizing against exam scores can overfit: reserve an additional independently authored test set for publication. Fixed decoding settings do not guarantee identical output across backends.

## Local fine-tuning setup

Use a dedicated Python 3.11/3.12 environment and a small instruction model whose architecture is supported by Transformers, PEFT and the chosen deployment converter. GPU acceleration is optional; this trainer supports CPU float32 and uses CUDA/ROCm through PyTorch where available. CPU training can be slow. The first version deliberately uses plain LoRA, batch size 1, not QLoRA or a promise of accelerated AMD training.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r training\requirements.txt
$env:TEACHER_PYTHON = (Resolve-Path .\.venv\Scripts\python.exe).Path
npm start
```

Install the appropriate PyTorch build for your hardware using its official instructions. Download a compatible Hugging Face instruction model separately into a local folder (including tokenizer, config and safetensors). An inference GGUF and the original training weights are different artifacts. Check the base model license before redistribution.

The app spawns the local trainer with network-disabled Hugging Face settings and `trust_remote_code=False`. It never automatically downloads weights. Assistant-only labels exclude prompt tokens from loss; incompatible templates and oversized examples are rejected rather than silently altered. Training writes LoRA adapters and a manifest with versions, hyperparameters, losses and dataset hash under `.local/training/`. Existing jobs cannot run concurrently. Cancelling a job terminates its process; an incomplete job is not a trained model.

Training can also be run directly:

```powershell
.\.venv\Scripts\python training\train.py --model C:\models\student --data .local\exports\EXPORT_ID\train.jsonl --output .local\training\my-run --steps 20
```

## Bring the student back to Lemonade

This prototype produces a real LoRA adapter when the dependencies and weights are available. **Conversion and import are manual**, and depend on model architecture support.

```powershell
.\.venv\Scripts\python training\merge.py --base C:\models\student --adapter .local\training\RUN_ID\adapter --output .local\merged-student
```

Use the official llama.cpp `convert_hf_to_gguf.py` with the merged model folder and an output GGUF path; install that converter's dependencies separately. Optionally quantize using `llama-quantize`. Follow Lemonade's custom-model import instructions for your installed version, then refresh the app connection and select the imported model. Compare equivalent quantization/configuration for baseline and trained models. Do not claim a training improvement from a change of quantization or prompts.

References:
- https://lemonade-server.ai/docs/guide/configuration/custom-models/
- https://github.com/ggml-org/llama.cpp
- https://huggingface.co/docs/peft/developer_guides/lora
- https://developer.amd.com/playbooks/unsloth-llms-finetuning/

## Data and reproducibility

All classroom state lives in `.local/classroom.json`. Approved JSONL teaching packs live in `.local/exports/`; adapters and training records live in `.local/training/`. These directories are excluded from Git. No exam scores are prefilled. Inference runs persist completed/partial results. Active job status is in memory; restarting the app does not resume a job. Back up `.local` to preserve your work.

For a challenge submission, publish the app, a reviewed teaching pack, a pinned environment, exact base/converted model hashes, hardware/backend settings, training cost, and complete before/after results. Include unsuccessful runs. Do not publish private data or model files without redistribution rights.

## Verify

```powershell
npm test
npm run check
python -m py_compile training/train.py training/merge.py
```

Tests cover solver correctness, scene separation, simulator constraints, mutation isolation, prompt baselines, loopback enforcement, and the local inference adapter. Real inference and training require local model files and are separate integration checks.

## License

MIT. Base models and third-party training packages retain their own licenses.
