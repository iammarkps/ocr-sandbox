# CLAUDE.md

## Project Overview
Modal-based serverless OCR CLI using NVIDIA Nemotron (`nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`) for images and PDFs.

## Stack
- Runtime: Modal (serverless GPU)
- Model: `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`
- Python: 3.12
- Package manager: `uv`
- Locked deps: see `uv.lock`

## Default Runtime Profile (Performance)
- `gpu="A100"`
- `max_containers=5`
- `max_new_tokens=10000`
- `max_file_mb=200`
- `max_pdf_pages=200`
- `pdf_dpi=150`
- `max_image_side=2048`
- `max_image_pixels=80_000_000`
- `min_containers=1`
- `buffer_containers=1`
- `scaledown_window=1200`
- `staged_input_ttl_seconds=86400`

## Commands
```bash
# Download model to Modal Volume (run once)
uv run modal run app.py::download_model

# OCR image/PDF
uv run modal run app.py --file-path <path>

# Optional output and overwrite flags
uv run modal run app.py --file-path <path> --output <out.md>
uv run modal run app.py --file-path <path> --overwrite true
```

## Environment Overrides
Supported env vars:
- `NEMOTRON_OCR_GPU`
- `NEMOTRON_OCR_MIN_CONTAINERS`
- `NEMOTRON_OCR_BUFFER_CONTAINERS`
- `NEMOTRON_OCR_MAX_CONTAINERS` (hard-capped at 5)
- `NEMOTRON_OCR_SCALEDOWN_WINDOW`
- `NEMOTRON_OCR_MAX_NEW_TOKENS`
- `NEMOTRON_OCR_MAX_FILE_MB`
- `NEMOTRON_OCR_MAX_PDF_PAGES`
- `NEMOTRON_OCR_PDF_DPI`
- `NEMOTRON_OCR_PAGE_BATCH_SIZE`
- `NEMOTRON_OCR_PDF_PIPELINE`
- `NEMOTRON_OCR_MAX_IMAGE_SIDE`
- `NEMOTRON_OCR_MAX_IMAGE_PIXELS`
- `NEMOTRON_OCR_STAGED_INPUT_TTL_SECONDS`
- `NEMOTRON_OCR_MODEL_REVISION`

## Model Download and Integrity
- Model revision is controlled by `MODEL_REVISION` (default `main`; pin to a commit hash for reproducibility)
- `download_model` validates integrity before skipping:
  - required: `config.json`, `tokenizer.json`, `tokenizer_config.json`
  - required: at least one `*.safetensors` or `*.bin` weight shard
- Invalid partial model dirs are deleted and re-downloaded

## OCR Flow
- Image OCR: `run_page(image_bytes)`
- PDF OCR default: `stage_pdf_input(pdf_bytes, run_id)` + `run_pdf_range.starmap(...)`
  - stages the PDF once in the Modal input volume
  - enforces max page limit before OCR
  - writes ordered range results incrementally to `<output>.tmp`, then atomically replaces the final file
  - eagerly cleans up staged input on full success
  - defers cleanup on partial stream failures and relies on TTL-based stale-run cleanup during later staging
- Legacy PDF OCR fallback: `pdf_to_page_images(pdf_bytes)` + `run_page_batch.map(page_batches)`
  - renders pages to PNGs in Modal before OCR
- Class is configured with `enable_memory_snapshot=True` and `@modal.enter(snap=True)` to reduce repeated model load overhead.

## Nemotron Inference Gotchas
- `AutoProcessor.from_pretrained()` and `AutoModelForCausalLM.from_pretrained()` both require `trust_remote_code=True`.
- Always include a system message `{"role": "system", "content": "/no_think"}` to suppress chain-of-thought output.
- Use `tokenizer.apply_chat_template()` for text formatting (not `processor.apply_chat_template()`).
- In `model.generate()`, pass `pixel_values`, `input_ids`, and `attention_mask` as explicit kwargs — do NOT splat `**inputs`. The Hybrid Mamba+Attention architecture rejects unexpected BatchEncoding keys.
- Model dtype must be `torch.bfloat16` (it is the BF16 variant).
- Requires extra deps: `mamba-ssm==2.2.5`, `causal-conv1d`, `timm`, `open-clip-torch`.
- Max 4 images per inference call; processor handles image tiling internally (up to 12 tiles of 512×512).

## Base Docker Image
- Image: `ghcr.io/iammark/nemotron-ocr-base:latest`
- Contains: Python 3.12, PyTorch 2.7.1 (cu128), causal-conv1d, mamba-ssm==2.2.5
- Built by `.github/workflows/docker.yml` when `Dockerfile` changes on main
- Manual rebuild: `gh workflow run docker.yml` or `docker build -t ghcr.io/iammark/nemotron-ocr-base:latest .`
- **Important**: torch version in `Dockerfile` must match the pin in `pyproject.toml`

## Debugging
- Start with full traceback.
- Check preflight output for guardrail/config failures before debugging remote inference.
