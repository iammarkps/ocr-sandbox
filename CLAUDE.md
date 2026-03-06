# CLAUDE.md

## Project Overview
Modal-based serverless OCR CLI using Typhoon OCR (`scb10x/typhoon-ocr1.5-2b`) for images and PDFs.

## Stack
- Runtime: Modal (serverless GPU)
- Model: `scb10x/typhoon-ocr1.5-2b`
- Python: 3.12
- Package manager: `uv`
- Locked deps: see `uv.lock`

## Default Runtime Profile (Performance)
- `gpu="A10"`
- `max_containers=5`
- `max_new_tokens=10000`
- `max_file_mb=200`
- `max_pdf_pages=200`
- `pdf_dpi=150`
- `max_image_side=1800`
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
- `TYPHOON_OCR_GPU`
- `TYPHOON_OCR_MIN_CONTAINERS`
- `TYPHOON_OCR_BUFFER_CONTAINERS`
- `TYPHOON_OCR_MAX_CONTAINERS` (hard-capped at 5)
- `TYPHOON_OCR_SCALEDOWN_WINDOW`
- `TYPHOON_OCR_MAX_NEW_TOKENS`
- `TYPHOON_OCR_MAX_FILE_MB`
- `TYPHOON_OCR_MAX_PDF_PAGES`
- `TYPHOON_OCR_PDF_DPI`
- `TYPHOON_OCR_PAGE_BATCH_SIZE`
- `TYPHOON_OCR_PDF_PIPELINE`
- `TYPHOON_OCR_MAX_IMAGE_SIDE`
- `TYPHOON_OCR_MAX_IMAGE_PIXELS`
- `TYPHOON_OCR_STAGED_INPUT_TTL_SECONDS`
- `TYPHOON_OCR_MODEL_REVISION`

## Model Download and Integrity
- Model snapshot is pinned by revision (`MODEL_REVISION`, default `be9399b`)
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

## Qwen3-VL Gotchas
- `AutoProcessor.from_pretrained()` is broken for this setup in `transformers>=4.57`; construct `Qwen3VLProcessor` manually.
- Use `process_vision_info` from `qwen_vl_utils`.
- Do not pass `videos=None` to processor.

## Debugging
- Start with full traceback.
- Check preflight output for guardrail/config failures before debugging remote inference.
