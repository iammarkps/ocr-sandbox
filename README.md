# Typhoon OCR

Serverless OCR CLI powered by [Typhoon OCR 1.5 2B](https://huggingface.co/scb10x/typhoon-ocr1.5-2b) on [Modal](https://modal.com). It extracts text from images and PDFs as Markdown.

## Features

- Thai and English OCR
- PDF and image input support
- Markdown output with tables/equations/figures preserved
- High-throughput defaults for faster OCR

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- Modal account
- Hugging Face access token for model download

## Setup (Secure Token Workflow)

```bash
# Install project dependencies from lockfile
uv sync --locked

# Authenticate Modal via the project environment
uv run modal setup

# Enter HF token without putting it directly in shell history
read -s "HF_TOKEN?Enter Hugging Face token: "
echo
uv run modal secret create huggingface-secret HF_TOKEN="$HF_TOKEN"
unset HF_TOKEN

# Download model into Modal Volume (run once)
uv run modal run app.py::download_model
```

## Usage

```bash
# OCR an image or PDF
uv run modal run app.py --file-path document.png
uv run modal run app.py --file-path document.pdf

# Custom output path
uv run modal run app.py --file-path document.pdf --output /tmp/result.md

# Allow overwriting an existing output file
uv run modal run app.py --file-path document.pdf --overwrite true
```

By default, output is saved as `<input_basename>.md`.

## Performance Defaults

These defaults prioritize throughput and performance:

- `GPU`: `A10`
- `max_containers`: `5`
- `max_new_tokens`: `10000`
- `max_file_mb`: `200`
- `max_pdf_pages`: `200`
- `pdf_dpi`: `150`
- `pdf_page_batch_size`: `4`
- `pdf_pipeline`: `range_map` (`legacy` available as fallback)
- `max_image_side`: `1800`
- `max_image_pixels`: `80_000_000`
- `staged_input_ttl_seconds`: `86400` (24 hours)
- model revision pinned via `TYPHOON_OCR_MODEL_REVISION` (default: `be9399b`)

## Advanced Configuration (Environment Overrides)

```bash
export TYPHOON_OCR_GPU=A100
export TYPHOON_OCR_MIN_CONTAINERS=1
export TYPHOON_OCR_BUFFER_CONTAINERS=1
export TYPHOON_OCR_MAX_CONTAINERS=2
export TYPHOON_OCR_SCALEDOWN_WINDOW=1200
export TYPHOON_OCR_MAX_NEW_TOKENS=3072
export TYPHOON_OCR_MAX_FILE_MB=50
export TYPHOON_OCR_MAX_PDF_PAGES=80
export TYPHOON_OCR_PDF_DPI=200
export TYPHOON_OCR_PAGE_BATCH_SIZE=4
export TYPHOON_OCR_PDF_PIPELINE=range_map
export TYPHOON_OCR_MAX_IMAGE_SIDE=1800
export TYPHOON_OCR_MAX_IMAGE_PIXELS=20000000
export TYPHOON_OCR_STAGED_INPUT_TTL_SECONDS=86400
export TYPHOON_OCR_MODEL_REVISION=be9399b
```

For fastest startup with max parallel fan-out, pre-warm all workers:

```bash
export TYPHOON_OCR_MIN_CONTAINERS=5
export TYPHOON_OCR_BUFFER_CONTAINERS=5
```

## Guardrails and Failure Modes

- Unsupported extension: command exits before remote execution
- File larger than configured max MB: command exits locally
- PDF pages above `max_pdf_pages`: remote execution fails fast before OCR
- Existing output file without `--overwrite true`: command exits safely

Example local guardrail failures:

```bash
# Unsupported file type
uv run modal run app.py --file-path notes.txt

# Output exists and overwrite is disabled
uv run modal run app.py --file-path input.pdf --output ./input.md
```

## Processing Flow

1. Local preflight validates file type, size, and output path.
2. `range_map` pipeline stages the PDF once in a Modal volume, then `run_pdf_range.starmap(...)` processes page ranges in parallel GPU workers without materializing all rendered PNG pages locally.
3. Ordered range results are written incrementally to a temp output file as they arrive, then atomically swapped into place at the end.
4. `legacy` pipeline remains available and uses `pdf_to_page_images.remote(...)` + `run_page_batch.map(...)`.

Ordered streaming note:
- The command keeps deterministic output ordering with `order_outputs=True`.
- If an earlier range is slow, later completed ranges wait before being written.

Staged input cleanup note:
- Successful `range_map` runs eagerly remove the staged PDF from the Modal input volume.
- Failed or interrupted streams defer cleanup to a later `stage_pdf_input(...)` call, which deletes stale run directories older than `TYPHOON_OCR_STAGED_INPUT_TTL_SECONDS`.

GPU render tradeoff:
- `range_map` moves PDF rendering into GPU workers.
- This can increase GPU render time slightly, but usually reduces end-to-end latency and peak memory by removing large PNG round-trips.

## GPU Compatibility Note

If you override to `B200` and see `CUDA error: no kernel image is available for execution on the device`, use:

```bash
export TYPHOON_OCR_GPU=H100
```

This keeps high performance while using broader kernel compatibility.
