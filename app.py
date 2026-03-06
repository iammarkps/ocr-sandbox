import io
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import modal

if TYPE_CHECKING:
    from PIL import Image as PILImage

app = modal.App("typhoon-ocr")

MODEL_ID = "scb10x/typhoon-ocr1.5-2b"
MODEL_DIR = "/models/typhoon-ocr1.5-2b"
PDF_POINTS_PER_INCH = 72
MAX_ALLOWED_CONTAINERS = 5


class OCRError(Exception):
    """Base class for OCR failures."""


class ConfigurationError(OCRError, RuntimeError):
    """Raised when environment configuration is invalid."""


class ValidationError(OCRError, ValueError):
    """Raised when user input fails validation."""


class PDFProcessingError(OCRError, RuntimeError):
    """Raised when PDF processing fails."""


def _parse_int_env(name: str, default: int, *, min_value: int = 1, max_value: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got: {raw!r}") from exc
    if value < min_value:
        raise ConfigurationError(f"{name} must be >= {min_value}, got: {value}")
    if max_value is not None and value > max_value:
        raise ConfigurationError(f"{name} must be <= {max_value}, got: {value}")
    return value


def _parse_str_env(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def _parse_choice_env(name: str, default: str, allowed: set[str]) -> str:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip()
    if value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ConfigurationError(f"{name} must be one of: {allowed_values}. Got: {value!r}")
    return value


@dataclass(frozen=True)
class Config:
    gpu: str = "A10"
    max_containers: int = MAX_ALLOWED_CONTAINERS
    min_containers: int = 1
    buffer_containers: int = 1
    scaledown_window: int = 1200
    max_new_tokens: int = 10000
    max_file_mb: int = 200
    max_pdf_pages: int = 200
    pdf_dpi: int = 150
    page_batch_size: int = 4
    pdf_pipeline: str = "range_map"
    max_image_side: int = 1800
    max_image_pixels: int = 80_000_000
    staged_input_ttl_seconds: int = 86_400
    model_revision: str = "be9399b"

    @classmethod
    def from_env(cls) -> "Config":
        config = cls(
            gpu=_parse_str_env("TYPHOON_OCR_GPU", cls.gpu),
            max_containers=_parse_int_env(
                "TYPHOON_OCR_MAX_CONTAINERS",
                cls.max_containers,
                min_value=1,
                max_value=MAX_ALLOWED_CONTAINERS,
            ),
            min_containers=_parse_int_env(
                "TYPHOON_OCR_MIN_CONTAINERS",
                cls.min_containers,
                min_value=0,
                max_value=MAX_ALLOWED_CONTAINERS,
            ),
            buffer_containers=_parse_int_env(
                "TYPHOON_OCR_BUFFER_CONTAINERS",
                cls.buffer_containers,
                min_value=0,
                max_value=MAX_ALLOWED_CONTAINERS,
            ),
            scaledown_window=_parse_int_env("TYPHOON_OCR_SCALEDOWN_WINDOW", cls.scaledown_window, min_value=60),
            max_new_tokens=_parse_int_env("TYPHOON_OCR_MAX_NEW_TOKENS", cls.max_new_tokens, min_value=1),
            max_file_mb=_parse_int_env("TYPHOON_OCR_MAX_FILE_MB", cls.max_file_mb, min_value=1),
            max_pdf_pages=_parse_int_env("TYPHOON_OCR_MAX_PDF_PAGES", cls.max_pdf_pages, min_value=1),
            pdf_dpi=_parse_int_env("TYPHOON_OCR_PDF_DPI", cls.pdf_dpi, min_value=PDF_POINTS_PER_INCH),
            page_batch_size=_parse_int_env("TYPHOON_OCR_PAGE_BATCH_SIZE", cls.page_batch_size, min_value=1, max_value=32),
            pdf_pipeline=_parse_choice_env(
                "TYPHOON_OCR_PDF_PIPELINE",
                cls.pdf_pipeline,
                {"legacy", "range_map"},
            ),
            max_image_side=_parse_int_env("TYPHOON_OCR_MAX_IMAGE_SIDE", cls.max_image_side, min_value=256),
            max_image_pixels=_parse_int_env(
                "TYPHOON_OCR_MAX_IMAGE_PIXELS",
                cls.max_image_pixels,
                min_value=1_000_000,
            ),
            staged_input_ttl_seconds=_parse_int_env(
                "TYPHOON_OCR_STAGED_INPUT_TTL_SECONDS",
                cls.staged_input_ttl_seconds,
                min_value=3_600,
            ),
            model_revision=_parse_str_env("TYPHOON_OCR_MODEL_REVISION", cls.model_revision),
        )

        if config.min_containers > config.max_containers:
            raise ConfigurationError(
                f"TYPHOON_OCR_MIN_CONTAINERS ({config.min_containers}) must be <= "
                f"TYPHOON_OCR_MAX_CONTAINERS ({config.max_containers})."
            )
        if config.buffer_containers > config.max_containers:
            raise ConfigurationError(
                f"TYPHOON_OCR_BUFFER_CONTAINERS ({config.buffer_containers}) must be <= "
                f"TYPHOON_OCR_MAX_CONTAINERS ({config.max_containers})."
            )

        return config


CONFIG = Config.from_env()

MODEL_REVISION = CONFIG.model_revision
GPU = CONFIG.gpu
MAX_CONTAINERS = CONFIG.max_containers
MIN_CONTAINERS = CONFIG.min_containers
BUFFER_CONTAINERS = CONFIG.buffer_containers
SCALEDOWN_WINDOW = CONFIG.scaledown_window
MAX_NEW_TOKENS = CONFIG.max_new_tokens
MAX_FILE_MB = CONFIG.max_file_mb
MAX_PDF_PAGES = CONFIG.max_pdf_pages
PDF_DPI = CONFIG.pdf_dpi
PAGE_BATCH_SIZE = CONFIG.page_batch_size
PDF_PIPELINE = CONFIG.pdf_pipeline
MAX_IMAGE_SIDE = CONFIG.max_image_side
MAX_IMAGE_PIXELS = CONFIG.max_image_pixels
STAGED_INPUT_TTL_SECONDS = CONFIG.staged_input_ttl_seconds

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
REQUIRED_MODEL_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")

PROMPT = """Extract all text from the image.

Instructions:
- Only return the clean Markdown.
- Do not include any explanation or extra text.
- You must include all information on the page.

Formatting Rules:
- Tables: Render using <table>...</table> HTML
- Equations: Use LaTeX ($...$ inline, $$...$$ block)
- Images/Charts: Wrap in <figure>...</figure>
- Page Numbers: Wrap in <page_number>...</page_number>
- Checkboxes: ☐ unchecked, ☑ checked"""

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject(
        "pyproject.toml",
        # Blackwell GPUs (for example B200) need CUDA 12.8+ PyTorch wheels.
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
)

volume = modal.Volume.from_name("typhoon-ocr-models", create_if_missing=True)
input_volume = modal.Volume.from_name("typhoon-ocr-inputs", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


def _model_dir_integrity_status(model_dir: str | Path) -> tuple[bool, str]:
    model_path = Path(model_dir)
    if not model_path.exists():
        return False, "missing model directory"

    missing = [name for name in REQUIRED_MODEL_FILES if not (model_path / name).is_file()]
    if missing:
        return False, f"missing required file(s): {', '.join(missing)}"

    has_weights = any(model_path.glob("*.safetensors")) or any(model_path.glob("*.bin"))
    if not has_weights:
        return False, "missing model weight shard (*.safetensors or *.bin)"

    return True, "ok"


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _validate_local_input(path: Path) -> tuple[str, int]:
    if not path.exists():
        raise ValidationError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValidationError(f"Unsupported file type {suffix!r}. Supported extensions: {allowed}")

    file_size = path.stat().st_size
    max_bytes = MAX_FILE_MB * 1024 * 1024
    if file_size > max_bytes:
        raise ValidationError(
            f"Input file is too large ({_format_bytes(file_size)}). Max allowed is {MAX_FILE_MB} MB."
        )

    return suffix, file_size


def _validate_output_path(out_path: Path, overwrite: bool) -> None:
    if out_path.exists() and not overwrite:
        raise ValidationError(
            f"Output file already exists: {out_path}. "
            "Pass --overwrite=true or use --output <new_path>."
        )


def _chunk_items(items: list[bytes], chunk_size: int) -> list[list[bytes]]:
    if chunk_size <= 0:
        raise ValidationError("chunk_size must be > 0")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _page_ranges(page_count: int, range_size: int) -> list[tuple[int, int]]:
    if range_size <= 0:
        raise ValidationError("range_size must be > 0")
    if page_count <= 0:
        return []
    return [(start, min(start + range_size, page_count)) for start in range(0, page_count, range_size)]


def _iter_page_blocks_for_range(start_page: int, end_page: int, range_results: list[str]):
    expected_pages = end_page - start_page
    if len(range_results) != expected_pages:
        raise PDFProcessingError(
            f"OCR range result length mismatch for pages {start_page + 1}-{end_page}: "
            f"expected {expected_pages}, got {len(range_results)}."
        )

    for local_index, text in enumerate(range_results):
        page_num = start_page + local_index + 1
        yield page_num, f"<!-- Page {page_num} -->\n{text.rstrip()}"


def _is_stale_run_dir(run_dir: Path, now: float, ttl_seconds: int) -> bool:
    if not run_dir.is_dir():
        return False
    try:
        age_seconds = now - run_dir.stat().st_mtime
    except FileNotFoundError:
        return False
    return age_seconds >= ttl_seconds


def _cleanup_stale_input_runs(inputs_root: Path, now: float, ttl_seconds: int) -> int:
    import shutil

    if not inputs_root.exists():
        return 0

    cleaned = 0
    for run_dir in inputs_root.iterdir():
        if not _is_stale_run_dir(run_dir, now, ttl_seconds):
            continue
        try:
            shutil.rmtree(run_dir)
        except OSError as exc:
            print(f"Warning: failed to clean stale staged input {run_dir}: {exc}")
            continue
        cleaned += 1
    return cleaned


@contextmanager
def open_pdf(source: bytes | str | Path) -> Iterator[Any]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(source)
    try:
        yield pdf
    finally:
        close = getattr(pdf, "close", None)
        if callable(close):
            close()


@app.function(image=image, volumes={"/models": volume}, secrets=[hf_secret], timeout=1200)
def download_model():
    """Download model weights to the Volume. Run once: uv run modal run app.py::download_model"""
    import shutil

    print(f"Preparing model download for {MODEL_ID} @ {MODEL_REVISION}")

    valid, reason = _model_dir_integrity_status(MODEL_DIR)
    if valid:
        print(f"Model already present at {MODEL_DIR} ({MODEL_REVISION}), skipping download.")
        return

    model_path = Path(MODEL_DIR)
    if model_path.exists():
        print(f"Model directory exists but is invalid ({reason}). Re-downloading.")
        shutil.rmtree(model_path)

    from huggingface_hub import snapshot_download

    print(f"Downloading {MODEL_ID} @ {MODEL_REVISION}...")
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=MODEL_DIR)

    valid, reason = _model_dir_integrity_status(MODEL_DIR)
    if not valid:
        raise OCRError(f"Downloaded model failed integrity checks: {reason}")

    volume.commit()
    print("Done. Model saved to volume.")


@app.cls(
    image=image,
    gpu=GPU,
    volumes={"/models": volume, "/inputs": input_volume},
    timeout=300,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    buffer_containers=BUFFER_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    enable_memory_snapshot=True,
)
class TyphoonOCR:
    @modal.enter(snap=True)
    def load_model(self):
        valid, reason = _model_dir_integrity_status(MODEL_DIR)
        if not valid:
            raise OCRError(
                f"Model not ready ({reason}). Run 'uv run modal run app.py::download_model' first."
            )

        print(f"Loading model {MODEL_ID} @ {MODEL_REVISION}")

        from transformers import (
            AutoImageProcessor,
            AutoModelForImageTextToText,
            AutoTokenizer,
            Qwen3VLProcessor,
            Qwen3VLVideoProcessor,
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_DIR, dtype="auto", device_map="auto"
        )
        self.model.eval()

        # Build processor manually — AutoProcessor.from_pretrained() crashes
        # because AutoVideoProcessor mapping is None in this transformers version.
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        image_processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
        video_processor = Qwen3VLVideoProcessor.from_pretrained(MODEL_DIR)
        self.processor = Qwen3VLProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=tokenizer.chat_template,
        )
        print("Model loaded.")

    def _normalize_image(self, img: "PILImage.Image") -> "PILImage.Image":
        from PIL import Image

        if img.mode != "RGB":
            img = img.convert("RGB")

        width, height = img.size
        if width <= 0 or height <= 0:
            raise ValidationError("Invalid image dimensions.")

        if width * height > MAX_IMAGE_PIXELS:
            raise ValidationError(
                f"Image too large ({width}x{height}). Max allowed pixels: {MAX_IMAGE_PIXELS}."
            )

        if width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE:
            if width >= height:
                scale = MAX_IMAGE_SIDE / float(width)
                new_size = (MAX_IMAGE_SIDE, max(1, int(height * scale)))
            else:
                scale = MAX_IMAGE_SIDE / float(height)
                new_size = (max(1, int(width * scale)), MAX_IMAGE_SIDE)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        return img

    def _ocr_pil_image(self, img: "PILImage.Image") -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids, strict=False)]

        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    @modal.method()
    def run_page(self, image_bytes: bytes) -> str:
        """OCR a standalone image."""
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as img:
            img.load()
            return self._ocr_pil_image(self._normalize_image(img))

    @modal.method()
    def run_page_batch(self, page_batch: list[bytes]) -> list[str]:
        """OCR a batch of pages in one remote call to reduce RPC overhead."""
        from PIL import Image

        images = []
        for image_bytes in page_batch:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img.load()
                images.append(self._normalize_image(img))

        return [self._ocr_pil_image(img) for img in images]

    @modal.method()
    def run_pdf_range(self, staged_pdf_path: str, start_page: int, end_page: int, dpi: int = PDF_DPI) -> list[str]:
        input_volume.reload()

        with open_pdf(staged_pdf_path) as pdf:
            page_count = len(pdf)
            if start_page < 0 or end_page > page_count or start_page >= end_page:
                raise PDFProcessingError(
                    f"Invalid page range [{start_page}, {end_page}) for PDF with {page_count} page(s)."
                )

            scale = dpi / PDF_POINTS_PER_INCH
            results: list[str] = []
            for page_index in range(start_page, end_page):
                page = pdf[page_index]
                bitmap = page.render(scale=scale)
                img = bitmap.to_pil()
                try:
                    results.append(self._ocr_pil_image(self._normalize_image(img)))
                finally:
                    img.close()
            return results


@app.function(image=image, volumes={"/inputs": input_volume})
def stage_pdf_input(pdf_bytes: bytes, run_id: str) -> tuple[str, int]:
    now = time.time()
    stale_runs_deleted = _cleanup_stale_input_runs(Path("/inputs"), now, STAGED_INPUT_TTL_SECONDS)
    if stale_runs_deleted:
        print(f"Cleaned up {stale_runs_deleted} stale staged PDF run(s).")

    run_dir = Path("/inputs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    staged_path = run_dir / "input.pdf"
    staged_path.write_bytes(pdf_bytes)

    with open_pdf(str(staged_path)) as pdf:
        page_count = len(pdf)

    if page_count == 0:
        raise PDFProcessingError("PDF has no pages.")
    if page_count > MAX_PDF_PAGES:
        raise PDFProcessingError(
            f"PDF has {page_count} pages, which exceeds limit of {MAX_PDF_PAGES}."
        )

    input_volume.commit()
    return str(staged_path), page_count


@app.function(image=image, volumes={"/inputs": input_volume})
def cleanup_staged_pdf(run_id: str) -> None:
    import shutil

    run_dir = Path("/inputs") / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    input_volume.commit()


@app.function(image=image)
def pdf_to_page_images(pdf_bytes: bytes, dpi: int = PDF_DPI) -> list[bytes]:
    """Convert PDF pages to PNG bytes inside Modal."""
    with open_pdf(pdf_bytes) as pdf:
        page_count = len(pdf)

        if page_count == 0:
            raise PDFProcessingError("PDF has no pages.")
        if page_count > MAX_PDF_PAGES:
            raise PDFProcessingError(
                f"PDF has {page_count} pages, which exceeds limit of {MAX_PDF_PAGES}."
            )

        scale = dpi / PDF_POINTS_PER_INCH
        pages: list[bytes] = []
        for i in range(page_count):
            page = pdf[i]
            bitmap = page.render(scale=scale)
            img = bitmap.to_pil()
            try:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                pages.append(buf.getvalue())
            finally:
                img.close()
        return pages


def _print_preflight(path: Path, suffix: str, file_size: int) -> None:
    print("Preflight")
    print(f"- Input: {path}")
    print(f"- Type: {suffix}")
    print(f"- Size: {_format_bytes(file_size)}")
    print(f"- GPU: {GPU}")
    print(f"- Min containers: {MIN_CONTAINERS}")
    print(f"- Buffer containers: {BUFFER_CONTAINERS}")
    print(f"- Max containers: {MAX_CONTAINERS}")
    print(f"- Scaledown window (s): {SCALEDOWN_WINDOW}")
    print(f"- Max new tokens: {MAX_NEW_TOKENS}")
    print(f"- Max PDF pages: {MAX_PDF_PAGES}")
    print(f"- PDF DPI: {PDF_DPI}")
    print(f"- PDF page batch size: {PAGE_BATCH_SIZE}")
    print(f"- PDF pipeline: {PDF_PIPELINE}")
    print(f"- Staged input TTL (s): {STAGED_INPUT_TTL_SECONDS}")


def _run_image(path: Path, file_bytes: bytes, ocr: TyphoonOCR) -> str:
    print(f"Running OCR on image: {path}")
    return ocr.run_page.remote(file_bytes)


def _run_pdf_legacy(file_bytes: bytes, ocr: TyphoonOCR) -> str:
    print("Rendering PDF pages in Modal (legacy pipeline)...")
    page_images = pdf_to_page_images.remote(file_bytes, PDF_DPI)
    page_batches = _chunk_items(page_images, PAGE_BATCH_SIZE)
    print(
        f"Running OCR on {len(page_images)} page(s) in {len(page_batches)} batch(es) "
        f"of up to {PAGE_BATCH_SIZE} page(s) with up to {MAX_CONTAINERS} GPU container(s)..."
    )
    batched_results = list(ocr.run_page_batch.map(page_batches, order_outputs=True))
    results = [text for batch in batched_results for text in batch]
    return "\n\n---\n\n".join(f"<!-- Page {i + 1} -->\n{text}" for i, text in enumerate(results))


def _run_pdf_range_map(file_bytes: bytes, out_path: Path, ocr: TyphoonOCR) -> None:
    run_id = uuid.uuid4().hex
    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    started_at = time.perf_counter()
    staged_pdf_path: str | None = None
    page_count = 0
    processed_ranges = 0
    stream_exhausted = False
    wrote_page = False

    print("Staging PDF in Modal input volume...")
    print("\n--- OCR Result ---")

    try:
        staged_pdf_path, page_count = stage_pdf_input.remote(file_bytes, run_id)
        ranges = _page_ranges(page_count, PAGE_BATCH_SIZE)
        print(
            f"Running OCR on {page_count} page(s) in {len(ranges)} range batch(es) "
            f"of up to {PAGE_BATCH_SIZE} page(s) with up to {MAX_CONTAINERS} GPU container(s)..."
        )
        print("Note: ordered streaming may wait for earlier ranges to finish.")

        previous_range_time = time.perf_counter()
        range_args = ((staged_pdf_path, start_page, end_page, PDF_DPI) for start_page, end_page in ranges)

        with temp_path.open("w", encoding="utf-8") as tmp_file:
            for range_idx, ((start_page, end_page), range_result) in enumerate(
                zip(ranges, ocr.run_pdf_range.starmap(range_args, order_outputs=True), strict=True), start=1
            ):
                now = time.perf_counter()
                range_elapsed = now - previous_range_time
                previous_range_time = now
                print(
                    f"[Range {range_idx}/{len(ranges)}] pages {start_page + 1}-{end_page} "
                    f"completed in {range_elapsed:.2f}s"
                )
                processed_ranges = range_idx

                for _, page_block in _iter_page_blocks_for_range(start_page, end_page, range_result):
                    if wrote_page:
                        tmp_file.write("\n\n---\n\n")
                        print("\n---\n")
                    tmp_file.write(page_block)
                    print(page_block)
                    wrote_page = True
                tmp_file.flush()

            if wrote_page:
                tmp_file.write("\n")
            stream_exhausted = True

        if processed_ranges != len(ranges):
            raise PDFProcessingError(
                f"OCR returned incomplete range results: expected {len(ranges)}, got {processed_ranges}."
            )

        temp_path.replace(out_path)
        total_elapsed = time.perf_counter() - started_at
        print(f"\nSaved to {out_path}")
        print(f"Processed {page_count} page(s) in {total_elapsed:.2f}s.")
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    finally:
        if stream_exhausted and staged_pdf_path is not None:
            try:
                cleanup_staged_pdf.remote(run_id)
            except Exception as cleanup_exc:
                print(f"Warning: cleanup failed: {cleanup_exc}")
        else:
            print(
                f"Deferred staged PDF cleanup for run_id={run_id} due to incomplete stream consumption."
            )


def _run_pdf(file_bytes: bytes, out_path: Path, ocr: TyphoonOCR) -> str | None:
    if PDF_PIPELINE == "legacy":
        return _run_pdf_legacy(file_bytes, ocr)
    _run_pdf_range_map(file_bytes, out_path, ocr)
    return None


@app.local_entrypoint()
def main(file_path: str = "input.pdf", output: str = "", overwrite: bool = False):
    path = Path(file_path)

    try:
        suffix, file_size = _validate_local_input(path)
        out_path = Path(output) if output else path.with_suffix(".md")
        _validate_output_path(out_path, overwrite)
    except ValidationError as exc:
        print(exc)
        return

    _print_preflight(path, suffix, file_size)

    file_bytes = path.read_bytes()
    ocr = TyphoonOCR()

    if suffix == ".pdf":
        output_text = _run_pdf(file_bytes, out_path, ocr)
        if output_text is None:
            return
    else:
        output_text = _run_image(path, file_bytes, ocr)

    normalized_output = output_text.rstrip() + "\n"

    print("\n--- OCR Result ---")
    print(normalized_output)

    out_path.write_text(normalized_output, encoding="utf-8")
    print(f"\nSaved to {out_path}")
