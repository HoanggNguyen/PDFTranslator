"""End-to-end orchestration: OCR (Phase 1) -> Translate (Phase 2) -> Render (Phase 3).

This module wires the three existing phases into a single callable used by the
Gradio app (``app.py``). It reuses the public APIs of each phase and adds:
  - a process-wide lazy singleton for ``StageAParser`` (its 3-5GB models load once),
  - language-name + font handling shared across the run,
  - intermediate JSON artifacts written to a per-request work dir.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Callable, Optional

from pdf2zh.render import RenderConfig, render_document
from pdf2zh.scanned import PDFTypeDetector, StageAParser
from pdf2zh.translation import TranslatorConfig, translate_document

logger = logging.getLogger(__name__)

# Full language names — Phase 2 prompts interpolate these directly (prompts.py).
SUPPORTED_LANGUAGES = [
    "English",
    "Vietnamese",
    "Simplified Chinese",
    "Japanese",
    "Korean",
    "French",
    "German",
    "Spanish",
]

# Directories searched by Typst for fonts (populated in the Docker image).
FONT_DIRS = [os.environ.get("PDF2ZH_FONT_DIR", "/app/fonts")]

# Fonts pre-installed in the image (see Dockerfile). The UI exposes these.
# Family names must match what Typst sees (apt fonts-noto-* + bundled Be Vietnam Pro).
BUNDLED_FONTS = ["Noto Sans", "Noto Serif", "Be Vietnam Pro", "Noto Sans CJK SC"]
DEFAULT_FONT = "Noto Sans"  # neutral, full Vietnamese coverage
# Appended after the user's choice so missing glyphs fall back gracefully.
FALLBACK_TAIL = ["Noto Sans", "Noto Serif", "Noto Sans CJK SC"]


def font_chain(selected: str) -> list[str]:
    """User-selected font first, then multilingual fallbacks (deduped, ordered)."""
    chain = [selected, *FALLBACK_TAIL]
    return list(dict.fromkeys(c for c in chain if c))


# --------------------------------------------------------------------------- #
# Phase-1 model singleton
# --------------------------------------------------------------------------- #
_parser: Optional[StageAParser] = None


def get_parser() -> StageAParser:
    """Process-wide lazy singleton. The Surya/Paddle models load exactly once."""
    global _parser
    if _parser is None:
        logger.info("Loading StageAParser models (one-time)...")
        _parser = StageAParser(device="auto", enable_latex=False)
        logger.info("StageAParser ready.")
    return _parser


def warmup() -> None:
    """Load models at app startup so the first request isn't penalized."""
    get_parser()


# --------------------------------------------------------------------------- #
# Config builders
# --------------------------------------------------------------------------- #
def build_translator_config(
    src_lang: str,
    tgt_lang: str,
    provider: str,
    api_key: str,
    model: str | None,
) -> TranslatorConfig:
    """Build Phase-2 config. Languages are set on the config directly (the
    pipeline reads ``cfg.source_language`` before the doc dict), and the API key
    is passed through so ``resolve_provider`` never needs an env var."""
    return TranslatorConfig(
        source_language=src_lang,
        target_language=tgt_lang,
        provider=provider,
        model=(model.strip() or None) if model else None,
        api_key=api_key.strip(),
    )


def build_render_config(font: str, pages: list[int] | None) -> RenderConfig:
    """Build Phase-3 config. The chosen font heads a fallback chain; the default
    Helvetica lacks Vietnamese glyphs so we always override it."""
    cfg = RenderConfig()
    cfg.font_family = font_chain(font)
    cfg.typst_font_paths = FONT_DIRS
    cfg.typst_binary = os.environ.get("TYPST_BIN", "typst")
    cfg.pages = pages
    cfg.redact_native_text = True
    cfg.min_font_size_pt = 7.0
    return cfg


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_pipeline(
    pdf_path: str,
    src_lang: str,
    tgt_lang: str,
    provider: str,
    api_key: str,
    model: str | None,
    pages: list[int] | None,
    font: str,
    work_dir: str | Path,
    progress: Callable[[float, str], None] | None = None,
) -> str:
    """Run Phase 1 -> 2 -> 3 and return the path to the translated PDF.

    ``pages`` is a 0-based index list (or None for all) shared by Phase 1 and 3.
    ``progress(frac, msg)`` is an optional UI callback.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    def _p(frac: float, msg: str) -> None:
        logger.info(msg)
        if progress is not None:
            progress(frac, msg)

    # 0. Fail fast before any GPU work.
    if not pdf_path:
        raise ValueError("Vui lòng tải lên một file PDF.")
    if not api_key or not api_key.strip():
        raise ValueError("Thiếu API key — nhập API key của provider ở thanh bên.")
    if not src_lang or not tgt_lang:
        raise ValueError("Chọn ngôn ngữ nguồn và ngôn ngữ đích.")

    # 1. Detect type (informational only — the Surya path handles all types).
    _p(0.02, "Đang nhận diện loại PDF...")
    try:
        pdf_type = PDFTypeDetector().detect(pdf_path)
        logger.info("PDF type: %s", pdf_type)
    except Exception as exc:  # detection is best-effort, never fatal
        logger.warning("PDF type detection failed: %s", exc)

    # 2. Phase 1 — OCR / layout parse (slowest step).
    _p(0.05, "Phase 1/3 — OCR & phân tích bố cục (bước chậm nhất)...")
    parser = get_parser()
    parsed_doc = parser.parse_pdf(pdf_path, cache_path=None, pages=pages)
    parsed_dict = parsed_doc.to_dict()
    (work / "phase1_parsed.json").write_text(parsed_doc.to_json(), encoding="utf-8")

    # 3. Phase 2 — translate (synchronous; runs asyncio internally).
    _p(0.55, f"Phase 2/3 — Đang dịch {src_lang} → {tgt_lang}...")
    tcfg = build_translator_config(src_lang, tgt_lang, provider, api_key, model)
    translated_dict = translate_document(parsed_dict, tcfg)
    (work / "phase2_translated.json").write_text(
        json.dumps(translated_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 4. Phase 3 — render translated PDF (shells out to typst).
    _p(0.85, "Phase 3/3 — Đang dựng PDF bản dịch (typst)...")
    out_path = str(work / f"translated_{uuid.uuid4().hex[:8]}.pdf")
    rcfg = build_render_config(font, pages)
    render_document(pdf_path, translated_dict, out_path, rcfg)

    _p(1.0, "Hoàn tất.")
    return out_path
