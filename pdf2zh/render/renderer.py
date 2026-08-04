from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

import fitz

from .background import RGB, prepare_cover, sample_text_color
from .compiler import _TYPST_LOCATION, TypstCompileError, compile_typst
from .config import RenderConfig
from .labels import skip_oversize_element
from .markup import (
    has_bare_latex,
    has_malformed_typst_math,
    has_unbalanced_math_tags,
    is_pure_math_text,
)
from .overlay import composite_overlay
from .sizing import assign_render_sizes
from .source_builder import build_typst_source

logger = logging.getLogger(__name__)

# Element markup definitions emitted by source_builder: #let e<p>_<i>_tm = [...],
# #let e<p>_<i>_c<n>_md = "...", etc. Used to map compile errors back to elements.
_ELEMENT_LET_RE = re.compile(r"#let (e\d+_\d+(?:_c\d+)?)_(?:tm|md|body|cover) = ")

# Safety cap for the compile-repair loop; each retry downgrades at least one
# new element. Large documents (100+ pages of dense math) can have more than a
# handful of independently-broken elements, so this stays generous — each
# retry is cheap (one more typst compile) next to failing the whole render.
_MAX_COMPILE_REPAIRS = 30


def _failing_element_vars(source: str, stderr: str) -> set[str]:
    """Map Typst error line numbers back to the element vars whose markup failed.

    For each ``…overlay.typ:LINE:COL`` in stderr, scan upward from LINE to the
    nearest ``#let e<p>_<i>_…`` definition — that element's markup contains the
    error. Errors outside any element definition are not attributed.
    """
    lines = source.splitlines()
    found: set[str] = set()
    for m in _TYPST_LOCATION.finditer(stderr):
        line_no = min(int(m.group(1)), len(lines))
        for idx in range(line_no - 1, -1, -1):
            let_m = _ELEMENT_LET_RE.match(lines[idx])
            if let_m:
                found.add(let_m.group(1))
                break
    return found


def render_document(
    pdf_path: str | Path,
    parsed: dict,
    output_path: str | Path,
    cfg: RenderConfig,
) -> dict:
    """Render translated PDF using Typst-based pipeline.

    Steps:
      1. Assign consistent font sizes (cluster per label group).
      2. Sample background and text colors per element from original PDF.
      3. Build Typst source with absolute-positioned cover rects + text blocks.
      4. Compile Typst → overlay PDF.
      5. Composite overlay onto original via show_pdf_page.
      6. Subset fonts + compress.

    Returns stats dict.
    """
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)

    stats = {
        "pages": 0,
        "elements_rendered": 0,
        "elements_skipped": 0,
        "cells_rendered": 0,
        "bg_samples": 0,
    }

    # 1. Assign sizes
    sizes = assign_render_sizes(parsed, cfg.sizing)
    logger.info("Sizing: %d size assignments", len(sizes))

    # 2. Sample colors
    bg_colors: dict[str, RGB] = {}
    text_colors: dict[str, RGB] = {}
    _sample_colors(pdf_path, parsed, cfg, sizes, bg_colors, text_colors, stats)

    # 3. Count rendered/skipped
    for page_idx, page in enumerate(parsed.get("pages", [])):
        if cfg.pages is not None and page.get("page_index", page_idx) not in cfg.pages:
            continue
        stats["pages"] += 1
        for elem_idx, elem in enumerate(page.get("elements", [])):
            category = elem.get("category", "")
            if category == "BYPASS":
                continue
            if category == "TABLE":
                for cell in elem.get("cells", []):
                    cell_source = cell.get("source_text") or ""
                    if cell_source.strip() and cell.get("translated_text"):
                        stats["cells_rendered"] += 1
                    else:
                        stats["elements_skipped"] += 1
            else:
                translated = elem.get("translated_text") or ""
                source = elem.get("source_text") or ""
                if translated and translated != source:
                    stats["elements_rendered"] += 1
                elif translated == source and category == "EQUATION":
                    stats["elements_skipped"] += 1
                elif translated:
                    stats["elements_rendered"] += 1
                else:
                    stats["elements_skipped"] += 1

    # 4. Build Typst source
    typst_source = build_typst_source(parsed, sizes, bg_colors, text_colors, cfg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        work_dir = Path(tmp_dir)

        if cfg.keep_typst_source:
            source_path = output_path.with_suffix(".typ")
            source_path.write_text(typst_source, encoding="utf-8")
            logger.info("Typst source saved to %s", source_path)

        overlay_pdf = work_dir / "overlay.pdf"

        # 5. Compile — self-healing: if an element's markup breaks the build,
        # rebuild with that element downgraded to plain text and retry.
        fallback_vars: set[str] = set()
        for attempt in range(_MAX_COMPILE_REPAIRS + 1):
            try:
                compile_typst(
                    typst_source,
                    font_paths=cfg.typst_font_paths,
                    output_pdf=overlay_pdf,
                    typst_bin=cfg.typst_binary,
                    work_dir=work_dir,
                )
                break
            except TypstCompileError as exc:
                bad_vars = (
                    _failing_element_vars(typst_source, exc.stderr) - fallback_vars
                )
                if not bad_vars or attempt == _MAX_COMPILE_REPAIRS:
                    raise
                fallback_vars |= bad_vars
                logger.warning(
                    "typst compile failed; retrying with plain-text fallback for: %s",
                    ", ".join(sorted(bad_vars)),
                )
                typst_source = build_typst_source(
                    parsed, sizes, bg_colors, text_colors, cfg, fallback_vars
                )
                if cfg.keep_typst_source:
                    output_path.with_suffix(".typ").write_text(
                        typst_source, encoding="utf-8"
                    )
        if fallback_vars:
            stats["elements_fallback"] = len(fallback_vars)

        # 6. Redact native text layer if present (non-scanned PDFs)
        base_pdf = pdf_path
        if cfg.redact_native_text and _has_native_text(pdf_path):
            redacted_pdf = work_dir / "redacted.pdf"
            _redact_text_layer(pdf_path, parsed, bg_colors, cfg, redacted_pdf)
            base_pdf = redacted_pdf
            logger.info("Native text layer redacted → %s", redacted_pdf)

        # 7. Composite
        output_path.parent.mkdir(parents=True, exist_ok=True)
        composite_overlay(base_pdf, overlay_pdf, output_path, cfg.pages)

    logger.info(
        "render_document done: pages=%d rendered=%d skipped=%d cells=%d",
        stats["pages"],
        stats["elements_rendered"],
        stats["elements_skipped"],
        stats["cells_rendered"],
    )
    return stats


def _has_native_text(pdf_path: Path, max_pages: int = 3) -> bool:
    doc = fitz.open(str(pdf_path))
    try:
        for i in range(min(max_pages, doc.page_count)):
            if doc[i].get_text("text").strip():
                return True
    finally:
        doc.close()
    return False


def _redact_text_layer(
    pdf_path: Path,
    parsed: dict,
    bg_colors: dict[str, RGB],
    cfg: RenderConfig,
    out_path: Path,
) -> None:
    """Erase translatable elements from the original text layer via redaction."""
    doc = fitz.open(str(pdf_path))
    pad = cfg.background.eraser_padding_pt
    try:
        for page_idx, page_data in enumerate(parsed.get("pages", [])):
            orig = page_data.get("page_index", page_idx)
            if cfg.pages is not None and orig not in cfg.pages:
                continue
            if orig >= doc.page_count:
                continue
            page = doc[orig]
            pw = page_data.get("page_width", page.rect.width)
            ph = page_data.get("page_height", page.rect.height)
            had_annot = False

            for elem_idx, elem in enumerate(page_data.get("elements", [])):
                category = elem.get("category", "")
                if category == "BYPASS":
                    continue
                uid = f"p{page_idx}:e{elem_idx}"

                # Mirror the overlay's skip rule exactly: whatever the overlay
                # will not redraw must not be redacted here, or the original is
                # erased with nothing put back. Minor/structural elements that
                # span most of the page are mis-detections — keep the original.
                if skip_oversize_element(
                    elem.get("label", "Text"),
                    elem.get("bbox_pdf", [0, 0, 10, 10]),
                    pw,
                    ph,
                ):
                    continue

                if category == "TABLE":
                    for cell_idx, cell in enumerate(elem.get("cells", [])):
                        cell_source = cell.get("source_text") or ""
                        if not cell_source.strip() or not cell.get("translated_text"):
                            continue
                        cell_uid = f"{uid}:c{cell_idx}"
                        # Strip native text only over bbox_text (tight box), not
                        # the whole grid cell — mirrors the overlay's cover_rect
                        # and keeps the cell's borders/background intact.
                        cx0, cy0, cx1, cy1 = cell.get("bbox_text") or cell.get(
                            "bbox_pdf", elem.get("bbox_pdf", [0, 0, 10, 10])
                        )
                        fill = bg_colors.get(cell_uid, (255, 255, 255))
                        page.add_redact_annot(
                            fitz.Rect(cx0 - pad, cy0 - pad, cx1 + pad, cy1 + pad),
                            fill=[c / 255 for c in fill],
                        )
                        had_annot = True
                else:
                    translated = elem.get("translated_text") or ""
                    source = elem.get("source_text") or ""
                    if not translated:
                        continue
                    if translated.strip() == source.strip():
                        continue
                    if (
                        is_pure_math_text(translated)
                        or has_unbalanced_math_tags(translated)
                        or has_bare_latex(translated)
                        or has_malformed_typst_math(translated)
                    ):
                        continue
                    x0, y0, x1, y1 = elem.get("bbox_pdf", [0, 0, 10, 10])
                    fill = bg_colors.get(uid, (255, 255, 255))
                    page.add_redact_annot(
                        fitz.Rect(x0 - pad, y0 - pad, x1 + pad, y1 + pad),
                        fill=[c / 255 for c in fill],
                    )
                    had_annot = True

            if had_annot:
                # IMAGE_NONE + LINE_ART_NONE: only strip text layer, don't rasterize
                page.apply_redactions(
                    images=fitz.PDF_REDACT_IMAGE_NONE,
                    graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                )

        doc.save(str(out_path), garbage=3, deflate=True)
    finally:
        doc.close()


def _sample_colors(
    pdf_path: Path,
    parsed: dict,
    cfg: RenderConfig,
    sizes: dict[str, float],
    bg_colors: dict[str, RGB],
    text_colors: dict[str, RGB],
    stats: dict,
) -> None:
    if not cfg.background.enabled and not cfg.text_color.enabled:
        return

    doc = fitz.open(str(pdf_path))
    try:
        for page_idx, page_data in enumerate(parsed.get("pages", [])):
            orig = page_data.get("page_index", page_idx)
            if cfg.pages is not None and orig not in cfg.pages:
                continue
            if orig >= doc.page_count:
                continue
            page = doc[orig]
            pw = page_data.get("page_width", page.rect.width)
            ph = page_data.get("page_height", page.rect.height)

            for elem_idx, elem in enumerate(page_data.get("elements", [])):
                category = elem.get("category", "")
                if category == "BYPASS":
                    continue

                uid = f"p{page_idx}:e{elem_idx}"
                bbox = elem.get("bbox_pdf", [0, 0, 10, 10])

                if category == "TABLE":
                    for cell_idx, cell in enumerate(elem.get("cells", [])):
                        cell_source = cell.get("source_text") or ""
                        if not cell_source.strip():
                            continue
                        cell_uid = f"{uid}:c{cell_idx}"
                        # renderer.py:270 — dùng bbox_text (đồng bộ với render/redact), fallback về bbox_pdf
                        cbbox = cell.get("bbox_text") or cell.get("bbox_pdf", bbox)
                        bg = prepare_cover(page, cbbox, pw, ph, cfg.background)
                        bg_colors[cell_uid] = bg.rgb
                        tc = sample_text_color(
                            page, cbbox, pw, ph, bg.rgb, cfg.text_color
                        )
                        text_colors[cell_uid] = tc
                        stats["bg_samples"] += 1
                else:
                    # User-added boxes may carry explicit color overrides
                    # (review.add_element); honor them instead of sampling.
                    ov_bg = elem.get("bg_color")
                    ov_tc = elem.get("text_color")
                    if ov_bg:
                        bg_rgb = tuple(ov_bg)
                    else:
                        bg_rgb = prepare_cover(page, bbox, pw, ph, cfg.background).rgb
                    bg_colors[uid] = bg_rgb
                    if ov_tc:
                        text_colors[uid] = tuple(ov_tc)
                    else:
                        text_colors[uid] = sample_text_color(
                            page, bbox, pw, ph, bg_rgb, cfg.text_color
                        )
                    stats["bg_samples"] += 1
    finally:
        doc.close()
