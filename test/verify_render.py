"""Phase-3 render smoke + feasibility test.

Runs in two modes:

1. Pre-flight (default) — validates the plan's assumptions against the real PDF
   + parsed JSON without needing pdf2zh.render to exist yet:
     * inputs load, schema is correct
     * PyMuPDF + numpy can access pixmap samples
     * detect_bg_color / detect_text_color produce sensible values on a real bbox
     * font registration + text insertion work on a synthetic page
     * stats: overflow risk, native-text presence, redaction necessity

2. Render (--render --font ... --output ...) — once pdf2zh.render is built,
   runs the full pipeline and verifies the output PDF is non-empty and the
   redacted+rendered text is extractable.

Usage:
    python test/verify_render.py \
        --input GK_ChuanMucKeToan_Nhom02.pdf \
        --parsed output-3.translated.json
    python test/verify_render.py \
        --input GK_ChuanMucKeToan_Nhom02.pdf \
        --parsed output-3.translated.json \
        --render --font fonts/NotoSans-Regular.ttf --output GK.rendered.pdf
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

# Ensure project root is on sys.path when script is run directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz
import numpy as np

# ---------------------------------------------------------------------------
# Pure helpers — duplicated here so the script runs without pdf2zh.render.
# Once color.py exists, the renderer should produce identical results.
# ---------------------------------------------------------------------------


def _pixmap_to_array(pm: fitz.Pixmap) -> np.ndarray:
    arr = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, pm.n)
    if pm.n == 4:
        arr = arr[:, :, :3]
    return arr


def _bbox_to_pixels(bbox, pw, ph, pm):
    sx = pm.width / pw
    sy = pm.height / ph
    x0, y0, x1, y1 = bbox
    px0 = max(0, int(round(x0 * sx)))
    py0 = max(0, int(round(y0 * sy)))
    px1 = min(pm.width, int(round(x1 * sx)))
    py1 = min(pm.height, int(round(y1 * sy)))
    return px0, py0, px1, py1


def detect_bg_color(arr, bbox_px, edge=2, qstep=16):
    px0, py0, px1, py1 = bbox_px
    if px1 - px0 < 2 * edge + 1 or py1 - py0 < 2 * edge + 1:
        return (255, 255, 255)
    top = arr[py0 : py0 + edge, px0:px1]
    bot = arr[py1 - edge : py1, px0:px1]
    left = arr[py0 + edge : py1 - edge, px0 : px0 + edge]
    right = arr[py0 + edge : py1 - edge, px1 - edge : px1]
    band = np.concatenate(
        [
            top.reshape(-1, 3),
            bot.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3),
        ]
    )
    if band.size == 0:
        return (255, 255, 255)
    q = (band // qstep) * qstep + qstep // 2
    keys = q[:, 0].astype(np.int32) * 65536 + q[:, 1].astype(np.int32) * 256 + q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    winner = vals[counts.argmax()]
    return (int((winner >> 16) & 0xFF), int((winner >> 8) & 0xFF), int(winner & 0xFF))


def detect_text_color(arr, bbox_px, bg, edge=2, qstep=16, dist=32, min_ratio=0.05):
    px0, py0, px1, py1 = bbox_px
    inner = arr[py0 + edge : py1 - edge, px0 + edge : px1 - edge]
    if inner.size == 0:
        return (0, 0, 0)
    flat = inner.reshape(-1, 3).astype(np.int32)
    bg_arr = np.array(bg, dtype=np.int32)
    d = np.sqrt(((flat - bg_arr) ** 2).sum(axis=1))
    keep = flat[d > dist]
    if keep.size == 0:
        return (0, 0, 0)
    q = (keep // qstep) * qstep + qstep // 2
    keys = q[:, 0] * 65536 + q[:, 1] * 256 + q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    idx = counts.argmax()
    if counts[idx] / max(1, len(flat)) < min_ratio:
        return (0, 0, 0)
    winner = vals[idx]
    return (int((winner >> 16) & 0xFF), int((winner >> 8) & 0xFF), int(winner & 0xFF))


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_inputs(pdf_path: Path, json_path: Path) -> tuple[fitz.Document, dict]:
    assert pdf_path.exists(), f"PDF missing: {pdf_path}"
    assert json_path.exists(), f"JSON missing: {json_path}"
    doc = fitz.open(str(pdf_path))
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert "pages" in parsed, "parsed JSON missing 'pages'"
    print(f"  pdf pages = {doc.page_count}, parsed pages = {len(parsed['pages'])}")
    assert doc.page_count == len(
        parsed["pages"]
    ), "page count mismatch — JSON came from a different PDF?"
    return doc, parsed


def check_schema(parsed: dict) -> None:
    cat_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    fontsize_zero = 0
    cells_total = 0
    cells_with_translated = 0
    for page in parsed["pages"]:
        for el in page["elements"]:
            cat_counter[el["category"]] += 1
            label_counter[el["label"]] += 1
            assert "bbox_pdf" in el and len(el["bbox_pdf"]) == 4
            if el["category"] != "BYPASS" and el.get("font_size", 0) == 0:
                fontsize_zero += 1
            for c in el.get("cells", []):
                cells_total += 1
                assert (
                    "source_text" in c and "bbox_pdf" in c
                ), "cell missing source_text or bbox_pdf — old fixture schema?"
                if c.get("translated_text"):
                    cells_with_translated += 1
    print(f"  categories: {dict(cat_counter)}")
    print(f"  labels: {dict(label_counter)}")
    print(f"  zero font_size (non-BYPASS): {fontsize_zero}")
    print(f"  cells: {cells_total} total, {cells_with_translated} translated")
    assert cells_total > 0, "no cells found — TABLE elements missing cells"
    if cells_total:
        assert (
            cells_with_translated > 0
        ), "no cells have translated_text — phase 2 cell translation broken"


def check_overflow_risk(parsed: dict) -> None:
    """Plan-feasibility check: how often is JSON font_size > bbox height?

    Validates the decision to treat font_size as an UPPER BOUND, not truth.
    """
    risk_count = 0
    total = 0
    for page in parsed["pages"]:
        for el in page["elements"]:
            if el["category"] not in ("FLOWING_TEXT", "IN_PLACE"):
                continue
            total += 1
            x0, y0, x1, y1 = el["bbox_pdf"]
            h = y1 - y0
            fs = el.get("font_size", 0) or 0
            if fs > h:
                risk_count += 1
    pct = 100 * risk_count / max(1, total)
    print(
        f"  font_size > bbox_height in {risk_count}/{total} ({pct:.0f}%) text elements"
    )
    if pct > 10:
        print(
            "  WARN: > 10% overflow if font_size used as truth — shrink-to-fit REQUIRED"
        )


def check_native_text(doc: fitz.Document) -> None:
    """Plan-feasibility check: scanned vs native PDF.

    If the PDF has a native text layer, simple draw_rect erasure is insufficient —
    we must use add_redact_annot + apply_redactions to remove the text layer.
    """
    pages_with_text = 0
    sample = ""
    for i in range(min(3, doc.page_count)):
        t = doc[i].get_text("text").strip()
        if t:
            pages_with_text += 1
            if not sample:
                sample = t[:80]
    print(f"  pages with native text (first 3): {pages_with_text}/3")
    if sample:
        print(f"  sample: {sample!r}")
    if pages_with_text:
        print("  WARN: native text detected — redaction (not just rect erase) required")


def check_pixmap_and_color(doc: fitz.Document, parsed: dict) -> None:
    page = doc[0]
    pm = page.get_pixmap()
    arr = _pixmap_to_array(pm)
    print(f"  pixmap {pm.width}×{pm.height} n={pm.n} → array {arr.shape} {arr.dtype}")
    pw, ph = parsed["pages"][0]["page_width"], parsed["pages"][0]["page_height"]
    # Pick the first non-BYPASS element with a non-trivial bbox.
    target = None
    for el in parsed["pages"][0]["elements"]:
        if el["category"] != "BYPASS" and (el["bbox_pdf"][2] - el["bbox_pdf"][0]) > 50:
            target = el
            break
    assert target is not None, "no testable element on page 0"
    bbox_px = _bbox_to_pixels(target["bbox_pdf"], pw, ph, pm)
    bg = detect_bg_color(arr, bbox_px)
    txt = detect_text_color(arr, bbox_px, bg)
    print(f"  element label={target['label']} bbox_px={bbox_px}")
    print(f"  bg={bg}  text={txt}")
    assert all(0 <= c <= 255 for c in bg + txt), "color channel out of range"


def check_synthetic_render(font_path: Path | None) -> None:
    """Confirm fitz can register a font and insert non-trivial text into a page."""
    fp = str(font_path) if (font_path and font_path.exists()) else None
    doc = fitz.open()
    page = doc.new_page(width=400, height=200)
    if fp:
        page.insert_font(fontname="Body", fontfile=fp)
        fontname = "Body"
    else:
        fontname = "helv"
    page.draw_rect(fitz.Rect(20, 20, 380, 60), color=(1, 1, 1), fill=(1, 1, 1))
    rem = page.insert_textbox(
        fitz.Rect(20, 20, 380, 60),
        "Phase-3 smoke: ăn cơm chưa? (UTF-8 OK)" if fp else "Phase-3 smoke (no font)",
        fontname=fontname,
        fontsize=12,
        color=(0, 0, 0),
        align=fitz.TEXT_ALIGN_LEFT,
    )
    print(f"  insert_textbox returned remaining={rem:.1f}")
    assert rem >= 0, "even synthetic insert overflowed — fitz/font setup wrong"
    extracted = page.get_text("text")
    if fp:
        assert "Phase-3" in extracted, f"text not extractable: {extracted!r}"
    print(f"  synthetic page text-extracts: {extracted.strip()[:60]!r}")


def check_render_module() -> bool:
    try:
        importlib.import_module("pdf2zh.render")
        return True
    except ModuleNotFoundError:
        return False


def run_full_render(args) -> None:
    from pdf2zh.render import RenderConfig, render_document  # noqa: WPS433

    cfg = RenderConfig(font_path=args.font)
    parsed = json.loads(Path(args.parsed).read_text(encoding="utf-8"))
    render_document(args.input, parsed, args.output, cfg)
    out = Path(args.output)
    assert out.exists() and out.stat().st_size > 0, "render produced empty file"
    rdoc = fitz.open(str(out))
    text0 = rdoc[0].get_text("text")
    print(f"  output {out.name}: {out.stat().st_size:,} bytes, {rdoc.page_count} pages")
    print(f"  page 0 extracted text excerpt: {text0.strip()[:120]!r}")
    # Translated content from JSON should be findable somewhere in output.
    first_translation = next(
        (
            el["translated_text"]
            for el in parsed["pages"][0]["elements"]
            if el["translated_text"]
        ),
        None,
    )
    if first_translation:
        # Stripped of HTML tags for comparison.
        snippet = first_translation.replace("<b>", "").replace("</b>", "")[:30]
        assert (
            snippet in text0
        ), f"first translation not found in rendered page: {snippet!r}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", required=True, help="Source PDF")
    ap.add_argument("--parsed", required=True, help="Translated JSON (phase 2 output)")
    ap.add_argument("--font", default=None, help="TTF font path (Unicode)")
    ap.add_argument(
        "--render", action="store_true", help="Run full render via pdf2zh.render"
    )
    ap.add_argument("--output", default=None, help="Output PDF path (when --render)")
    args = ap.parse_args()

    pdf_path = Path(args.input)
    json_path = Path(args.parsed)
    font_path = Path(args.font) if args.font else None

    print("\n[1] check_inputs")
    doc, parsed = check_inputs(pdf_path, json_path)

    print("\n[2] check_schema")
    check_schema(parsed)

    print("\n[3] check_overflow_risk")
    check_overflow_risk(parsed)

    print("\n[4] check_native_text")
    check_native_text(doc)

    print("\n[5] check_pixmap_and_color")
    check_pixmap_and_color(doc, parsed)

    print("\n[6] check_synthetic_render")
    check_synthetic_render(font_path)

    print("\n[7] render module status: ", end="")
    have_render = check_render_module()
    print("AVAILABLE" if have_render else "not yet implemented (pdf2zh.render)")

    if args.render:
        if not have_render:
            print(
                "\nERROR: --render requested but pdf2zh.render module does not exist yet."
            )
            return 2
        if not args.font or not args.output:
            print("\nERROR: --render requires --font and --output.")
            return 2
        print("\n[8] run_full_render")
        run_full_render(args)

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
