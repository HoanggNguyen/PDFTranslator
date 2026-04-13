#!/usr/bin/env python3
"""Visual bbox verification tool for Stage A output.

This script renders a PDF with colored bounding boxes overlaid to verify
that the Stage A parser correctly identifies and locates elements.

Color coding:
- Blue: FLOWING_TEXT (regular text blocks)
- Green: IN_PLACE (headers, footers, captions)
- Red: BYPASS (figures, pictures)
- Purple: TABLE (table boundaries)
- Orange: EQUATION (formulas)
- Yellow (thin): TABLE cells

Usage:
    python scripts/verify_bbox.py --input sample.pdf --output verify_output.pdf
    python scripts/verify_bbox.py --input sample.pdf --output verify_output.pdf --pages 0,1,2
"""

import argparse
import logging
import sys
from pathlib import Path

import fitz  # PyMuPDF

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf2zh.scanned.enums import ElementCategory
from pdf2zh.scanned.parser import StageAParser
from pdf2zh.scanned.schema import validate_stage_output

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Color definitions (RGB tuples, 0-1 scale)
CATEGORY_COLORS = {
    ElementCategory.FLOWING_TEXT: (0.0, 0.0, 1.0),  # Blue
    ElementCategory.IN_PLACE: (0.0, 0.8, 0.0),  # Green
    ElementCategory.BYPASS: (1.0, 0.0, 0.0),  # Red
    ElementCategory.TABLE: (0.5, 0.0, 0.5),  # Purple
    ElementCategory.EQUATION: (1.0, 0.5, 0.0),  # Orange
}

CELL_COLOR = (0.8, 0.8, 0.0)  # Yellow for table cells


def draw_bbox(page: fitz.Page, bbox: list[float], color: tuple, width: float = 2.0) -> None:
    """Draw a rectangle on the page.

    Args:
        page: fitz Page to draw on
        bbox: [x0, y0, x1, y1] in PDF points
        color: RGB tuple (0-1 scale)
        width: Line width in points
    """
    rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
    page.draw_rect(rect, color=color, width=width)


def draw_label(page: fitz.Page, bbox: list[float], label: str, color: tuple) -> None:
    """Draw a label above the bbox.

    Args:
        page: fitz Page to draw on
        bbox: [x0, y0, x1, y1] in PDF points
        label: Text label to display
        color: RGB tuple for text color
    """
    # Position label above the bbox
    text_point = fitz.Point(bbox[0], bbox[1] - 2)

    # Draw label with small font
    page.insert_text(
        text_point,
        label,
        fontsize=8,
        color=color,
    )


def verify_pdf(
    input_path: str,
    output_path: str,
    pages: list[int] | None = None,
    device: str = "auto",
) -> None:
    """Parse a PDF and create a verification output with bbox overlays.

    Args:
        input_path: Path to input PDF
        output_path: Path to save verification PDF
        pages: Optional list of page indices to process
        device: Device for Surya models
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_path}")

    logger.info(f"Parsing {input_path}...")

    # Parse the PDF through explicit Stage A phases
    parser = StageAParser(device=device)
    layout_result = parser.parse_layout(input_path, pages=pages)
    ocr_result = parser.parse_ocr(input_path, pages=pages)
    table_result = parser.parse_tables(input_path, layout_result, ocr_result)
    equation_result = parser.parse_equations(
        input_path,
        layout_result,
        enable_latex=parser.hardware.enable_latex,
    )
    parsed_doc = parser.merge_results(
        input_path,
        layout_result,
        ocr_result,
        table_result=table_result,
        equation_result=equation_result,
    )

    logger.info(f"Found {len(parsed_doc.pages)} pages")

    # Open the original PDF
    doc = fitz.open(input_path)

    # Draw bboxes on each page
    for page_data in parsed_doc.pages:
        page_idx = page_data.page_index
        if page_idx >= len(doc):
            continue

        page = doc[page_idx]
        logger.info(f"Page {page_idx}: {len(page_data.elements)} elements")

        # Draw element bboxes
        for elem in page_data.elements:
            color = CATEGORY_COLORS.get(elem.category, (0.5, 0.5, 0.5))
            draw_bbox(page, elem.bbox_pdf, color, width=2.0)
            draw_label(page, elem.bbox_pdf, f"{elem.label}", color)

            # Draw cell bboxes for tables
            if elem.category == ElementCategory.TABLE:
                for cell in elem.cells:
                    draw_bbox(page, cell.bbox_pdf, CELL_COLOR, width=1.0)

    # Save the annotated PDF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    doc.close()

    logger.info(f"Saved verification PDF to {output_path}")

    # Save JSON output alongside the PDF
    json_path = output_path.with_suffix(".json")
    json_path.write_text(parsed_doc.to_json(indent=2), encoding="utf-8")
    logger.info(f"Saved JSON to {json_path}")

    # Run schema validation
    validation = validate_stage_output(parsed_doc.to_dict(), stage="A", skip_json_schema=True)

    # Print summary
    print("\nVerification Summary:")
    print("=" * 50)
    print(f"Input:  {input_path}")
    print(f"PDF:    {output_path}")
    print(f"JSON:   {json_path}")
    print(f"Pages:  {len(parsed_doc.pages)}")

    total_elements = sum(len(p.elements) for p in parsed_doc.pages)
    print(f"Elements: {total_elements}")

    # Count by category
    category_counts = {}
    for page_data in parsed_doc.pages:
        for elem in page_data.elements:
            cat = elem.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

    print("\nElements by category:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")

    # Validation result
    if validation.valid:
        print("\nSchema validation: PASS")
    else:
        print(f"\nSchema validation: FAIL ({len(validation.errors)} errors)")
        for err in validation.errors:
            print(f"  [{err.code}] {err.path}: {err.message}")

    print("\nColor legend:")
    print("  Blue: FLOWING_TEXT")
    print("  Green: IN_PLACE")
    print("  Red: BYPASS")
    print("  Purple: TABLE")
    print("  Orange: EQUATION")
    print("  Yellow (thin): Table cells")


def main():
    parser = argparse.ArgumentParser(
        description="Verify Stage A bbox detection with visual output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/verify_bbox.py --input sample.pdf --output verify.pdf
    python scripts/verify_bbox.py --input sample.pdf --output verify.pdf --pages 0,1,2
    python scripts/verify_bbox.py --input sample.pdf --output verify.pdf --device cpu
        """
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input PDF file path"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output verification PDF path"
    )
    parser.add_argument(
        "--pages", "-p",
        type=str,
        default=None,
        help="Comma-separated list of page indices (0-based)"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device for Surya models (default: auto)"
    )

    args = parser.parse_args()

    # Parse pages if specified
    pages = None
    if args.pages:
        pages = [int(p.strip()) for p in args.pages.split(",")]

    try:
        verify_pdf(
            input_path=args.input,
            output_path=args.output,
            pages=pages,
            device=args.device,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
