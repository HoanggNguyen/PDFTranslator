"""JSON Schema validation for Stage A output.

This module provides:
- ValidationError: Exception for validation failures
- ValidationResult: Result container with errors list
- validate_stage_output: Main validation function checking both JSON Schema
  and runtime invariants not expressible in the schema
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ValidationError:
    """A single validation error.

    Attributes:
        path: JSON path to the error location (e.g., "pages[0].elements[2].bbox_pdf")
        message: Human-readable error description
        code: Short error code for programmatic handling
    """
    path: str
    message: str
    code: str


@dataclass
class ValidationResult:
    """Result of validation.

    Attributes:
        valid: True if no errors were found
        errors: List of ValidationError instances
    """
    valid: bool = True
    errors: list[ValidationError] = field(default_factory=list)

    def add_error(self, path: str, message: str, code: str) -> None:
        """Add an error and mark result as invalid."""
        self.valid = False
        self.errors.append(ValidationError(path=path, message=message, code=code))


def _load_schema() -> dict[str, Any]:
    """Load the JSON Schema definition from the bundled ``schema.json`` file.

    The schema file lives alongside this module in the same package directory
    and is used by :func:`validate_stage_output` for structural validation of
    :class:`~pdf2zh.scanned.models.ParsedDocument` dictionaries.

    Returns:
        Parsed JSON Schema as a Python dictionary.
    """
    schema_path = Path(__file__).parent / "schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _is_finite(value: Any) -> bool:
    """Return True if *value* is a finite number (not NaN or ±Infinity).

    Non-numeric values (e.g. strings, None) are considered valid and return
    True so that callers can use this as a lightweight numeric guard without
    performing isinstance checks themselves.

    Args:
        value: Any Python object to check.  Only ``int`` and ``float`` values
               are tested for NaN / Infinity; all other types pass through.

    Returns:
        ``True`` if *value* is not numeric, or is a finite numeric value.
        ``False`` if *value* is ``float('nan')``, ``float('inf')``, or
        ``float('-inf')``.
    """
    if not isinstance(value, (int, float)):
        return True
    return not (math.isnan(value) or math.isinf(value))


def _check_bbox_valid(bbox: list[float], path: str, result: ValidationResult) -> bool:
    """Validate basic bbox structural invariants and record any errors.

    Checks that *bbox* is a 4-element list of finite numbers where
    ``x0 < x1`` and ``y0 < y1``.  Errors are appended to *result*;
    the function returns a boolean so callers can short-circuit further
    checks that depend on a valid bbox.

    Args:
        bbox: Candidate bounding box ``[x0, y0, x1, y1]``.
        path: JSON path string used as the error location (e.g.
              ``"pages[0].elements[2].bbox_pdf"``).
        result: Mutable :class:`ValidationResult` that errors are appended to.

    Returns:
        ``True`` if all invariants pass; ``False`` if any error was recorded.
    """
    if not isinstance(bbox, list) or len(bbox) != 4:
        result.add_error(path, "bbox must be a list of 4 numbers", "BBOX_FORMAT")
        return False

    for i, v in enumerate(bbox):
        if not _is_finite(v):
            result.add_error(path, f"bbox[{i}] contains NaN or Infinity", "BBOX_INVALID_NUMBER")
            return False

    x0, y0, x1, y1 = bbox
    if not (x0 < x1):
        result.add_error(path, f"bbox x0 ({x0}) must be < x1 ({x1})", "BBOX_X_ORDER")
        return False
    if not (y0 < y1):
        result.add_error(path, f"bbox y0 ({y0}) must be < y1 ({y1})", "BBOX_Y_ORDER")
        return False

    return True


def _check_bbox_within_page(
    bbox: list[float],
    page_width: float,
    page_height: float,
    path: str,
    result: ValidationResult
) -> None:
    """Verify that a bbox lies within the page boundaries and record violations.

    A small floating-point tolerance (0.01 pt) is allowed on all sides to
    accommodate rounding errors introduced by coordinate conversions.

    Args:
        bbox: Validated ``[x0, y0, x1, y1]`` in PDF points.
        page_width: Page width in PDF points (maximum allowed x coordinate).
        page_height: Page height in PDF points (maximum allowed y coordinate).
        path: JSON path string used as the error location.
        result: Mutable :class:`ValidationResult` that errors are appended to.
    """
    x0, y0, x1, y1 = bbox
    # Allow small tolerance for floating point issues
    tolerance = 0.01

    if x0 < -tolerance or y0 < -tolerance:
        result.add_error(
            path,
            f"bbox ({x0}, {y0}, {x1}, {y1}) has negative coordinates",
            "BBOX_NEGATIVE"
        )
    if x1 > page_width + tolerance:
        result.add_error(
            path,
            f"bbox x1 ({x1}) exceeds page_width ({page_width})",
            "BBOX_EXCEEDS_WIDTH"
        )
    if y1 > page_height + tolerance:
        result.add_error(
            path,
            f"bbox y1 ({y1}) exceeds page_height ({page_height})",
            "BBOX_EXCEEDS_HEIGHT"
        )


def _check_cell_within_table(
    cell_bbox: list[float],
    table_bbox: list[float],
    cell_path: str,
    result: ValidationResult
) -> None:
    """Verify that a cell bbox is contained within its parent table bbox.

    A small floating-point tolerance (0.01 pt) is applied on all edges to
    account for rounding during coordinate conversion from image to PDF space.

    Args:
        cell_bbox: Cell bounding box ``[x0, y0, x1, y1]`` in PDF points.
        table_bbox: Parent table bounding box ``[x0, y0, x1, y1]`` in PDF points.
        cell_path: JSON path string for the cell (used in error messages).
        result: Mutable :class:`ValidationResult` that errors are appended to.
    """
    tolerance = 0.01
    cx0, cy0, cx1, cy1 = cell_bbox
    tx0, ty0, tx1, ty1 = table_bbox

    if cx0 < tx0 - tolerance or cy0 < ty0 - tolerance:
        result.add_error(
            cell_path,
            f"cell bbox ({cx0}, {cy0}, {cx1}, {cy1}) starts before table bbox ({tx0}, {ty0}, {tx1}, {ty1})",
            "CELL_OUTSIDE_TABLE"
        )
    if cx1 > tx1 + tolerance or cy1 > ty1 + tolerance:
        result.add_error(
            cell_path,
            f"cell bbox ({cx0}, {cy0}, {cx1}, {cy1}) ends after table bbox ({tx0}, {ty0}, {tx1}, {ty1})",
            "CELL_OUTSIDE_TABLE"
        )


def validate_stage_output(
    data: dict[str, Any],
    stage: str = "A",
    skip_json_schema: bool = False
) -> ValidationResult:
    """Validate ParsedDocument output against JSON Schema and runtime invariants.

    Args:
        data: Dictionary representation of ParsedDocument
        stage: Current stage ("A", "B", "C", or "D") for stage-aware checks
        skip_json_schema: If True, skip JSON Schema validation (useful for testing)

    Returns:
        ValidationResult with valid=True if all checks pass, otherwise errors list

    Runtime invariants checked (not expressible in JSON Schema):
        1. bbox_pdf[0] < bbox_pdf[2] (x0 < x1)
        2. bbox_pdf[1] < bbox_pdf[3] (y0 < y1)
        3. Element bboxes fit within [0, 0, page_width, page_height]
        4. category == "BYPASS" -> source_text == ""
        5. category == "EQUATION" -> source_text may contain surrounding text
           that needs translation; latex holds a placeholder or recognized LaTeX
        6. category == "TABLE" -> len(cells) > 0
        7. category != "TABLE" -> cells == []
        8. Cell bboxes contained within parent TABLE bbox
        9. (Integration test only) len(pages) matches actual PDF page count
        10. chapter_id == "" for all pages after Stage A
        11. chapters == [] after Stage A
        12. end_page >= start_page for ChapterInfo
        13. No NaN or Infinity in numeric fields
    """
    result = ValidationResult()

    # JSON Schema validation (optional)
    if not skip_json_schema:
        try:
            import jsonschema
            schema = _load_schema()
            jsonschema.validate(data, schema)
        except ImportError:
            # jsonschema not installed, skip this validation
            pass
        except jsonschema.ValidationError as e:
            result.add_error(
                ".".join(str(p) for p in e.absolute_path),
                e.message,
                "JSON_SCHEMA"
            )
            # Continue to check other invariants

    # Check top-level fields
    if "pdf_path" not in data:
        result.add_error("pdf_path", "pdf_path is required", "MISSING_FIELD")
        return result

    pages = data.get("pages", [])
    chapters = data.get("chapters", [])

    # Invariant 11: chapters == [] after Stage A
    if stage == "A" and chapters:
        result.add_error("chapters", "chapters must be [] after Stage A", "STAGE_A_CHAPTERS")

    # Check each page
    for page_idx, page in enumerate(pages):
        page_path = f"pages[{page_idx}]"

        # Check page dimensions
        page_width = page.get("page_width", 0)
        page_height = page.get("page_height", 0)

        if not _is_finite(page_width):
            result.add_error(f"{page_path}.page_width", "contains NaN or Infinity", "INVALID_NUMBER")
        if not _is_finite(page_height):
            result.add_error(f"{page_path}.page_height", "contains NaN or Infinity", "INVALID_NUMBER")

        # Invariant 10: chapter_id == "" after Stage A
        if stage == "A" and page.get("chapter_id", "") != "":
            result.add_error(
                f"{page_path}.chapter_id",
                "chapter_id must be '' after Stage A",
                "STAGE_A_CHAPTER_ID"
            )

        # Check each element
        elements = page.get("elements", [])
        for elem_idx, elem in enumerate(elements):
            elem_path = f"{page_path}.elements[{elem_idx}]"

            # Check bbox
            bbox = elem.get("bbox_pdf", [])
            bbox_valid = _check_bbox_valid(bbox, f"{elem_path}.bbox_pdf", result)

            # Invariant 3: bbox within page bounds
            if bbox_valid and page_width > 0 and page_height > 0:
                _check_bbox_within_page(bbox, page_width, page_height, f"{elem_path}.bbox_pdf", result)

            category = elem.get("category", "")
            source_text = elem.get("source_text", "")
            cells = elem.get("cells", [])

            # Invariant 4: BYPASS -> source_text == ""
            if category == "BYPASS" and source_text != "":
                result.add_error(
                    f"{elem_path}.source_text",
                    "source_text must be '' for BYPASS category",
                    "BYPASS_TEXT"
                )

            # Invariant 5: EQUATION may have source_text (surrounding text
            # that needs translation). The latex field must be present and
            # non-empty, either as a placeholder or actual recognized LaTeX.
            if category == "EQUATION":
                latex = elem.get("latex", "")
                if not isinstance(latex, str) or not latex.strip():
                    result.add_error(
                        f"{elem_path}.latex",
                        "EQUATION latex must be a non-empty string",
                        "EQUATION_LATEX"
                    )

            # Invariant 6: TABLE -> len(cells) > 0
            if category == "TABLE" and len(cells) == 0:
                result.add_error(
                    f"{elem_path}.cells",
                    "cells must not be empty for TABLE category",
                    "TABLE_NO_CELLS"
                )

            # Invariant 7: non-TABLE -> cells == []
            if category != "TABLE" and len(cells) > 0:
                result.add_error(
                    f"{elem_path}.cells",
                    "cells must be [] for non-TABLE category",
                    "NON_TABLE_HAS_CELLS"
                )

            # Check cells
            for cell_idx, cell in enumerate(cells):
                cell_path = f"{elem_path}.cells[{cell_idx}]"

                cell_bbox = cell.get("bbox_pdf", [])
                cell_bbox_valid = _check_bbox_valid(cell_bbox, f"{cell_path}.bbox_pdf", result)

                # Invariant 8: cell bbox within table bbox
                if cell_bbox_valid and bbox_valid:
                    _check_cell_within_table(cell_bbox, bbox, cell_path, result)

                # Check for NaN/Infinity in row_id, col_id
                for field_name in ["row_id", "col_id"]:
                    val = cell.get(field_name)
                    if val is not None and not _is_finite(val):
                        result.add_error(
                            f"{cell_path}.{field_name}",
                            f"{field_name} contains NaN or Infinity",
                            "INVALID_NUMBER"
                        )

    # Check chapters (for later stages)
    for ch_idx, chapter in enumerate(chapters):
        ch_path = f"chapters[{ch_idx}]"

        start_page = chapter.get("start_page", 0)
        end_page = chapter.get("end_page", 0)

        # Invariant 12: end_page >= start_page
        if end_page < start_page:
            result.add_error(
                ch_path,
                f"end_page ({end_page}) must be >= start_page ({start_page})",
                "CHAPTER_PAGE_ORDER"
            )

    return result
