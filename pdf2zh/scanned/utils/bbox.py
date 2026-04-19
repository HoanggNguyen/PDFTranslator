"""Bounding box conversion and manipulation utilities.

This module handles coordinate conversion between Surya's model output
coordinates and fitz's PDF-point coordinates.

Coordinate spaces
-----------------
Surya produces three distinct coordinate spaces:

1. **PIL image space** — the raw PIL Image rendered from the PDF (via fitz or
   embedded image extraction).  This is what you pass into the predictor as
   ``List[Image.Image]``.

2. **image_processor numpy space** — after ``processor.image_processor()``
   resizes the PIL image to the model's ``max_size`` and converts it to a
   float32 numpy array.  ``LayoutResult.image_bbox`` and
   ``LayoutBox.polygon`` (and therefore ``LayoutBox.bbox``) live in *this*
   space.

3. **OCR image space** — Surya's recognition predictor keeps text-line polygons
   in the *PIL image* pixel space.  ``OCRResult.image_bbox`` and
   ``TextLine.bbox`` are in this space.

4. **PDF point space** — the target coordinate system for ``ElementData.bbox_pdf``.
   Both axes have top-left origin with Y increasing downward;  only scaling is
   needed (no Y-axis flip).

Key assumptions:
- All four spaces share top-left origin with Y increasing downward
- No Y-axis flip is required between any two spaces, only scaling
- The correct image dimensions to use for layout→PDF scaling come from
  ``layout_result.image_bbox``, NOT from the PIL image size
"""

from __future__ import annotations


def convert_bbox(
    surya_bbox: list[float] | tuple[float, ...],
    image_width: float,
    image_height: float,
    pdf_width: float,
    pdf_height: float,
) -> list[float]:
    """Convert Surya pixel coordinates to fitz PDF points.

    Both coordinate systems use top-left origin with Y increasing downward,
    so only scaling is needed (no Y-axis flip).

    Args:
        surya_bbox: [x0, y0, x1, y1] in the source image pixel space
        image_width: Width of the source image in pixels
        image_height: Height of the source image in pixels
        pdf_width: Width in PDF points (from page.rect.width)
        pdf_height: Height in PDF points (from page.rect.height)

    Returns:
        [x0, y0, x1, y1] in PDF points

    Raises:
        ValueError: If image dimensions are zero or negative
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image dimensions: {image_width}x{image_height}")

    sx0, sy0, sx1, sy1 = surya_bbox

    # Scale factors
    scale_x = pdf_width / image_width
    scale_y = pdf_height / image_height

    # Convert coordinates (no Y flip needed)
    x0 = sx0 * scale_x
    y0 = sy0 * scale_y
    x1 = sx1 * scale_x
    y1 = sy1 * scale_y

    return [x0, y0, x1, y1]


def polygon_to_bbox(polygon: list[list[float]]) -> list[float]:
    """Convert a Surya polygon to an axis-aligned bounding box.

    Surya returns ``PolygonBox`` objects whose ``polygon`` field contains four
    corners that may be slightly skewed (non-axis-aligned).  For coordinate
    conversion we need the axis-aligned envelope, which is what
    ``PolygonBox.bbox`` also computes.

    Args:
        polygon: 4-corner polygon as [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]

    Returns:
        [x_min, y_min, x_max, y_max] axis-aligned bbox
    """
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def image_bbox_to_pdf(
    surya_bbox: list[float] | tuple[float, ...],
    image_bbox: list[float],
    pdf_width: float,
    pdf_height: float,
) -> list[float]:
    """Scale a bbox from Surya's ``result.image_bbox`` space to PDF points.

    Surya's layout model (and other foundation-model-based predictors) internally
    resizes the input PIL image to a fixed ``max_size`` via ``image_processor``
    before running inference.  The output polygon/bbox coordinates are therefore
    in that *resized numpy array* space, not in the original PIL image space.
    ``LayoutResult.image_bbox`` (and ``OCRResult.image_bbox``) records the
    actual dimensions used: ``[0, 0, W, H]``.

    This function uses those recorded dimensions to compute the correct scale
    factor, avoiding the off-by-scale bug that occurs when you use the PIL
    image's ``.size`` instead.

    Args:
        surya_bbox: [x0, y0, x1, y1] in ``result.image_bbox`` coordinate space
        image_bbox: Surya result ``image_bbox`` field, e.g. ``[0, 0, 768, 768]``
        pdf_width: Target PDF page width in points
        pdf_height: Target PDF page height in points

    Returns:
        [x0, y0, x1, y1] in PDF points
    """
    # image_bbox = [0, 0, image_w, image_h]
    _, _, iw, ih = image_bbox
    return convert_bbox(surya_bbox, iw, ih, pdf_width, pdf_height)


def clamp_bbox(
    bbox: list[float],
    page_width: float,
    page_height: float,
) -> list[float]:
    """Clamp bbox coordinates to page bounds.

    Ensures the bbox fits within [0, 0, page_width, page_height].

    Args:
        bbox: [x0, y0, x1, y1] in PDF points
        page_width: Maximum x coordinate
        page_height: Maximum y coordinate

    Returns:
        Clamped [x0, y0, x1, y1]
    """
    x0, y0, x1, y1 = bbox

    x0 = max(0.0, min(x0, page_width))
    y0 = max(0.0, min(y0, page_height))
    x1 = max(0.0, min(x1, page_width))
    y1 = max(0.0, min(y1, page_height))

    return [x0, y0, x1, y1]


def offset_bbox(
    bbox: list[float],
    offset_x: float,
    offset_y: float,
) -> list[float]:
    """Apply offset to bbox coordinates.

    Used for converting cell coordinates from table-relative to page-absolute.

    Args:
        bbox: [x0, y0, x1, y1] in any coordinate space
        offset_x: X offset to add
        offset_y: Y offset to add

    Returns:
        Offset [x0, y0, x1, y1]
    """
    x0, y0, x1, y1 = bbox
    return [
        x0 + offset_x,
        y0 + offset_y,
        x1 + offset_x,
        y1 + offset_y,
    ]


def is_degenerate(bbox: list[float], min_size: float = 0.1) -> bool:
    """Check if bbox is degenerate (zero or negative area).

    A bbox is degenerate if:
    - x0 >= x1 (no width)
    - y0 >= y1 (no height)
    - Width or height is less than min_size

    Args:
        bbox: [x0, y0, x1, y1]
        min_size: Minimum acceptable dimension

    Returns:
        True if bbox is degenerate
    """
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0

    return width < min_size or height < min_size


def normalize_bbox(bbox: list[float]) -> list[float]:
    """Ensure bbox has x0 < x1 and y0 < y1 by swapping if needed.

    Args:
        bbox: [x0, y0, x1, y1] possibly with inverted coordinates

    Returns:
        Normalized [x0, y0, x1, y1] with x0 <= x1 and y0 <= y1
    """
    x0, y0, x1, y1 = bbox

    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0

    return [x0, y0, x1, y1]


def bbox_area(bbox: list[float]) -> float:
    """Calculate bbox area.

    Args:
        bbox: [x0, y0, x1, y1]

    Returns:
        Area (width * height), or 0 if degenerate
    """
    x0, y0, x1, y1 = bbox
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    return width * height


def bbox_intersection(
    bbox1: list[float],
    bbox2: list[float],
) -> list[float] | None:
    """Calculate intersection of two bboxes.

    Args:
        bbox1: First [x0, y0, x1, y1]
        bbox2: Second [x0, y0, x1, y1]

    Returns:
        Intersection bbox, or None if no intersection
    """
    x0 = max(bbox1[0], bbox2[0])
    y0 = max(bbox1[1], bbox2[1])
    x1 = min(bbox1[2], bbox2[2])
    y1 = min(bbox1[3], bbox2[3])

    if x0 >= x1 or y0 >= y1:
        return None

    return [x0, y0, x1, y1]


def bbox_iou(bbox1: list[float], bbox2: list[float]) -> float:
    """Calculate Intersection over Union (IoU) of two bboxes.

    Args:
        bbox1: First [x0, y0, x1, y1]
        bbox2: Second [x0, y0, x1, y1]

    Returns:
        IoU value between 0 and 1
    """
    intersection = bbox_intersection(bbox1, bbox2)
    if intersection is None:
        return 0.0

    inter_area = bbox_area(intersection)
    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area
