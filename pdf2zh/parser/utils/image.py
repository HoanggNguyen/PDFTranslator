"""PDF page rendering utilities.

This module provides functions to render PDF pages to PIL Images
for Surya processing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import fitz  # PyMuPDF
from PIL import Image

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _extract_dominant_image(
    page: fitz.Page,
    coverage_threshold: float = 0.8,
) -> Image.Image | None:
    """Extract the dominant full-page embedded image, if one exists.

    For scanned PDFs the page content is typically a single high-resolution
    raster image embedded at 300+ DPI.  Re-rendering through fitz at 150 DPI
    halves the effective resolution and causes Surya to miss fine layout
    structure.  Extracting the raw embedded image preserves the original DPI.

    Args:
        page: fitz Page object
        coverage_threshold: Minimum fraction of page area the image must cover

    Returns:
        PIL Image if a dominant embedded image is found, else None
    """
    page_rect = page.rect
    page_area = page_rect.width * page_rect.height
    if page_area == 0:
        return None

    image_list = page.get_images(full=True)
    best_xref = None
    best_coverage = 0.0

    for img_info in image_list:
        xref = img_info[0]
        try:
            rects = list(page.get_image_rects(xref))
            if not rects:
                continue
            covered = sum(r.width * r.height for r in rects) / page_area
            if covered > best_coverage:
                best_coverage = covered
                best_xref = xref
        except Exception:
            continue

    if best_xref is None or best_coverage < coverage_threshold:
        return None

    try:
        pix = fitz.Pixmap(page.parent, best_xref)
        # Normalize to RGB. pix.n counts all components including alpha.
        # Grayscale (n=1), Gray+A (n=2), CMYK (n=4 no alpha), CMYK+A (n=5)
        # all need conversion — only pure RGB (n=3, alpha=0) is already correct.
        n_colors = pix.n - pix.alpha
        if n_colors != 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        if pix.alpha:
            img = img.convert("RGB")
        logger.debug(
            f"Using embedded image (xref={best_xref}, "
            f"{pix.width}×{pix.height}px, n_colors={n_colors}, coverage={best_coverage:.0%})"
        )
        return img
    except Exception as e:
        logger.debug(f"Could not extract embedded image xref={best_xref}: {e}")
        return None


def render_page_to_image(
    page: fitz.Page,
    dpi: int = 150,
) -> Image.Image:
    """Render a PDF page for OCR — prefers native embedded image.

    Prefers extracting the dominant full-page embedded image at its native
    resolution.  Higher resolution means clearer characters for OCR.
    Falls back to fitz rendering when no dominant embedded image is found.

    Args:
        page: fitz Page object to render
        dpi: Resolution used for fitz fallback rendering (default 150)

    Returns:
        PIL Image in RGB mode
    """
    img = _extract_dominant_image(page)
    if img is not None:
        return img
    return _fitz_render(page, dpi)


def render_page_for_layout(
    page: fitz.Page,
    dpi: int = 96,
) -> Image.Image:
    """Render a PDF page for layout detection — always uses fitz at fixed DPI.

    Surya's layout model (and its sibling detection/reading-order models) is
    calibrated to receive images rendered at IMAGE_DPI = 96.  That is the DPI
    that Surya itself uses internally (see surya/input/processing.py:
    get_page_images).  Passing images at a much higher DPI (e.g. 150-300)
    causes the layout model to see the same document content at a larger pixel
    scale than it was trained on, which results in collapsed or missed layout
    bounding boxes.

    This function always renders through fitz so that the image scale is
    predictable regardless of whether embedded raster images exist.

    Args:
        page: fitz Page object to render
        dpi: Target DPI for layout rendering (default 96, matching Surya IMAGE_DPI)

    Returns:
        PIL Image in RGB mode
    """
    return _fitz_render(page, dpi)


def _fitz_render(page: fitz.Page, dpi: int) -> Image.Image:
    """Render a PDF page to a PIL Image using fitz at the specified DPI.

    Converts ``dpi`` to a scale factor relative to fitz's default 72 DPI and
    renders the page without an alpha channel (RGB mode).

    Args:
        page: fitz Page object to render.
        dpi: Output resolution in dots per inch.  Higher values produce larger
             images with finer detail but require more memory.

    Returns:
        PIL Image in ``RGB`` mode at the requested DPI.
    """
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def render_pages_batch(
    doc: fitz.Document,
    page_indices: list[int],
    dpi: int = 150,
) -> list[Image.Image]:
    """Render multiple PDF pages to PIL Images.

    Args:
        doc: fitz Document object
        page_indices: List of 0-based page indices to render
        dpi: Resolution in dots per inch

    Returns:
        List of PIL Images in same order as page_indices
    """
    images = []
    for page_idx in page_indices:
        page = doc[page_idx]
        img = render_page_to_image(page, dpi=dpi)
        images.append(img)
    return images


def get_page_dimensions(page: fitz.Page) -> tuple[float, float]:
    """Get page dimensions in PDF points.

    Args:
        page: fitz Page object

    Returns:
        (width, height) in PDF points
    """
    rect = page.rect
    return rect.width, rect.height


def crop_image_to_bbox(
    image: Image.Image,
    bbox: list[float],
    pdf_width: float,
    pdf_height: float,
) -> Image.Image:
    """Crop a rendered image to a bounding box.

    Args:
        image: PIL Image rendered from PDF page
        bbox: [x0, y0, x1, y1] in PDF points
        pdf_width: Original page width in PDF points
        pdf_height: Original page height in PDF points

    Returns:
        Cropped PIL Image
    """
    img_width, img_height = image.size

    # Calculate scale factors
    scale_x = img_width / pdf_width
    scale_y = img_height / pdf_height

    # Convert bbox to image pixels
    x0 = int(bbox[0] * scale_x)
    y0 = int(bbox[1] * scale_y)
    x1 = int(bbox[2] * scale_x)
    y1 = int(bbox[3] * scale_y)

    # Clamp to image bounds
    x0 = max(0, min(x0, img_width))
    y0 = max(0, min(y0, img_height))
    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))

    # Ensure valid crop region
    if x1 <= x0 or y1 <= y0:
        # Return a small placeholder image
        return Image.new("RGB", (1, 1), color=(255, 255, 255))

    return image.crop((x0, y0, x1, y1))
