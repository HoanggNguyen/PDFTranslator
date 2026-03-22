"""PDF type detection for routing to appropriate pipeline.

This module provides PDFTypeDetector to classify PDFs as:
- "scanned": Image-based PDFs requiring OCR
- "digital": Text-based PDFs with extractable text
- "mixed": PDFs with both scanned and digital pages
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

PDFType = Literal["scanned", "digital", "mixed"]


class PDFTypeDetector:
    """Detect whether a PDF is scanned, digital, or mixed.

    Detection is based on analyzing text extraction vs image coverage
    on a sample of pages.

    Attributes:
        text_threshold: Minimum characters per page to consider it digital
        image_coverage_threshold: Minimum image area ratio to consider scanned
        sample_pages: Maximum pages to sample for detection
    """

    def __init__(
        self,
        text_threshold: int = 100,
        image_coverage_threshold: float = 0.5,
        sample_pages: int = 5,
        text_block_threshold: int = 3,
    ) -> None:
        """Initialize detector with thresholds.

        Args:
            text_threshold: Min chars per page for digital classification
            image_coverage_threshold: Min image/page area ratio for scanned
            sample_pages: Max pages to analyze (evenly sampled)
            text_block_threshold: Min text blocks for digital fallback when
                font encoding fails (e.g. font.unknown PDFs)
        """
        self.text_threshold = text_threshold
        self.image_coverage_threshold = image_coverage_threshold
        self.sample_pages = sample_pages
        self.text_block_threshold = text_block_threshold

    def detect(self, pdf_path: str | Path) -> PDFType:
        """Detect PDF type.

        Args:
            pdf_path: Path to PDF file

        Returns:
            "scanned", "digital", or "mixed"

        Raises:
            FileNotFoundError: If PDF doesn't exist
            fitz.FileDataError: If file is not a valid PDF
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        try:
            return self._analyze_document(doc)
        finally:
            doc.close()

    def detect_from_bytes(self, pdf_bytes: bytes) -> PDFType:
        """Detect PDF type from bytes.

        Args:
            pdf_bytes: PDF file contents as bytes

        Returns:
            "scanned", "digital", or "mixed"
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            return self._analyze_document(doc)
        finally:
            doc.close()

    def _analyze_document(self, doc: fitz.Document) -> PDFType:
        """Analyze document pages to determine type."""
        page_count = len(doc)
        if page_count == 0:
            logger.warning("Empty PDF, defaulting to digital")
            return "digital"

        # Sample pages evenly
        if page_count <= self.sample_pages:
            sample_indices = list(range(page_count))
        else:
            step = page_count / self.sample_pages
            sample_indices = [int(i * step) for i in range(self.sample_pages)]

        scanned_count = 0
        digital_count = 0

        for page_idx in sample_indices:
            page = doc[page_idx]
            page_type = self._analyze_page(page)

            if page_type == "scanned":
                scanned_count += 1
            else:
                digital_count += 1

        # Classify based on majority
        total_sampled = len(sample_indices)

        if scanned_count == total_sampled:
            return "scanned"
        elif digital_count == total_sampled:
            return "digital"
        else:
            # Mixed detection
            scanned_ratio = scanned_count / total_sampled
            if scanned_ratio >= 0.8:
                return "scanned"
            elif scanned_ratio <= 0.2:
                return "digital"
            else:
                return "mixed"

    def _analyze_page(self, page: fitz.Page) -> Literal["scanned", "digital"]:
        """Analyze a single page."""
        # Extract text
        text = page.get_text("text")
        text_length = len(text.strip())

        # Check for sufficient extractable text
        if text_length >= self.text_threshold:
            return "digital"

        # Fallback: count text block objects even when font encoding is unknown.
        # PDFs with non-standard fonts (e.g. font.unknown) return empty raw text
        # but still have text block structures detectable via get_text("blocks").
        blocks = page.get_text("blocks")
        text_block_count = sum(1 for b in blocks if b[6] == 0)  # type 0 = text
        if text_block_count >= self.text_block_threshold:
            logger.debug(
                f"Font-encoding fallback: {text_block_count} text blocks found "
                f"despite {text_length} raw chars — classifying as digital"
            )
            return "digital"

        # Zero text by any measure → image-based page (scanned or screenshot PDF).
        # image_coverage detection below can miss inline images and PDFs produced
        # by screenshot tools that embed images outside the XObject registry.
        if text_length == 0 and text_block_count == 0:
            logger.debug("No text or text blocks found — classifying as scanned")
            return "scanned"

        # Check image coverage
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height

        if page_area == 0:
            return "digital"

        image_area = 0.0
        image_list = page.get_images(full=True)

        for img_info in image_list:
            xref = img_info[0]
            try:
                # Get image bbox on page
                for img_rect in page.get_image_rects(xref):
                    image_area += img_rect.width * img_rect.height
            except Exception:
                # If we can't get rect, estimate from image size
                try:
                    pix = fitz.Pixmap(page.parent, xref)
                    # Rough estimate: image covers significant portion
                    image_area += pix.width * pix.height * 0.5
                    pix = None
                except Exception:
                    pass

        image_coverage = image_area / page_area

        if image_coverage >= self.image_coverage_threshold:
            return "scanned"

        # Default to digital if unclear
        return "digital"
