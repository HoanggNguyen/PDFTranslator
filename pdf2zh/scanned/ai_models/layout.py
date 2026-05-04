"""Layout model: page layout detection."""

from __future__ import annotations

import logging
from typing import Any

from PIL import Image

from pdf2zh.scanned.ai_models.base import BaseImageToTextModel

logger = logging.getLogger(__name__)


class SuryaLayoutModel(BaseImageToTextModel):
    """
    Wraps Surya's LayoutPredictor.

    Layout detection uses its own FoundationPredictor checkpoint (separate
    from the OCR backbone), loaded immediately on initialization.
    """

    model_name = "SuryaLayout"

    def __init__(self) -> None:
        """Initialize Surya layout models immediately."""
        logger.info("Initializing %s...", self.model_name)

        from surya.foundation import FoundationPredictor
        from surya.layout import LayoutPredictor
        from surya.settings import settings

        # 1. Load foundation specifically for layout
        self.layout_foundation_predictor = FoundationPredictor(
            checkpoint=settings.LAYOUT_MODEL_CHECKPOINT,
        )
        logger.info("Loaded FoundationPredictor (layout backbone)")

        # 2. Load layout predictor
        self.model = LayoutPredictor(self.layout_foundation_predictor)
        logger.info("Loaded LayoutPredictor")

    def prepare(self, images: list[Image.Image]) -> list[Image.Image]:
        """Preprocess a batch of page images."""
        # Surya models accept raw PIL images directly
        return images

    def predict(self, images: list[Image.Image], batch_size: int | None) -> list[Any]:
        """
        Detect layout regions for a batch of prepared page images.
        """
        try:
            images = self.prepare(images)

            raw_results = self.model(
                images,
                batch_size=batch_size,
            )
            return self.postprocess(raw_results)

        except Exception:
            logger.exception(
                "Layout detection failed for batch of %d images — returning nulls.",
                len(images),
            )
            return [None] * len(images)

    def postprocess(self, raw_results: list[Any]) -> list[Any]:
        """Format raw Surya layout outputs."""
        # Custom formatting logic can go here in the future
        return raw_results
