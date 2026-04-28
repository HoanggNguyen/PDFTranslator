"""Layout model: page layout detection"""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseModel

logger = logging.getLogger(__name__)


class SuryaLayoutModel(BaseModel):
    """
    Wraps Surya's ``LayoutPredictor``.

    Layout detection uses its own ``FoundationPredictor`` checkpoint (separate
    from the OCR backbone), loaded lazily and kept alive on the GPU.
    """

    model_name = "SuryaLayout"

    def __init__(self, hardware: Any) -> None:
        super().__init__(hardware)
        self._layout_foundation_predictor: Any = None

    @property
    def _layout_foundation(self) -> Any:
        if self._layout_foundation_predictor is None:
            from surya.foundation import FoundationPredictor
            from surya.settings import settings

            self._layout_foundation_predictor = FoundationPredictor(
                checkpoint=settings.LAYOUT_MODEL_CHECKPOINT,
            )
            logger.info("Loaded FoundationPredictor (layout backbone)")
        return self._layout_foundation_predictor

    def _load(self) -> Any:
        from surya.layout import LayoutPredictor

        predictor = LayoutPredictor(self._layout_foundation)
        logger.info("Loaded LayoutPredictor")
        return predictor

    def predict(self, images) -> list[Any]:
        """
        Detect layout regions for a batch of page images.

        Args:
            images: List of PIL images (standard DPI).

        Returns:
            List of Surya layout result objects, one per image.
        """
        return self.predictor(
            images,
            batch_size=self.hardware.layout_batch_size,
        )
