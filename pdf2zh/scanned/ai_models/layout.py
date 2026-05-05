"""Layout model: page layout detection."""

from __future__ import annotations

import logging
from typing import Any

from PIL import Image

from pdf2zh.scanned.ai_models.base import BaseImageToTextModel

logger = logging.getLogger(__name__)


class SuryaLayoutModel(BaseImageToTextModel):
    model_name = "SuryaLayout"

    def __init__(self) -> None:
        super().__init__()

    def load_model(self) -> None:
        logger.info("Initializing %s into VRAM...", self.model_name)
        from surya.foundation import FoundationPredictor
        from surya.layout import LayoutPredictor
        from surya.settings import settings

        self.layout_foundation_predictor = FoundationPredictor(
            checkpoint=settings.LAYOUT_MODEL_CHECKPOINT,
        )
        self.model = LayoutPredictor(self.layout_foundation_predictor)
        logger.info("Loaded LayoutPredictor successfully.")

    def prepare(
        self, images: list[Image.Image], *args: Any, **kwargs: Any
    ) -> list[Image.Image]:
        return images

    def predict(
        self,
        prepared_inputs: list[Image.Image],
        batch_size: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> list[Any]:
        return self.model(prepared_inputs, batch_size=batch_size)

    def postprocess(
        self, raw_results: list[Any], *args: Any, **kwargs: Any
    ) -> list[Any]:
        return raw_results
