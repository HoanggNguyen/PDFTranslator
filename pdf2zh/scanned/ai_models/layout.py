"""Layout model: page layout detection."""

from __future__ import annotations

import logging
from typing import Any
import os

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

        checkpoint_path = "pdf2zh/scanned/model_path/layout"

        if not os.path.exists(checkpoint_path):
            logger.error(
                "Layout model checkpoint not found at %s. Please ensure the checkpoint is placed correctly.",
                checkpoint_path,
            )
            raise FileNotFoundError(f"Layout model checkpoint not found at {checkpoint_path}")

        self.layout_foundation_predictor = FoundationPredictor(
            checkpoint=checkpoint_path,
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
