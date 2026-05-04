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
        # KHÔNG import hay khởi tạo FoundationPredictor ở đây nữa

    def load_model(self) -> None:
        """Override hàm load_model để lazy load Surya."""
        logger.info("Initializing %s into VRAM...", self.model_name)
        from surya.foundation import FoundationPredictor
        from surya.layout import LayoutPredictor
        from surya.settings import settings

        self.layout_foundation_predictor = FoundationPredictor(
            checkpoint=settings.LAYOUT_MODEL_CHECKPOINT,
        )
        self.model = LayoutPredictor(self.layout_foundation_predictor)
        logger.info("Loaded LayoutPredictor successfully.")

    def prepare(self, images: list[Image.Image]) -> list[Image.Image]:
        return images

    def predict(self, images: list[Image.Image], batch_size: int | None = None) -> list[Any]:
        # Tự tin gọi self.model vì hàm __call__ của class cha đã đảm bảo nó được load
        return self.model(images, batch_size=batch_size)

    def postprocess(self, raw_results: list[Any]) -> list[Any]:
        return raw_results