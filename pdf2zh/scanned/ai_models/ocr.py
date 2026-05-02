"""OCR model: text detection + recognition"""

from __future__ import annotations

import logging
from typing import Any, Tuple

from PIL import Image

from pdf2zh.scanned.ai_models.base import BaseImageToTextModel

logger = logging.getLogger(__name__)


class SuryaOCRModel(BaseImageToTextModel):
    """
    Wraps Surya's DetectionPredictor + RecognitionPredictor.
    Models are loaded immediately upon initialization.
    """

    model_name = "SuryaOCR"

    def __init__(self, hardware: Any) -> None:
        """Initialize hardware config and load all Surya predictors immediately."""
        super().__init__(hardware)

        logger.info(
            "Initializing %s and loading models into memory...", self.model_name
        )

        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        # 1. Load backbone
        self.foundation_predictor = FoundationPredictor()
        logger.info("Loaded FoundationPredictor (OCR backbone)")

        # 2. Load detection
        self.detection_predictor = DetectionPredictor()
        logger.info("Loaded DetectionPredictor")

        # 3. Load recognition (requires foundation)
        self.recognition_predictor = RecognitionPredictor(self.foundation_predictor)
        logger.info("Loaded RecognitionPredictor")

    def prepare(
        self, images: list[Image.Image], highres_images: list[Image.Image]
    ) -> Tuple[list[Image.Image], list[Image.Image]]:
        """
        Preprocess raw images before inference.
        Surya models accept raw PIL images directly, so we just pass them through.
        """
        # Add any standard image preprocessing here if needed in the future
        return images, highres_images

    def predict(
        self,
        images: list[Image.Image],
        *,
        highres_images: list[Image.Image] | None = None,
        math_mode: bool = False,
        task_names: list[Any] | None = None,
        bboxes: list[Any] | None = None,
    ) -> list[Any]:
        """
        Run full-page OCR (detection -> recognition) on prepared images.
        """
        run_kwargs: dict[str, Any] = {"math_mode": math_mode}

        images, highres_images = self.prepare(images, highres_images)

        if not math_mode:
            logger.info("Running OCR with detection + recognition")
            run_kwargs.update(
                {
                    "det_predictor": self.detection_predictor,
                    "detection_batch_size": self.hardware.detection_batch_size,
                    "recognition_batch_size": self.hardware.ocr_batch_size,
                    "highres_images": highres_images,
                }
            )
        else:
            logger.info("Running OCR in math mode (LaTeX recognition)")
            run_kwargs.update(
                {
                    "recognition_batch_size": self.hardware.equation_batch_size,
                }
            )

        if task_names is not None:
            run_kwargs["task_names"] = task_names
        if bboxes is not None:
            run_kwargs["bboxes"] = bboxes

        # Raw inference using the Surya RecognitionPredictor
        raw_results = self.recognition_predictor(images, **run_kwargs)

        return self.postprocess(raw_results)

    def postprocess(self, raw_results: list[Any]) -> list[Any]:
        """
        Format raw Surya outputs into the final desired structure.
        """
        return raw_results
