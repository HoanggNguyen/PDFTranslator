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
    Models are loaded lazily upon first inference call.
    """

    model_name = "SuryaOCR"

    def __init__(self) -> None:
        super().__init__()
        self.foundation_predictor: Any = None
        self.detection_predictor: Any = None
        self.recognition_predictor: Any = None

        from surya.settings import settings

        settings.DETECTOR_BLANK_THRESHOLD = 0.5
        settings.DETECTOR_TEXT_THRESHOLD = 0.7

    def load_model(self) -> None:
        logger.info(
            "Initializing %s and loading models into memory...", self.model_name
        )

        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        self.foundation_predictor = FoundationPredictor()
        logger.info("Loaded FoundationPredictor (OCR backbone)")

        self.detection_predictor = DetectionPredictor()
        logger.info("Loaded DetectionPredictor")

        self.recognition_predictor = RecognitionPredictor(self.foundation_predictor)
        logger.info("Loaded RecognitionPredictor")

        self.model = self.recognition_predictor

    def unload_model(self) -> None:
        if self.model is not None:
            import torch

            logger.info("Unloading all %s predictors from VRAM...", self.model_name)

            del self.foundation_predictor
            del self.detection_predictor
            del self.recognition_predictor
            del self.model

            self.foundation_predictor = None
            self.detection_predictor = None
            self.recognition_predictor = None
            self.model = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def prepare(
        self,
        images: list[Image.Image],
        highres_images: list[Image.Image] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[list[Image.Image], list[Image.Image] | None]:
        """
        Preprocess raw images before inference.
        """
        return images, highres_images

    def predict(
        self,
        prepared_inputs: Tuple[list[Image.Image], list[Image.Image] | None],
        *args: Any,
        math_mode: bool = False,
        task_names: list[Any] | None = None,
        bboxes: list[Any] | None = None,
        detection_batch_size: int | None = None,
        ocr_batch_size: int | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """
        Run full-page OCR (detection -> recognition) on prepared images.
        """
        images, highres_images = prepared_inputs

        run_kwargs: dict[str, Any] = {"math_mode": True, "return_words": False}

        if not math_mode:
            logger.info("Running OCR with detection + recognition")
            run_kwargs.update(
                {
                    "det_predictor": self.detection_predictor,
                    "detection_batch_size": detection_batch_size,
                    "recognition_batch_size": ocr_batch_size,
                    "highres_images": highres_images,
                }
            )
        else:
            logger.info("Running OCR in math mode (LaTeX recognition)")
            run_kwargs.update(
                {
                    "recognition_batch_size": ocr_batch_size,
                }
            )

        if task_names is not None:
            run_kwargs["task_names"] = task_names
        if bboxes is not None:
            run_kwargs["bboxes"] = bboxes

        raw_results = self.recognition_predictor(images, **run_kwargs)

        return raw_results

    def postprocess(
        self, raw_results: list[Any], *args: Any, **kwargs: Any
    ) -> list[Any]:
        """
        Format raw Surya outputs into the final desired structure.
        """
        return raw_results
