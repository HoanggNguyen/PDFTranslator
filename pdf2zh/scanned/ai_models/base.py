"""Base class for all AI models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


class BaseImageToTextModel(ABC):
    """
    Shared interface for AI models.
    Pipeline: Init -> Prepare -> Predict -> Postprocess.
    """

    def __init__(self, hardware: Any) -> None:
        """Initialize hardware config and load model weights."""
        self.hardware = hardware
        self.model: Any = None  # Stores the loaded model instance

    @abstractmethod
    def prepare(
        self, images: list[Image.Image], *args: Any, **kwargs: Any
    ) -> list[Image.Image]:
        """Preprocess raw images (e.g., resize, format) before inference."""
        pass

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Run core inference on preprocessed inputs and return raw outputs."""
        pass

    @abstractmethod
    def postprocess(self, *args: Any, **kwargs: Any) -> Any:
        """Format raw model outputs into the final desired structure."""
        pass
