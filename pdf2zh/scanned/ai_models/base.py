"""Base class for all AI models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import logging

from PIL import Image

logger = logging.getLogger(__name__)

class BaseImageToTextModel(ABC):
    """
    Shared interface for AI models with Lazy Loading support.
    Pipeline: Call -> [Load Model] -> Prepare -> Predict -> Postprocess.
    """

    def __init__(self) -> None:
        """Chỉ khai báo các thuộc tính, KHÔNG tải weights vào VRAM ở đây."""
        self.model: Any = None  
        self.device: Any = None 

    @abstractmethod
    def load_model(self) -> None:
        """
        Khởi tạo model và đẩy vào VRAM.
        Các class con BẮT BUỘC phải override hàm này.
        """
        pass

    def unload_model(self) -> None:
        """
        Giải phóng model khỏi VRAM. Hàm này dùng chung cho mọi class con.
        """
        if self.model is not None:
            import torch
            logger.info("Unloading model from VRAM to free memory...")
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @abstractmethod
    def prepare(self, images: list[Image.Image], *args: Any, **kwargs: Any) -> Any:
        """Preprocess raw images."""
        pass

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Run core inference. (Model chắc chắn đã được load khi hàm này chạy)."""
        pass

    @abstractmethod
    def postprocess(self, *args: Any, **kwargs: Any) -> Any:
        """Format raw model outputs."""
        pass

    def __call__(self, images: list[Image.Image], auto_unload: bool = False, *args: Any, **kwargs: Any) -> Any:
        """
        Hàm trung tâm điều phối toàn bộ Pipeline (Template Method).
        """
        # 1. Lazy Loading: Chỉ load nếu model chưa tồn tại
        if self.model is None:
            self.load_model()

        try:
            # 2. Tiền xử lý
            prepared_inputs = self.prepare(images, *args, **kwargs)
            
            # 3. Dự đoán
            raw_outputs = self.predict(prepared_inputs, *args, **kwargs)
            
            # 4. Hậu xử lý
            final_results = self.postprocess(raw_outputs, *args, **kwargs)
            
            return final_results
        finally:
            # 5. Giải phóng VRAM ngay lập tức nếu auto_unload = True
            if auto_unload:
                self.unload_model()