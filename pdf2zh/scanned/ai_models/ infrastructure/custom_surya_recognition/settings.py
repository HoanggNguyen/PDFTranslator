import os
from typing import Callable, Dict, Optional

import torch
from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings
from pathlib import Path
from platformdirs import user_cache_dir


class Settings(BaseSettings):
    # General
    TORCH_DEVICE: Optional[str] = None
    IMAGE_DPI: int = 96  # Used for detection, layout, reading order
    IMAGE_DPI_HIGHRES: int = 192  # Used for OCR, table rec
    IN_STREAMLIT: bool = False  # Whether we're running in streamlit
    FLATTEN_PDF: bool = True  # Flatten PDFs by merging form fields before processing
    DISABLE_TQDM: bool = False  # Disable tqdm progress bars
    S3_BASE_URL: str = "https://models.datalab.to"
    PARALLEL_DOWNLOAD_WORKERS: int = (
        10  # Number of workers for parallel model downloads
    )
    MODEL_CACHE_DIR: str = str(Path(user_cache_dir("datalab")) / "models")
    LOGLEVEL: str = "INFO"  # Logging level

    # Paths
    DATA_DIR: str = "data"
    RESULT_DIR: str = "results"
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FONT_DIR: str = os.path.join(BASE_DIR, "static", "fonts")

    @computed_field
    def TORCH_DEVICE_MODEL(self) -> str:
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE

        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        try:
            import torch_xla

            if len(torch_xla.devices()) > 0:
                return "xla"
        except Exception:
            pass

        return "cpu"

    # Text recognition
    RECOGNITION_MODEL_CHECKPOINT: str = "s3://text_recognition/2025_05_16"
    RECOGNITION_MODEL_QUANTIZE: bool = False
    RECOGNITION_MAX_TOKENS: Optional[int] = None
    RECOGNITION_BATCH_SIZE: Optional[int] = (
        None  # Defaults to 8 for CPU/MPS, 256 otherwise
    )
    RECOGNITION_CHUNK_SIZE: Optional[int] = None
    RECOGNITION_BENCH_DATASET_NAME: str = "vikp/rec_bench"
    RECOGNITION_PAD_VALUE: int = 255  # Should be 0 or 255


    @computed_field
    def DETECTOR_STATIC_CACHE(self) -> bool:
        return (
            self.COMPILE_ALL
            or self.COMPILE_DETECTOR
            or self.TORCH_DEVICE_MODEL == "xla"
        )  # We need to static cache and pad to batch size for XLA, since it will recompile otherwise

    @computed_field
    def LAYOUT_STATIC_CACHE(self) -> bool:
        return (
            self.COMPILE_ALL or self.COMPILE_LAYOUT or self.TORCH_DEVICE_MODEL == "xla"
        )

    @computed_field
    def TABLE_REC_STATIC_CACHE(self) -> bool:
        return (
            self.COMPILE_ALL
            or self.COMPILE_TABLE_REC
            or self.TORCH_DEVICE_MODEL == "xla"
        )

    @computed_field
    def OCR_ERROR_STATIC_CACHE(self) -> bool:
        return (
            self.COMPILE_ALL
            or self.COMPILE_OCR_ERROR
            or self.TORCH_DEVICE_MODEL == "xla"
        )

    @computed_field
    def MODEL_DTYPE(self) -> torch.dtype:
        if self.TORCH_DEVICE_MODEL == "cpu":
            return torch.float32
        if self.TORCH_DEVICE_MODEL == "xla":
            return torch.bfloat16
        return torch.float16

    @computed_field
    def MODEL_DTYPE_BFLOAT(self) -> torch.dtype:
        if self.TORCH_DEVICE_MODEL == "cpu":
            return torch.float32
        if self.TORCH_DEVICE_MODEL == "mps":
            return torch.bfloat16
        return torch.bfloat16

    @computed_field
    def INFERENCE_MODE(self) -> Callable:
        if self.TORCH_DEVICE_MODEL == "xla":
            return torch.no_grad
        return torch.inference_mode

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()
