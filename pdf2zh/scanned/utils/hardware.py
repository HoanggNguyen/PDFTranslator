"""Hardware detection and configuration for Surya models.

This module provides automatic device detection and hardware profiles
for optimal Surya model performance across different hardware configurations.

DPI is NOT managed here — Surya's ``settings.IMAGE_DPI`` and
``settings.IMAGE_DPI_HIGHRES`` handle rendering resolution internally
when using ``surya.input.load.load_from_file``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

DeviceType = Literal["cuda", "mps", "cpu", "auto"]


@dataclass
class HardwareProfile:
    """Hardware configuration profile for Surya models.

    Attributes:
        device: PyTorch device string ("cuda", "mps", or "cpu")
        batch_size: Number of pages to process in parallel per batch
    """
    device: str
    batch_size: int


_DEFAULT_PROFILES: dict[str, HardwareProfile] = {
    "cuda": HardwareProfile(device="cuda", batch_size=12),
    "mps": HardwareProfile(device="mps", batch_size=4),
    "cpu": HardwareProfile(device="cpu", batch_size=2),
}


def _detect_device() -> str:
    """Detect the best available device.

    Priority: CUDA > MPS > CPU

    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("CUDA device detected")
            return "cuda"

        if torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) device detected")
            return "mps"

    except ImportError:
        logger.warning("PyTorch not available, falling back to CPU")

    logger.info("Using CPU device")
    return "cpu"


def resolve_hardware(
    device: DeviceType = "auto",
    batch_size: int | None = None,
    **_kwargs,
) -> HardwareProfile:
    """Resolve hardware configuration.

    Creates a HardwareProfile with appropriate defaults for the detected
    or specified device, with optional batch_size override.

    Args:
        device: Target device ("auto", "cuda", "mps", or "cpu")
        batch_size: Override default batch size for device

    Returns:
        HardwareProfile with resolved settings
    """
    if device == "auto":
        resolved_device = _detect_device()
    else:
        resolved_device = device

    profile = _DEFAULT_PROFILES.get(resolved_device, _DEFAULT_PROFILES["cpu"])

    return HardwareProfile(
        device=profile.device,
        batch_size=batch_size if batch_size is not None else profile.batch_size,
    )


def set_torch_device_env(device: str) -> None:
    """Set TORCH_DEVICE environment variable.

    This should be called before loading Surya models so they use
    the correct device.

    Args:
        device: Device string ("cuda", "mps", or "cpu")
    """
    os.environ["TORCH_DEVICE"] = device
    logger.debug(f"Set TORCH_DEVICE={device}")


def get_gpu_memory_mb() -> int | None:
    """Get available GPU memory in MB.

    Returns:
        Available memory in MB, or None if not applicable
    """
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device).total_memory
            reserved = torch.cuda.memory_reserved(device)
            free = total - reserved
            return int(free / (1024 * 1024))

        if torch.backends.mps.is_available():
            return 8192  # Assume 8GB available

    except (ImportError, RuntimeError):
        pass

    return None
