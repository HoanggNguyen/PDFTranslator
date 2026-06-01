from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import fitz
import numpy as np

from .config import BackgroundConfig, TextColorConfig

if TYPE_CHECKING:
    pass

RGB = tuple[int, int, int]


@dataclass
class CoverPlan:
    kind: str  # "flat" or "strip"
    rgb: RGB  # used when kind == "flat"
    pixmap: fitz.Pixmap | None = None  # used when kind == "strip"


# ---------------------------------------------------------------------------
# Background sampling
# ---------------------------------------------------------------------------


def prepare_cover(
    page: fitz.Page,
    bbox_pdf: list[float],
    page_width: float,
    page_height: float,
    cfg: BackgroundConfig,
) -> CoverPlan:
    if not cfg.enabled:
        return CoverPlan(kind="flat", rgb=cfg.fallback_bg)

    try:
        rgb = _sample_donut_median(page, bbox_pdf, page_width, page_height, cfg)
        return CoverPlan(kind="flat", rgb=rgb)
    except Exception:
        return CoverPlan(kind="flat", rgb=cfg.fallback_bg)


def _sample_donut_median(
    page: fitz.Page,
    bbox_pdf: list[float],
    page_width: float,
    page_height: float,
    cfg: BackgroundConfig,
) -> RGB:
    margin = cfg.sample_margin_pt
    x0, y0, x1, y1 = bbox_pdf
    outer = fitz.Rect(
        max(0.0, x0 - margin),
        max(0.0, y0 - margin),
        min(page_width, x1 + margin),
        min(page_height, y1 + margin),
    )
    if outer.is_empty:
        return cfg.fallback_bg

    mat = fitz.Matrix(cfg.dpi_scale, cfg.dpi_scale)
    pm = page.get_pixmap(matrix=mat, clip=outer, colorspace=fitz.csRGB, alpha=False)
    arr = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, 3)

    # Build donut mask: True for pixels OUTSIDE the inner bbox (donut band)
    sx = pm.width / outer.width
    sy = pm.height / outer.height
    inner_x0 = int((x0 - outer.x0) * sx)
    inner_y0 = int((y0 - outer.y0) * sy)
    inner_x1 = int((x1 - outer.x0) * sx)
    inner_y1 = int((y1 - outer.y0) * sy)

    mask = np.ones((pm.height, pm.width), dtype=bool)
    mask[
        max(0, inner_y0) : min(pm.height, inner_y1),
        max(0, inner_x0) : min(pm.width, inner_x1),
    ] = False

    donut_pixels = arr[mask].reshape(-1, 3)
    if len(donut_pixels) < cfg.min_sample_pixels:
        return cfg.fallback_bg

    if _is_text_contaminated(donut_pixels, cfg):
        return _trimmed_robust(donut_pixels, cfg)

    brightness_spread = int(donut_pixels.max()) - int(donut_pixels.min())
    if brightness_spread > cfg.complexity_brightness_spread:
        return _trimmed_robust(donut_pixels, cfg)

    r = int(np.median(donut_pixels[:, 0]))
    g = int(np.median(donut_pixels[:, 1]))
    b = int(np.median(donut_pixels[:, 2]))
    return (r, g, b)


def _trimmed_robust(pixels: np.ndarray, cfg: BackgroundConfig) -> RGB:
    """Drop darkest 20% (likely text bleed), then per-channel median."""
    brightness = pixels.mean(axis=1)
    threshold = np.percentile(brightness, 20)
    keep = pixels[brightness >= threshold]
    if len(keep) == 0:
        keep = pixels
    r = int(np.median(keep[:, 0]))
    g = int(np.median(keep[:, 1]))
    b = int(np.median(keep[:, 2]))
    return (r, g, b)


def _is_text_contaminated(pixels: np.ndarray, cfg: BackgroundConfig) -> bool:
    """Return True if pixels look light overall but have too many dark pixels (text bleed)."""
    median_val = float(np.median(pixels))
    if median_val < 245:
        return False
    dark_ratio = float((pixels < cfg.text_contamination_dark_value).any(axis=1).mean())
    return dark_ratio > cfg.text_contamination_dark_ratio


# ---------------------------------------------------------------------------
# Text color sampling
# ---------------------------------------------------------------------------


def sample_text_color(
    page: fitz.Page,
    bbox_pdf: list[float],
    page_width: float,
    page_height: float,
    bg: RGB,
    cfg: TextColorConfig,
) -> RGB:
    if not cfg.enabled:
        return cfg.fallback

    try:
        x0, y0, x1, y1 = bbox_pdf
        w = x1 - x0
        h = y1 - y0
        cx0 = x0 + w * (1 - cfg.center_fraction) / 2
        cy0 = y0 + h * (1 - cfg.center_fraction) / 2
        cx1 = x0 + w * (1 + cfg.center_fraction) / 2
        cy1 = y0 + h * (1 + cfg.center_fraction) / 2
        inner = fitz.Rect(cx0, cy0, cx1, cy1)
        if inner.is_empty:
            return cfg.fallback

        pm = page.get_pixmap(
            matrix=fitz.Matrix(2, 2), clip=inner, colorspace=fitz.csRGB, alpha=False
        )
        arr = np.frombuffer(pm.samples, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        bg_arr = np.array(bg, dtype=np.int32)
        dist = np.sqrt(((arr - bg_arr) ** 2).sum(axis=1))
        text_mask = dist > 80
        text_pixels = arr[text_mask]
        text_dist = dist[text_mask]
        if len(text_pixels) < 5 or len(text_pixels) / max(1, len(arr)) < 0.02:
            return cfg.fallback
        # Select pixels most different from background (core text, not antialiased edges).
        # Distance-based selection works for any text color including teal, blue, red…
        # "Darkest" heuristic would fail for non-dark colored text on light backgrounds.
        dist_threshold = np.percentile(text_dist, 50)
        core = text_pixels[text_dist >= dist_threshold]
        if len(core) == 0:
            core = text_pixels
        r = int(np.median(core[:, 0]))
        g = int(np.median(core[:, 1]))
        b = int(np.median(core[:, 2]))
        return (r, g, b)
    except Exception:
        return cfg.fallback
