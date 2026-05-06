from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StyleSpec:
    weight: str = "regular"   # "regular" | "bold"
    style_: str = "normal"    # "normal" | "italic"
    align: str = "left"       # "left" | "center" | "right"


@dataclass
class SizingConfig:
    detect: bool = True
    cluster_eps_pt: float = 2.0
    # Maps group name → list of labels belonging to that group
    cluster_groups: dict[str, list[str]] = field(default_factory=lambda: {
        "body": [
            "Text", "ListItem", "Footnote", "Handwriting",
            "Equation", "TextInlineMath", "TableOfContents",
        ],
        "headings": ["SectionHeader"],
        "header_footer": ["PageHeader", "PageFooter"],
        "caption": ["Caption"],
    })
    # "document" = cluster across all pages; "page" = cluster per page
    cluster_scope_by_group: dict[str, str] = field(default_factory=lambda: {
        "body": "document",
        "headings": "document",
        "header_footer": "page",
        "caption": "page",
    })
    cap_height_ratio: float = 0.8
    fallback_size: float = 11.0
    # Used by _estimate_fit_size: avg char width / font_size and line-height / font_size
    char_width_ratio: float = 0.55
    leading_ratio: float = 1.25


@dataclass
class BackgroundConfig:
    enabled: bool = True
    sample_margin_pt: float = 6.0
    dpi_scale: float = 2.0
    complexity_brightness_spread: float = 72.0
    text_contamination_dark_value: int = 220
    text_contamination_dark_ratio: float = 0.015
    min_sample_pixels: int = 24
    eraser_padding_pt: float = 1.5
    fallback_bg: tuple[int, int, int] = (255, 255, 255)


@dataclass
class TextColorConfig:
    enabled: bool = True
    center_fraction: float = 0.6
    fallback: tuple[int, int, int] = (0, 0, 0)


@dataclass
class CompressConfig:
    subset_fonts: bool = True
    deflate: bool = True
    pikepdf_image_recompress: bool = False
    target_dpi: int = 200
    jpeg_quality: int = 78


@dataclass
class RenderConfig:
    # Typst font configuration
    typst_font_paths: list[str] = field(default_factory=list)
    font_family: str = "Noto Sans"
    # Optional per-label style overrides
    styles: dict[str, StyleSpec] = field(default_factory=lambda: {
        "SectionHeader": StyleSpec(weight="bold"),
        "PageHeader": StyleSpec(align="center"),
        "PageFooter": StyleSpec(align="center"),
        "Caption": StyleSpec(style_="italic", align="center"),
        "TableOfContents": StyleSpec(),
    })
    default_style: StyleSpec = field(default_factory=StyleSpec)
    cell_style: StyleSpec = field(default_factory=StyleSpec)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    text_color: TextColorConfig = field(default_factory=TextColorConfig)
    compress: CompressConfig = field(default_factory=CompressConfig)
    min_font_size_pt: float = 7.0
    expand_downward: bool = True
    max_expand_pt: float = 80.0
    # Remove native text layer in translatable regions (needed for non-scanned PDFs)
    redact_native_text: bool = True
    pages: list[int] | None = None
    typst_binary: str = "typst"
    keep_typst_source: bool = False

    # Legacy PyMuPDF fallback fields (kept for the fallback renderer)
    font_path: str = ""
    font_name: str = "Body"

    @classmethod
    def from_json(cls, path: str | Path) -> "RenderConfig":
        import json
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cls()
        if "font_family" in data:
            cfg.font_family = data["font_family"]
        if "typst_font_paths" in data:
            cfg.typst_font_paths = data["typst_font_paths"]
        if "typst_binary" in data:
            cfg.typst_binary = data["typst_binary"]
        if "font_path" in data:
            cfg.font_path = data["font_path"]
        if "min_font_size_pt" in data:
            cfg.min_font_size_pt = float(data["min_font_size_pt"])
        if "pages" in data:
            cfg.pages = data["pages"]
        _load_nested(cfg.sizing, data.get("sizing", {}))
        _load_nested(cfg.background, data.get("background", {}))
        _load_nested(cfg.text_color, data.get("text_color", {}))
        _load_nested(cfg.compress, data.get("compress", {}))
        return cfg


def _load_nested(obj: Any, d: dict) -> None:
    for k, v in d.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
