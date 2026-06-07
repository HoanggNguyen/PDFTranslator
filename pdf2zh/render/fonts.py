from __future__ import annotations

import fitz

from .config import RenderConfig


def register_font_for_page(page: fitz.Page, cfg: RenderConfig) -> None:
    page.insert_font(fontname=cfg.font_name, fontfile=cfg.font_path)
