from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import RenderConfig
from .renderer import render_document


def _parse_pages(s: str) -> list[int]:
    pages: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            pages.extend(range(int(a), int(b) + 1))
        else:
            pages.append(int(part))
    return pages


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3: render translated scanned PDF")
    ap.add_argument("--pdf", required=True, help="Original scanned PDF")
    ap.add_argument("--parsed", required=True, help="Translated JSON (phase 2 output)")
    ap.add_argument("--output", required=True, help="Output PDF path")
    ap.add_argument("--font-config", default=None, help="JSON font/render config file")
    ap.add_argument("--font-family", default="Noto Sans", help="Typst font family name")
    ap.add_argument("--font-path", action="append", default=[], dest="font_paths",
                    help="Typst --font-path directory (repeatable)")
    ap.add_argument("--pages", default=None, help="Page filter e.g. 0-4,7,10")
    ap.add_argument("--min-font", type=float, default=7.0, dest="min_font_size_pt")
    ap.add_argument("--typst-bin", default="typst", help="Path to typst binary")
    ap.add_argument("--keep-typst-source", action="store_true",
                    help="Save intermediate .typ file alongside output")
    ap.add_argument("--no-bg-sampling", action="store_true",
                    help="Disable background color sampling (use white)")
    ap.add_argument("--no-redact", action="store_true",
                    help="Skip native text layer redaction (faster, but original text remains selectable)")
    ap.add_argument("--aggressive-compress", action="store_true",
                    help="Re-encode images via pikepdf")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.font_config:
        cfg = RenderConfig.from_json(args.font_config)
    else:
        cfg = RenderConfig()

    # CLI args override config file
    if args.font_family:
        cfg.font_family = args.font_family
    if args.font_paths:
        cfg.typst_font_paths = args.font_paths
    cfg.typst_binary = args.typst_bin
    cfg.min_font_size_pt = args.min_font_size_pt
    cfg.keep_typst_source = args.keep_typst_source
    cfg.compress.pikepdf_image_recompress = args.aggressive_compress
    if args.no_bg_sampling:
        cfg.background.enabled = False
        cfg.text_color.enabled = False
    if args.no_redact:
        cfg.redact_native_text = False
    if args.pages:
        cfg.pages = _parse_pages(args.pages)

    parsed = json.loads(Path(args.parsed).read_text(encoding="utf-8"))
    try:
        stats = render_document(args.pdf, parsed, args.output, cfg)
        print(
            f"Done: pages={stats['pages']} rendered={stats['elements_rendered']} "
            f"skipped={stats['elements_skipped']} cells={stats['cells_rendered']}"
        )
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
