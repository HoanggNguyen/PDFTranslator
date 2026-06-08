from __future__ import annotations

import re

from .background import RGB
from .config import RenderConfig, StyleSpec
from .labels import normalize_label, style_key
from .markup import (
    _split_math_vars,
    escape_typst_string,
    has_bare_latex,
    has_malformed_typst_math,
    has_unbalanced_math_tags,
    is_pure_math_text,
    parse_toc_entries,
    to_typst_markup,
    to_typst_native,
)

CMARKER_VERSION = "0.1.8"


MITEX_VERSION = "0.2.6"

# Detects legacy LaTeX inside <math> tags (backslash commands like \frac, \sum)
_LATEX_IN_MATH = re.compile(r"<math[^>]*>[^<]*\\[a-zA-Z]", re.DOTALL)

# Detects bare Typst math function calls (frac(...), sqrt(...), etc.) outside <math> tags
_BARE_TYPST_MATH = re.compile(
    r"(?:^|[^a-zA-Z])(?:frac|sqrt|root|binom|sum|prod|integral|mat|vec|cases|abs|norm|floor|ceil)\s*\(",
    re.IGNORECASE,
)

_FIT_HELPERS = """\
#let pdftr_fit_size(lo, hi, eps, fits) = {
  if hi - lo <= eps {
    lo
  } else {
    let mid = lo + (hi - lo) / 2
    if fits(mid) {
      pdftr_fit_size(mid, hi, eps, fits)
    } else {
      pdftr_fit_size(lo, mid, eps, fits)
    }
  }
}
#let pdftr_floor_size(value, floor) = if value < floor { floor } else { value }
#let pdftr_floor_leading(value, floor) = if value < floor { floor } else { value }
#let pdftr_fit_markdown(markdown, max_size: 10pt, min_size: 9pt, max_leading: 0.66em, min_leading: 0.54em, fit_height: none, weight: "regular", style: "normal", eps: 0.08pt, math: none) = {
  layout(size => {
    let allowed-height = if fit_height == none { size.height } else { calc.min(size.height, fit_height) }
    let render(text_size, leading) = block(width: size.width)[#{
      set text(size: text_size, weight: weight, style: style)
      set par(leading: leading)
      cmarker.render(markdown, math: math)
    }]
    let fits(text_size, leading) = measure(width: size.width, render(text_size, leading)).height <= allowed-height
    if fits(max_size, max_leading) {
      render(max_size, max_leading)
    } else {
      let fallback_min_size = pdftr_floor_size(min_size - 1.6pt, 5.4pt)
      let fallback_min_leading = pdftr_floor_leading(min_leading - 0.12em, 0.14em)
      let emergency_min_size = pdftr_floor_size(fallback_min_size - 1.2pt, 4.8pt)
      let emergency_min_leading = pdftr_floor_leading(fallback_min_leading - 0.08em, 0.10em)
      let chosen_leading = if fits(min_size, max_leading) { max_leading } else { min_leading }
      let chosen_size = if not fits(min_size, chosen_leading) {
        let fallback_leading = pdftr_floor_leading(chosen_leading - 0.12em, fallback_min_leading)
        let emergency_leading = pdftr_floor_leading(fallback_leading - 0.08em, emergency_min_leading)
        if not fits(fallback_min_size, fallback_leading) {
          if not fits(emergency_min_size, emergency_leading) {
            emergency_min_size
          } else {
            pdftr_fit_size(emergency_min_size, fallback_min_size, eps, size_pt => fits(size_pt, emergency_leading))
          }
        } else {
          pdftr_fit_size(fallback_min_size, min_size, eps, size_pt => fits(size_pt, fallback_leading))
        }
      } else {
        pdftr_fit_size(min_size, max_size, eps, size_pt => fits(size_pt, chosen_leading))
      }
      let final_leading = if fits(min_size, chosen_leading) {
        chosen_leading
      } else if fits(fallback_min_size, pdftr_floor_leading(chosen_leading - 0.12em, fallback_min_leading)) {
        pdftr_floor_leading(chosen_leading - 0.12em, fallback_min_leading)
      } else {
        emergency_min_leading
      }
      render(chosen_size, final_leading)
    }
  })
}
#let pdftr_fit_typst(markup, max_size: 10pt, min_size: 9pt, max_leading: 0.66em, min_leading: 0.54em, fit_height: none, weight: "regular", style: "normal", eps: 0.08pt, no_wrap: false) = {
  layout(size => {
    let allowed-height = if fit_height == none { size.height } else { calc.min(size.height, fit_height) }
    let render(text_size, leading) = block(width: size.width)[#{
      set text(size: text_size, weight: weight, style: style)
      set par(leading: leading)
      eval(markup, mode: "markup")
    }]
    if no_wrap {
      // Single-line mode: find largest font where content does not wrap.
      // Compare height at container width vs height at huge width — equal means no wrap.
      let no_wrap_fits(text_size) = {
        let h_narrow = measure(width: size.width, render(text_size, max_leading)).height
        let h_wide = measure(width: 10000pt, block(width: 10000pt)[#{
          set text(size: text_size, weight: weight, style: style)
          set par(leading: max_leading)
          eval(markup, mode: "markup")
        }]).height
        h_narrow <= h_wide
      }
      let chosen_size = if no_wrap_fits(max_size) {
        max_size
      } else if not no_wrap_fits(min_size) {
        min_size
      } else {
        pdftr_fit_size(min_size, max_size, eps, size_pt => no_wrap_fits(size_pt))
      }
      render(chosen_size, max_leading)
    } else {
      let fits(text_size, leading) = measure(width: size.width, render(text_size, leading)).height <= allowed-height
      if fits(max_size, max_leading) {
        render(max_size, max_leading)
      } else {
        let fallback_min_size = pdftr_floor_size(min_size - 1.6pt, 5.4pt)
        let fallback_min_leading = pdftr_floor_leading(min_leading - 0.12em, 0.14em)
        let emergency_min_size = pdftr_floor_size(fallback_min_size - 1.2pt, 4.8pt)
        let emergency_min_leading = pdftr_floor_leading(fallback_min_leading - 0.08em, 0.10em)
        let chosen_leading = if fits(min_size, max_leading) { max_leading } else { min_leading }
        let chosen_size = if not fits(min_size, chosen_leading) {
          let fallback_leading = pdftr_floor_leading(chosen_leading - 0.12em, fallback_min_leading)
          let emergency_leading = pdftr_floor_leading(fallback_leading - 0.08em, emergency_min_leading)
          if not fits(fallback_min_size, fallback_leading) {
            if not fits(emergency_min_size, emergency_leading) {
              emergency_min_size
            } else {
              pdftr_fit_size(emergency_min_size, fallback_min_size, eps, size_pt => fits(size_pt, emergency_leading))
            }
          } else {
            pdftr_fit_size(fallback_min_size, min_size, eps, size_pt => fits(size_pt, fallback_leading))
          }
        } else {
          pdftr_fit_size(min_size, max_size, eps, size_pt => fits(size_pt, chosen_leading))
        }
        let final_leading = if fits(min_size, chosen_leading) {
          chosen_leading
        } else if fits(fallback_min_size, pdftr_floor_leading(chosen_leading - 0.12em, fallback_min_leading)) {
          pdftr_floor_leading(chosen_leading - 0.12em, fallback_min_leading)
        } else {
          emergency_min_leading
        }
        render(chosen_size, final_leading)
      }
    }
  })
}"""


def _rgb_typst(rgb: RGB) -> str:
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"


def _font_typst(font: str | list[str]) -> str:
    """Render Typst font expression. Accepts a single name or fallback chain."""
    if isinstance(font, str):
        return f'"{font}"'
    return "(" + ", ".join(f'"{f}"' for f in font) + ")"


def _style_for(label: str, cfg: RenderConfig) -> StyleSpec:
    return cfg.styles.get(style_key(label), cfg.default_style)


def _cover_rect(
    var: str, x0: float, y0: float, x1: float, y1: float, bg: RGB, pad: float
) -> str:
    w = max(4.0, x1 - x0 + 2 * pad)
    h = max(4.0, y1 - y0 + 2 * pad)
    dx = x0 - pad
    dy = y0 - pad
    return (
        f"#let {var}_cover = rect(width: {w:.2f}pt, height: {h:.2f}pt,"
        f" fill: {_rgb_typst(bg)}, stroke: none)\n"
        f"#context {{ place(top + left, dx: {dx:.2f}pt, dy: {dy:.2f}pt, {var}_cover) }}\n"
    )


def _text_block(
    var: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    markdown: str,
    font_size: float,
    min_font: float,
    weight: str,
    style_: str,
    text_color: RGB,
    font_family: str,
    expanded_w: float | None = None,
    valign: str = "top",
) -> str:
    w = max(4.0, expanded_w if expanded_w is not None else (x1 - x0))
    h = max(4.0, y1 - y0)
    # Ensure min_size <= max_size; otherwise the binary-search fit helper
    # gets lo > hi and behaves incorrectly.
    effective_min = min(min_font, font_size)
    escaped = escape_typst_string(markdown)
    fit_call = (
        f"pdftr_fit_markdown({var}_md,"
        f" max_size: {font_size:.2f}pt, min_size: {effective_min:.2f}pt,"
        f' weight: "{weight}", style: "{style_}")'
    )
    if valign == "bottom":
        content = f"align(bottom + left, {fit_call})"
    elif valign == "center":
        content = f"align(center + horizon, {fit_call})"
    else:
        content = fit_call
    return (
        f'#let {var}_md = "{escaped}"\n'
        f"#let {var}_body = block(width: {w:.2f}pt, height: {h:.2f}pt)[#{{\n"
        f"  set text(font: {_font_typst(font_family)}, fill: {_rgb_typst(text_color)})\n"
        f"  {content}\n"
        f"}}]\n"
        f"#context {{ place(top + left, dx: {x0:.2f}pt, dy: {y0:.2f}pt, {var}_body) }}\n"
    )


def _text_block_typst(
    var: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    typst_markup: str,
    font_size: float,
    min_font: float,
    weight: str,
    style_: str,
    text_color: RGB,
    font_family: str,
    no_wrap: bool = False,
    expanded_w: float | None = None,
    valign: str = "top",
) -> str:
    w = max(4.0, expanded_w if expanded_w is not None else (x1 - x0))
    h = max(4.0, y1 - y0)
    effective_min = min(min_font, font_size)
    escaped = escape_typst_string(typst_markup)
    no_wrap_arg = ", no_wrap: true" if no_wrap else ""
    fit_call = (
        f"pdftr_fit_typst({var}_tm,"
        f" max_size: {font_size:.2f}pt, min_size: {effective_min:.2f}pt,"
        f' weight: "{weight}", style: "{style_}"{no_wrap_arg})'
    )
    if valign == "center":
        content = f"align(center + horizon, {fit_call})"
    elif valign == "bottom":
        content = f"align(bottom + left, {fit_call})"
    else:
        content = fit_call
    return (
        f'#let {var}_tm = "{escaped}"\n'
        f"#let {var}_body = block(width: {w:.2f}pt, height: {h:.2f}pt)[#{{\n"
        f"  set text(font: {_font_typst(font_family)}, fill: {_rgb_typst(text_color)})\n"
        f"  {content}\n"
        f"}}]\n"
        f"#context {{ place(top + left, dx: {x0:.2f}pt, dy: {y0:.2f}pt, {var}_body) }}\n"
    )


_TOC_TOP_LEVEL_RE = re.compile(r"^\d+\s+\S")


def _toc_block(
    var: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    translated_text: str,
    font_size: float,
    min_font: float,
    text_color: RGB,
    font_family: str,
    rendered_pages: set[int] | None = None,
) -> str:
    """Render TOC entries with right-aligned page numbers + clickable links.

    Each entry whose target page is in `rendered_pages` (1-indexed) gets
    wrapped in `#link(<pdftr-page-N>)[...]` so clicking jumps to that page.

    Top-level entries (section number with no dot, e.g. '1 Introduction')
    render in bold; sub-entries stay regular.

    Auto-shrinks via `pdftr_fit_typst` when entries overflow the bbox.
    """
    entries = parse_toc_entries(translated_text)
    markup_lines: list[str] = []
    for title, page_num in entries:
        title_escaped = escape_typst_string(to_typst_markup(title))
        weight = "bold" if _TOC_TOP_LEVEL_RE.match(title) else "regular"
        if page_num:
            row = (
                f"grid(columns: (1fr, auto), gutter: 4pt, "
                f'text(weight: "{weight}", "{title_escaped} "), '
                f'align(right, text(weight: "{weight}", "{page_num}")))'
            )
            page_int = int(page_num)
            if rendered_pages is None or page_int in rendered_pages:
                markup_lines.append(f"#link(<pdftr-page-{page_int}>)[#{row}]")
            else:
                markup_lines.append(f"#{row}")
        else:
            markup_lines.append(f'#par(text(weight: "{weight}", "{title_escaped}"))')
    typst_markup = "\n".join(markup_lines)
    return _text_block_typst(
        var,
        x0,
        y0,
        x1,
        y1,
        typst_markup,
        font_size,
        min_font,
        "regular",
        "normal",
        text_color,
        font_family,
    )


def build_typst_source(
    parsed: dict,
    sizes: dict[str, float],
    bg_colors: dict[str, RGB],
    text_colors: dict[str, RGB],
    cfg: RenderConfig,
) -> str:
    lines: list[str] = [
        f"#set text(font: {_font_typst(cfg.font_family)})",
        f'#import "@preview/cmarker:{CMARKER_VERSION}"',
        _FIT_HELPERS,
    ]

    pages = parsed.get("pages", [])
    # 1-indexed page numbers that will appear in the output PDF — used to gate
    # TOC links so we don't emit links to pages that were filtered out.
    rendered_pages = {
        i + 1 for i in range(len(pages)) if cfg.pages is None or i in cfg.pages
    }
    for page_idx, page in enumerate(pages):
        if cfg.pages is not None and page_idx not in cfg.pages:
            continue
        pw = page.get("page_width", 595.0)
        ph = page.get("page_height", 842.0)
        lines.append(
            f"#set page(width: {pw:.2f}pt, height: {ph:.2f}pt, margin: 0pt, fill: none)"
        )
        # Anchor for TOC links — page number is 1-indexed (user-facing).
        lines.append(f"#metadata(none)<pdftr-page-{page_idx + 1}>")

        elems = page.get("elements", [])

        for elem_idx, elem in enumerate(elems):
            category = elem.get("category", "")
            label = normalize_label(elem.get("label", "Text"))
            uid = f"p{page_idx}:e{elem_idx}"
            var = f"e{page_idx}_{elem_idx}"

            if category == "BYPASS":
                continue

            bbox = elem.get("bbox_pdf", [0, 0, 100, 20])
            x0, y0, x1, y1 = bbox

            # Elements covering >50% of the page are likely cover images or
            # mis-detections — keep original, don't overlay.
            elem_w, elem_h = x1 - x0, y1 - y0
            if (elem_w * elem_h) / (pw * ph) >= 0.50:
                continue
            bg = bg_colors.get(uid, cfg.background.fallback_bg)
            tc = text_colors.get(uid, (0, 0, 0))
            style = _style_for(label, cfg)
            font_size = sizes.get(uid, cfg.sizing.fallback_size)

            if category == "EQUATION":
                translated = elem.get("translated_text") or ""
                source = elem.get("source_text") or ""
                if not translated or translated == source:
                    continue
                # Malformed / bare-LaTeX output → preserve original text layer.
                # For EQUATION, skip is_pure_math_text: translated_text is intentionally
                # Typst math markup and should be rendered even if it has no prose.
                if (
                    has_unbalanced_math_tags(translated)
                    or has_bare_latex(translated)
                    or has_malformed_typst_math(translated)
                ):
                    continue
                if category != "EQUATION" and is_pure_math_text(translated):
                    continue
                lines.append(
                    _cover_rect(
                        var, x0, y0, x1, y1, bg, cfg.background.eraser_padding_pt
                    )
                )
                # EQUATION elements that reach here always contain prose mixed
                # with math (pure-math equations have no translated_text and are
                # skipped above). Fractions make the bbox taller than the actual
                # text size, so use the cluster font_size, not y1 - y0.
                eq_max_size = font_size
                if "<math" in translated or "<typst" in translated:
                    typst_markup = to_typst_native(translated)
                    lines.append(
                        _text_block_typst(
                            var,
                            x0,
                            y0,
                            x1,
                            y1,
                            typst_markup,
                            eq_max_size,
                            cfg.min_font_size_pt,
                            style.weight,
                            style.style_,
                            tc,
                            cfg.font_family,
                            valign="center",
                        )
                    )
                else:
                    markdown = to_typst_markup(translated)
                    lines.append(
                        _text_block(
                            var,
                            x0,
                            y0,
                            x1,
                            y1,
                            markdown,
                            eq_max_size,
                            cfg.min_font_size_pt,
                            style.weight,
                            style.style_,
                            tc,
                            cfg.font_family,
                            valign="center",
                        )
                    )

            elif category == "TABLE":
                cells = elem.get("cells", [])
                for cell_idx, cell in enumerate(cells):
                    cell_translated = cell.get("translated_text") or ""
                    if not cell_translated:
                        continue
                    cell_uid = f"{uid}:c{cell_idx}"
                    cell_var = f"{var}_c{cell_idx}"
                    cbbox = cell.get("bbox_pdf", bbox)
                    cx0, cy0, cx1, cy1 = cbbox
                    cell_bg = bg_colors.get(cell_uid, bg)
                    cell_tc = text_colors.get(cell_uid, tc)
                    cell_size = sizes.get(cell_uid, font_size)
                    cell_md = to_typst_markup(cell_translated)
                    inset = cfg.sizing.cell_bbox_inset_pt
                    lines.append(
                        _cover_rect(
                            cell_var,
                            cx0,
                            cy0,
                            cx1,
                            cy1,
                            cell_bg,
                            cfg.background.eraser_padding_pt,
                        )
                    )
                    lines.append(
                        _text_block(
                            cell_var,
                            cx0 + inset,
                            cy0 + inset,
                            cx1 - inset,
                            cy1 - inset,
                            cell_md,
                            cell_size,
                            cfg.min_font_size_pt,
                            cfg.cell_style.weight,
                            cfg.cell_style.style_,
                            cell_tc,
                            cfg.font_family,
                        )
                    )

            else:  # FLOWING_TEXT, IN_PLACE
                translated = elem.get("translated_text") or ""
                source = elem.get("source_text") or ""
                if not translated:
                    continue
                # LLM returned text unchanged → pure math notation, keep original.
                if translated.strip() == source.strip():
                    continue
                # Pure math / malformed LLM / bare LaTeX outside tags →
                # preserve original PDF text layer.
                if (
                    is_pure_math_text(translated)
                    or has_unbalanced_math_tags(translated)
                    or has_bare_latex(translated)
                    or has_malformed_typst_math(translated)
                ):
                    continue

                lines.append(
                    _cover_rect(
                        var, x0, y0, x1, y1, bg, cfg.background.eraser_padding_pt
                    )
                )

                if label == "TableOfContents":
                    lines.append(
                        _toc_block(
                            var,
                            x0,
                            y0,
                            x1,
                            y1,
                            translated,
                            font_size,
                            cfg.min_font_size_pt,
                            tc,
                            cfg.font_family,
                            rendered_pages,
                        )
                    )
                elif "<typst" in translated or "<math" in translated:
                    typst_markup = to_typst_native(translated)
                    is_single_line = (y1 - y0) < font_size * 1.8
                    # Single-line: expand block width to available horizontal
                    # space so text flows right instead of wrapping down.
                    exp_w = (pw - x0) if is_single_line else None
                    lines.append(
                        _text_block_typst(
                            var,
                            x0,
                            y0,
                            x1,
                            y1,
                            typst_markup,
                            font_size,
                            cfg.min_font_size_pt,
                            style.weight,
                            style.style_,
                            tc,
                            cfg.font_family,
                            no_wrap=is_single_line,
                            expanded_w=exp_w,
                        )
                    )
                elif _BARE_TYPST_MATH.search(translated):
                    # Bare Typst math functions (no <math> tags) — wrap in $ and use native path
                    typst_markup = f"${_split_math_vars(translated)}$"
                    is_single_line = (y1 - y0) < font_size * 1.8
                    exp_w = (pw - x0) if is_single_line else None
                    lines.append(
                        _text_block_typst(
                            var,
                            x0,
                            y0,
                            x1,
                            y1,
                            typst_markup,
                            font_size,
                            cfg.min_font_size_pt,
                            style.weight,
                            style.style_,
                            tc,
                            cfg.font_family,
                            no_wrap=is_single_line,
                            expanded_w=exp_w,
                        )
                    )
                else:
                    # No <math> tags, no Typst functions — plain text/LaTeX, use cmarker/mitex
                    markdown = to_typst_markup(translated)
                    is_single_line = (y1 - y0) < font_size * 1.8
                    exp_w = (pw - x0) if is_single_line else None
                    lines.append(
                        _text_block(
                            var,
                            x0,
                            y0,
                            x1,
                            y1,
                            markdown,
                            font_size,
                            cfg.min_font_size_pt,
                            style.weight,
                            style.style_,
                            tc,
                            cfg.font_family,
                            expanded_w=exp_w,
                        )
                    )

        # Add pagebreak between pages (not after the last one)
        if page_idx < len(pages) - 1:
            lines.append("#pagebreak()")

    return "\n".join(lines) + "\n"
