from __future__ import annotations

from .background import RGB
from .config import RenderConfig, StyleSpec
from .labels import normalize_label, style_key
from .markup import escape_typst_string, parse_toc_line, to_typst_markup

CMARKER_VERSION = "0.1.8"
MITEX_VERSION = "0.2.6"

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
#let pdftr_fit_markdown(markdown, max_size: 10pt, min_size: 9pt, max_leading: 0.66em, min_leading: 0.54em, fit_height: none, weight: "regular", style: "normal", eps: 0.08pt) = {
  layout(size => {
    let allowed-height = if fit_height == none { size.height } else { calc.min(size.height, fit_height) }
    let render(text_size, leading) = block(width: size.width)[#{
      set text(size: text_size, weight: weight, style: style)
      set par(leading: leading)
      cmarker.render(markdown, math: mitex)
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
}"""


def _rgb_typst(rgb: RGB) -> str:
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"


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
    x0: float, y0: float, x1: float, y1: float,
    markdown: str,
    font_size: float,
    min_font: float,
    weight: str,
    style_: str,
    text_color: RGB,
    font_family: str,
) -> str:
    w = max(4.0, x1 - x0)
    h = max(4.0, y1 - y0)
    escaped = escape_typst_string(markdown)
    min_size = min_font
    return (
        f'#let {var}_md = "{escaped}"\n'
        f"#let {var}_body = block(width: {w:.2f}pt, height: {h:.2f}pt)[#{{\n"
        f'  set text(font: "{font_family}", fill: {_rgb_typst(text_color)})\n'
        f"  pdftr_fit_markdown({var}_md,"
        f" max_size: {font_size:.2f}pt, min_size: {min_size:.2f}pt,"
        f' weight: "{weight}", style: "{style_}")\n'
        f"}}]\n"
        f"#context {{ place(top + left, dx: {x0:.2f}pt, dy: {y0:.2f}pt, {var}_body) }}\n"
    )


def _toc_block(
    var: str,
    x0: float, y0: float, x1: float, y1: float,
    translated_text: str,
    font_size: float,
    min_font: float,
    text_color: RGB,
    font_family: str,
) -> str:
    """Render TOC lines with right-aligned page numbers."""
    lines = translated_text.split("\n")
    w = max(4.0, x1 - x0)
    color_str = _rgb_typst(text_color)
    parts = [
        f"#let {var}_toc = block(width: {w:.2f}pt)[#{{",
        f'  set text(font: "{font_family}", size: {font_size:.2f}pt, fill: {color_str})',
        "  set par(leading: 0.6em)",
    ]
    for i, line in enumerate(lines):
        parsed = parse_toc_line(line)
        if parsed:
            title, page_num = parsed
            title_escaped = escape_typst_string(to_typst_markup(title))
            parts.append(
                f'  grid(columns: (1fr, auto), gutter: 4pt,'
                f' "{title_escaped} ", align(right, "{page_num}"))'
            )
        else:
            escaped = escape_typst_string(to_typst_markup(line))
            parts.append(f'  par["{escaped}"]')
    parts.append("}]")
    parts.append(
        f"#context {{ place(top + left, dx: {x0:.2f}pt, dy: {y0:.2f}pt, {var}_toc) }}"
    )
    return "\n".join(parts) + "\n"


def build_typst_source(
    parsed: dict,
    sizes: dict[str, float],
    bg_colors: dict[str, RGB],
    text_colors: dict[str, RGB],
    cfg: RenderConfig,
) -> str:
    lines: list[str] = [
        f'#set text(font: "{cfg.font_family}")',
        f"#import \"@preview/cmarker:{CMARKER_VERSION}\"",
        f'#import "@preview/mitex:{MITEX_VERSION}": mitex',
        "#show math.equation.where(block: false): set math.frac(style: \"horizontal\")",
        _FIT_HELPERS,
    ]

    pages = parsed.get("pages", [])
    for page_idx, page in enumerate(pages):
        if cfg.pages is not None and page_idx not in cfg.pages:
            continue
        pw = page.get("page_width", 595.0)
        ph = page.get("page_height", 842.0)
        lines.append(
            f"#set page(width: {pw:.2f}pt, height: {ph:.2f}pt, margin: 0pt, fill: none)"
        )

        for elem_idx, elem in enumerate(page.get("elements", [])):
            category = elem.get("category", "")
            label = normalize_label(elem.get("label", "Text"))
            uid = f"p{page_idx}:e{elem_idx}"
            var = f"e{page_idx}_{elem_idx}"

            if category == "BYPASS":
                continue

            bbox = elem.get("bbox_pdf", [0, 0, 100, 20])
            x0, y0, x1, y1 = bbox
            bg = bg_colors.get(uid, cfg.background.fallback_bg)
            tc = text_colors.get(uid, (0, 0, 0))
            style = _style_for(label, cfg)
            font_size = sizes.get(uid, cfg.sizing.fallback_size)

            if category == "EQUATION":
                translated = elem.get("translated_text") or ""
                source = elem.get("source_text") or ""
                # Skip if LLM returned unchanged (pure math)
                if not translated or translated == source:
                    continue
                markdown = to_typst_markup(translated, is_equation=True)
                lines.append(_cover_rect(var, x0, y0, x1, y1, bg, cfg.background.eraser_padding_pt))
                lines.append(_text_block(var, x0, y0, x1, y1, markdown, font_size,
                                         cfg.min_font_size_pt, style.weight, style.style_,
                                         tc, cfg.font_family))

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
                    lines.append(_cover_rect(cell_var, cx0, cy0, cx1, cy1, cell_bg,
                                             cfg.background.eraser_padding_pt))
                    lines.append(_text_block(cell_var, cx0, cy0, cx1, cy1, cell_md,
                                             cell_size, cfg.min_font_size_pt,
                                             cfg.cell_style.weight, cfg.cell_style.style_,
                                             cell_tc, cfg.font_family))

            else:  # FLOWING_TEXT, IN_PLACE
                translated = elem.get("translated_text") or ""
                if not translated:
                    continue

                lines.append(_cover_rect(var, x0, y0, x1, y1, bg, cfg.background.eraser_padding_pt))

                if label == "TableOfContents":
                    lines.append(_toc_block(var, x0, y0, x1, y1, translated,
                                            font_size, cfg.min_font_size_pt,
                                            tc, cfg.font_family))
                else:
                    markdown = to_typst_markup(translated)
                    lines.append(_text_block(var, x0, y0, x1, y1, markdown, font_size,
                                             cfg.min_font_size_pt, style.weight, style.style_,
                                             tc, cfg.font_family))

        # Add pagebreak between pages (not after the last one)
        if page_idx < len(pages) - 1:
            lines.append("#pagebreak()")

    return "\n".join(lines) + "\n"
