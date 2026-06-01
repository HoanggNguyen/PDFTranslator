from __future__ import annotations

import math
import re

from .config import SizingConfig
from .labels import group_for_label, normalize_label

_TAG_RE = re.compile(r"<[^>]+>")
_TYPST_BLOCK_RE = re.compile(r"<typst\b[^>]*>.*?</typst>", re.DOTALL | re.IGNORECASE)


def _autofit(text: str, bbox_w: float, bbox_h: float, cfg: SizingConfig) -> float:
    """Binary-search the largest font_size where `text` fits in (bbox_w × bbox_h).

    More accurate than the closed-form sqrt model because it correctly handles
    multi-line wrapping for both short texts (may only need 1 line) and long
    texts (may need many lines).
    """
    n = max(1, len(_TAG_RE.sub("", text).strip()))
    lo, hi = 4.0, bbox_h  # upper bound: can't exceed bbox height

    for _ in range(24):  # converges to ~0.001pt precision
        mid = (lo + hi) / 2.0
        chars_per_line = max(1.0, bbox_w / (mid * cfg.char_width_ratio))
        n_lines = math.ceil(n / chars_per_line)
        needed_h = n_lines * mid * cfg.leading_ratio
        if needed_h <= bbox_h:
            lo = mid
        else:
            hi = mid

    # Additional cap: autofit should never exceed cap_height_ratio × bbox_h
    # (single-line glyph height is always a fraction of bbox height).
    return min(lo, bbox_h * cfg.cap_height_ratio)


def _estimate_height(text: str, bbox_w: float, font_size: float, cfg: SizingConfig) -> float:
    """Estimate rendered height of text at font_size in a bbox_w-wide column."""
    n = max(1, len(_TAG_RE.sub("", text).strip()))
    chars_per_line = max(1.0, bbox_w / (font_size * cfg.char_width_ratio))
    n_lines = math.ceil(n / chars_per_line)
    return n_lines * font_size * cfg.leading_ratio


def _overflow_collides(
    bbox: list[float],
    text: str,
    font_size: float,
    cfg: SizingConfig,
    other_bboxes: list[list[float]],
) -> bool:
    """True if text at font_size overflows bbox AND that overflow region hits another element.

    Single-line elements (h < 2× font_size) overflow horizontally to the right;
    multi-line elements overflow vertically downward.
    """
    x0, y0, x1, y1 = bbox
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    n = max(1, len(_TAG_RE.sub("", text).strip()))

    if h < font_size * 2.0:
        # Single-line: text extends to the right rather than wrapping down.
        natural_w = n * font_size * cfg.char_width_ratio
        if natural_w <= w:
            return False
        # Overflow zone: horizontal strip to the right of the bbox.
        ov_x1 = x0 + natural_w
        for ob in other_bboxes:
            ox0, oy0, ox1, oy1 = ob
            if ox0 < ov_x1 and ox1 > x1 and oy0 < y1 and oy1 > y0:
                return True
        return False
    else:
        # Multi-line: text wraps and extends downward.
        needed_h = _estimate_height(text, w, font_size, cfg)
        if needed_h <= h:
            return False
        ov_y0, ov_y1 = y1, y0 + needed_h
        for ob in other_bboxes:
            ox0, oy0, ox1, oy1 = ob
            if ox0 < x1 and ox1 > x0 and oy0 < ov_y1 and oy1 > ov_y0:
                return True
        return False


def assign_render_sizes(parsed: dict, cfg: SizingConfig) -> dict[str, float]:
    """Return {uid: font_size_pt} for every element and cell.

    Strategy:
      1. Cluster source_text autofits per label+page → source_canonical (the
         representative size for that group, reflecting original layout intent).
      2. For each element: use source_canonical unless translated text overflows
         AND the overflow region collides with another element on the same page.
         Harmless overflow (into empty space) is allowed to preserve uniformity.
         Table cells always use MIN cap since adjacent cells always collide.

    uid format:
      "p{page_idx}:e{elem_idx}"              for elements
      "p{page_idx}:e{elem_idx}:c{cell_idx}"  for table cells
    """
    # bucket → [(uid, source_autofit)]
    raw: dict[str, list[tuple[str, float]]] = {}
    # uid → translated_text autofit ceiling
    translated_ceiling: dict[str, float] = {}
    # uid → {page_idx, bbox, translated} for collision check
    elem_meta: dict[str, dict] = {}
    # page_idx → all element bboxes on that page (for collision detection)
    page_all_bboxes: dict[int, list[list[float]]] = {}

    for page_idx, page in enumerate(parsed.get("pages", [])):
        all_bboxes: list[list[float]] = []
        for elem in page.get("elements", []):
            bbox = elem.get("bbox_pdf")
            if bbox:
                all_bboxes.append(bbox)
        page_all_bboxes[page_idx] = all_bboxes

        for elem_idx, elem in enumerate(page.get("elements", [])):
            uid = f"p{page_idx}:e{elem_idx}"
            category = elem.get("category", "")
            label = normalize_label(elem.get("label", ""))
            group = group_for_label(label, cfg)

            if category != "BYPASS" and group:
                source = elem.get("source_text") or ""
                translated = elem.get("translated_text") or ""
                bbox = elem.get("bbox_pdf", [0, 0, 10, 10])
                w = max(1.0, bbox[2] - bbox[0])
                h = max(1.0, bbox[3] - bbox[1])

                src_fs = (
                    _autofit(source, w, h, cfg) if source.strip() else cfg.fallback_size
                )
                # <typst> blocks contain grid layout syntax — their char count is
                # meaningless for autofit. Let Typst engine determine the size.
                if _TYPST_BLOCK_RE.search(translated):
                    t_fs = cfg.fallback_size
                elif translated.strip():
                    t_fs = _autofit(translated, w, h, cfg)
                else:
                    t_fs = cfg.fallback_size

                translated_ceiling[uid] = t_fs
                elem_meta[uid] = {
                    "page_idx": page_idx,
                    "bbox": bbox,
                    "translated": translated,
                }
                scope = cfg.cluster_scope_by_group.get(group, "page")
                scope_key = "doc" if scope == "document" else str(page_idx)
                raw.setdefault(f"{group}|{scope_key}", []).append((uid, src_fs))

            # TABLE cells: cluster per table.
            cells = elem.get("cells", [])
            if cells:
                table_bucket = f"table|{uid}"
                parent_bbox = elem.get("bbox_pdf", [0, 0, 10, 10])
                for cell_idx, cell in enumerate(cells):
                    cell_uid = f"{uid}:c{cell_idx}"
                    cell_source = cell.get("source_text") or ""
                    cell_translated = cell.get("translated_text") or ""
                    cbbox = cell.get("bbox_pdf", parent_bbox)
                    cw = max(1.0, cbbox[2] - cbbox[0])
                    ch = max(1.0, cbbox[3] - cbbox[1])

                    src_cs = (
                        _autofit(cell_source, cw, ch, cfg)
                        if cell_source.strip()
                        else cfg.fallback_size
                    )
                    t_cs = (
                        _autofit(cell_translated, cw, ch, cfg)
                        if cell_translated.strip()
                        else cfg.fallback_size
                    )
                    translated_ceiling[cell_uid] = t_cs
                    raw.setdefault(table_bucket, []).append((cell_uid, src_cs))

    # ---- cluster on source, assign per-element sizes ----
    result: dict[str, float] = {}

    for bucket, items in raw.items():
        valid = [(uid, s) for uid, s in items if s > 0]
        fallback = cfg.fallback_size
        is_table = bucket.startswith("table|")

        if not valid:
            for uid, _ in items:
                result[uid] = fallback
            continue

        clusters = _greedy_cluster([(s, uid) for uid, s in valid], cfg.cluster_eps_pt)
        best_cluster = max(clusters, key=lambda c: (len(c), _median([s for s, _ in c])))
        source_canonical = _median([s for s, _ in best_cluster])

        if is_table:
            # Table cells are always adjacent — overflow always collides.
            # Use MIN translated ceiling so no cell ever overflows into its neighbor.
            t_ceilings = [translated_ceiling.get(uid, fallback) for uid, _ in items]
            canonical = min(source_canonical, min(t_ceilings)) * cfg.cell_font_scale
            canonical = max(max(2.0, fallback * 0.5), canonical)
            for uid, _ in items:
                result[uid] = canonical
        else:
            # Non-table: use source_canonical for all elements (most popular size).
            # Only reduce for elements whose overflow would collide with another element.
            floor = fallback
            for uid, _ in items:
                t_ceiling = translated_ceiling.get(uid, fallback)
                if t_ceiling >= source_canonical:
                    # Translated text fits at source_canonical — no overflow.
                    result[uid] = source_canonical
                else:
                    # Translated text overflows. Allow it only if the overflow
                    # region doesn't collide with any other element on the page.
                    meta = elem_meta.get(uid, {})
                    page_idx = meta.get("page_idx", -1)
                    bbox = meta.get("bbox", [0, 0, 10, 10])
                    translated = meta.get("translated", "")
                    others = [
                        b for b in page_all_bboxes.get(page_idx, [])
                        if b is not bbox
                    ]
                    if _overflow_collides(bbox, translated, source_canonical, cfg, others):
                        result[uid] = max(floor, min(source_canonical, t_ceiling))
                    else:
                        result[uid] = source_canonical

    return result


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _greedy_cluster(
    items: list[tuple[float, str]], eps: float
) -> list[list[tuple[float, str]]]:
    """1-D greedy binning: extend current cluster while next value ≤ cluster_max + eps."""
    if not items:
        return []
    items = sorted(items, key=lambda x: x[0])
    clusters: list[list[tuple[float, str]]] = [[items[0]]]
    for size, uid in items[1:]:
        cur_max = max(s for s, _ in clusters[-1])
        if size <= cur_max + eps:
            clusters[-1].append((size, uid))
        else:
            clusters.append([(size, uid)])
    return clusters
