from __future__ import annotations

import math
import re

from .config import SizingConfig
from .labels import group_for_label, normalize_label

_TAG_RE = re.compile(r"<[^>]+>")


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


def assign_render_sizes(parsed: dict, cfg: SizingConfig) -> dict[str, float]:
    """Return {uid: font_size_pt} for every element and cell.

    Two-pass strategy:
      1. Cluster on source_text autofit: binary-search the largest font_size
         where source_text fits. Cluster by label+page to find the median of
         the largest cluster — this reflects the original layout intent.
      2. Overflow guard: compute translated_text autofit per element. If any
         element in the bucket overflows at the source canonical, lower the
         canonical uniformly for the whole bucket (same label, same page).
         All elements in a bucket share one final size.

    uid format:
      "p{page_idx}:e{elem_idx}"              for elements
      "p{page_idx}:e{elem_idx}:c{cell_idx}"  for table cells
    """
    # bucket → [(uid, source_autofit)]
    raw: dict[str, list[tuple[str, float]]] = {}
    # uid → translated_text autofit ceiling (overflow check)
    translated_ceiling: dict[str, float] = {}

    for page_idx, page in enumerate(parsed.get("pages", [])):
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

                src_fs = _autofit(source, w, h, cfg) if source.strip() else cfg.fallback_size
                t_fs = _autofit(translated, w, h, cfg) if translated.strip() else cfg.fallback_size

                translated_ceiling[uid] = t_fs
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

                    src_cs = _autofit(cell_source, cw, ch, cfg) if cell_source.strip() else cfg.fallback_size
                    t_cs = _autofit(cell_translated, cw, ch, cfg) if cell_translated.strip() else cfg.fallback_size
                    translated_ceiling[cell_uid] = t_cs
                    raw.setdefault(table_bucket, []).append((cell_uid, src_cs))

    # ---- cluster on source, cap by min translated ceiling in bucket ----
    result: dict[str, float] = {}

    for bucket, items in raw.items():
        valid = [(uid, s) for uid, s in items if s > 0]
        fallback = cfg.fallback_size

        if not valid:
            for uid, _ in items:
                result[uid] = fallback
            continue

        clusters = _greedy_cluster([(s, uid) for uid, s in valid], cfg.cluster_eps_pt)
        best_cluster = max(clusters, key=lambda c: (len(c), _median([s for s, _ in c])))
        source_canonical = _median([s for s, _ in best_cluster])

        # Cap by translated ceiling. For table cells: use MIN so any cell that
        # can't fit pulls down the whole table — no cell overflows.
        # For other groups: use median so a single long outlier doesn't shrink everyone.
        t_ceilings = [translated_ceiling.get(uid, fallback) for uid, _ in items]
        is_table = bucket.startswith("table|")
        translated_cap = min(t_ceilings) if is_table else _median(t_ceilings)
        canonical = min(source_canonical, translated_cap)

        # Table cells get a slight reduction to avoid crowding cell borders.
        if is_table:
            canonical = canonical * cfg.cell_font_scale

        # Absolute floor to avoid zero/negative; the Typst fit helper still
        # auto-shrinks per element down to its own emergency min (~4.8pt).
        # We use `fallback` as a floor only for non-table buckets where the
        # cluster mode is meant to look uniform; tables get a low floor so
        # individual cramped cells can actually shrink.
        floor = max(2.0, fallback * 0.5) if is_table else fallback
        canonical = max(floor, canonical)

        for uid, _ in items:
            result[uid] = canonical

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
