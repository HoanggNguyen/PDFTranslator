from __future__ import annotations

import math
import re

from .config import SizingConfig
from .labels import group_for_label, normalize_label

_TAG_RE = re.compile(r"<[^>]+>")


def _estimate_fit_size(
    text: str,
    bbox_w: float,
    bbox_h: float,
    cfg: SizingConfig,
) -> float:
    """Estimate the largest font size (pt) that fits `text` in a (bbox_w × bbox_h) box.

    Continuous model: height = (N * fs * char_width_ratio / W) * fs * leading_ratio
    Solving for fs: fs = sqrt(H * W / (N * cwr * lr))
    """
    n = max(1, len(_TAG_RE.sub("", text).strip()))
    fs = math.sqrt(bbox_h * bbox_w / (n * cfg.char_width_ratio * cfg.leading_ratio))
    return max(cfg.fallback_size, min(200.0, fs))


def assign_render_sizes(parsed: dict, cfg: SizingConfig) -> dict[str, float]:
    """Return {uid: font_size_pt} for every element and cell.

    Sizing strategy (in priority order):
      1. Estimate the fitting font size from translated text length + bbox dimensions.
      2. Cap at phase-1 cap height (phase1_line_height × cap_height_ratio) when available.
      3. Cluster the content-aware sizes within each label group for visual consistency.
      4. Snap each uid to the cluster maximum.

    uid format:
      "p{page_idx}:e{elem_idx}"              for elements
      "p{page_idx}:e{elem_idx}:c{cell_idx}"  for table cells
    """
    raw: dict[str, list[tuple[str, float]]] = {}  # bucket → [(uid, fitted_size)]

    for page_idx, page in enumerate(parsed.get("pages", [])):
        for elem_idx, elem in enumerate(page.get("elements", [])):
            uid = f"p{page_idx}:e{elem_idx}"
            category = elem.get("category", "")
            label = normalize_label(elem.get("label", ""))
            group = group_for_label(label, cfg)

            if category != "BYPASS" and group:
                phase1_fs = float(elem.get("font_size") or 0.0)
                cap = phase1_fs * cfg.cap_height_ratio if phase1_fs > 0 else 0.0

                translated = elem.get("translated_text") or ""
                bbox = elem.get("bbox_pdf", [0, 0, 10, 10])
                w = max(1.0, bbox[2] - bbox[0])
                h = max(1.0, bbox[3] - bbox[1])

                if translated.strip():
                    fitted = _estimate_fit_size(translated, w, h, cfg)
                    fs = min(cap, fitted) if cap > 0 else fitted
                else:
                    fs = cap if cap > 0 else cfg.fallback_size

                scope = cfg.cluster_scope_by_group.get(group, "document")
                scope_key = "doc" if scope == "document" else str(page_idx)
                raw.setdefault(f"{group}|{scope_key}", []).append((uid, fs))

            # TABLE cells: cluster per-table
            cells = elem.get("cells", [])
            if cells:
                table_bucket = f"table|{uid}"
                parent_bbox = elem.get("bbox_pdf", [0, 0, 10, 10])
                for cell_idx, cell in enumerate(cells):
                    cell_uid = f"{uid}:c{cell_idx}"
                    cs_phase1 = float(cell.get("cell_font_size") or 0.0)
                    cap = cs_phase1 * cfg.cap_height_ratio if cs_phase1 > 0 else 0.0

                    cell_text = cell.get("translated_text") or ""
                    cbbox = cell.get("bbox_pdf", parent_bbox)
                    cw = max(1.0, cbbox[2] - cbbox[0])
                    ch = max(1.0, cbbox[3] - cbbox[1])

                    if cell_text.strip():
                        fitted = _estimate_fit_size(cell_text, cw, ch, cfg)
                        cs = min(cap, fitted) if cap > 0 else fitted
                    else:
                        cs = cap if cap > 0 else cfg.fallback_size

                    raw.setdefault(table_bucket, []).append((cell_uid, cs))

    # ---- cluster and snap ----
    result: dict[str, float] = {}

    for _, items in raw.items():
        valid = [(uid, s) for uid, s in items if s > 0]
        fallback = cfg.fallback_size

        if not valid:
            for uid, _ in items:
                result[uid] = fallback
            continue

        clusters = _greedy_cluster([(s, uid) for uid, s in valid], cfg.cluster_eps_pt)
        uid_to_size: dict[str, float] = {}
        for cluster in clusters:
            cluster_size = max(s for s, _ in cluster)
            for _, uid in cluster:
                uid_to_size[uid] = cluster_size

        for uid, _ in items:
            result[uid] = uid_to_size.get(uid, fallback)

    return result


def _greedy_cluster(
    items: list[tuple[float, str]], eps: float
) -> list[list[tuple[float, str]]]:
    """1-D greedy binning: extend cluster while next size <= cluster_max + eps."""
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
