from __future__ import annotations

from .config import SizingConfig

# Normalise hyphenated Surya labels to CamelCase (matches actual JSON output)
_NORMALISE: dict[str, str] = {
    "Section-header": "SectionHeader",
    "List-item": "ListItem",
    "Page-header": "PageHeader",
    "Page-footer": "PageFooter",
    "Table-of-contents": "TableOfContents",
    "Text-inline-math": "TextInlineMath",
    "Formula": "Equation",
    "Picture": "Figure",
}


def normalize_label(label: str) -> str:
    return _NORMALISE.get(label, label)


# Minor/structural labels whose box should never legitimately span most of a
# page. When one does, it is almost certainly a layout mis-detection, so we keep
# the original and do not translate/overlay it. Real content — body text
# (Text/ListItem), tables, equations, TOC — may legitimately be large and is
# never skipped by size. (PageHeader/PageFooter/Figure/Code are BYPASS and are
# already skipped upstream; listed here only to document intent.)
OVERSIZE_KEEP_ORIGINAL_LABELS: frozenset[str] = frozenset(
    {
        "SectionHeader",
        "PageHeader",
        "PageFooter",
        "Caption",
        "Footnote",
        "Title",
    }
)


def skip_oversize_element(
    label: str,
    bbox_pdf: list[float],
    page_width: float,
    page_height: float,
    threshold: float = 0.5,
) -> bool:
    """Whether a large element should be left as the original (not translated).

    Returns True only when the element's label is a minor/structural one
    (:data:`OVERSIZE_KEEP_ORIGINAL_LABELS`) *and* its box covers at least
    ``threshold`` of the page area. This single predicate is used by BOTH the
    overlay builder and the native-text redaction pass so they always agree on
    which elements to skip — a mismatch would erase content without redrawing it.
    """
    if normalize_label(label) not in OVERSIZE_KEEP_ORIGINAL_LABELS:
        return False
    x0, y0, x1, y1 = bbox_pdf
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    page_area = page_width * page_height
    return page_area > 0 and area >= threshold * page_area


def group_for_label(label: str, cfg: SizingConfig) -> str | None:
    norm = normalize_label(label)
    for group, members in cfg.cluster_groups.items():
        if norm in members:
            return group
    return None


def style_key(label: str) -> str:
    return normalize_label(label)
