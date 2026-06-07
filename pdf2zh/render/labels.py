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


def group_for_label(label: str, cfg: SizingConfig) -> str | None:
    norm = normalize_label(label)
    for group, members in cfg.cluster_groups.items():
        if norm in members:
            return group
    return None


def style_key(label: str) -> str:
    return normalize_label(label)
