"""Enums and label mappings for the scanned PDF pipeline.

This module contains:
- ElementCategory: The 5 translation handling categories
- SuryaLabel: String constants for Surya's hyphenated labels
- SURYA_LABEL_MAP: Mapping from Surya labels to ElementCategory
- DEFAULT_CATEGORY: Fallback for unknown labels
"""

from enum import Enum


class ElementCategory(str, Enum):
    """Categories determining how downstream stages handle each element.

    Values:
        BYPASS: Pixel-copy from original; never translate (Picture, Figure, Form)
        FLOWING_TEXT: Translate full source_text; may merge adjacent blocks
        IN_PLACE: Translate source_text; render at exact bbox position
        TABLE: Translate each cell's source_text; render cell grid
        EQUATION: source_text contains any surrounding text to translate;
            latex holds a placeholder or recognized LaTeX marking math position
    """

    BYPASS = "BYPASS"
    FLOWING_TEXT = "FLOWING_TEXT"
    IN_PLACE = "IN_PLACE"
    TABLE = "TABLE"
    EQUATION = "EQUATION"


class SuryaLabel:
    """String constants for Surya layout labels (v0.9+ hyphenated format).

    These match the exact strings returned by Surya's LayoutPredictor.
    """

    # Text elements -> FLOWING_TEXT
    TEXT = "Text"
    LIST_ITEM = "ListItem"
    FOOTNOTE = "Footnote"

    # Headers/footers/captions -> IN_PLACE
    SECTION_HEADER = "SectionHeader"
    PAGE_HEADER = "PageHeader"
    PAGE_FOOTER = "PageFooter"
    CAPTION = "Caption"
    TABLE_OF_CONTENTS = "TableOfContents"

    # Graphics -> BYPASS
    PICTURE = "Picture"
    FIGURE = "Figure"
    FORM = "Form"

    # Tables -> TABLE
    TABLE = "Table"

    # Math -> EQUATION
    EQUATION = "Equation"

    # Code
    CODE = "Code"


# Mapping from Surya labels to ElementCategory
SURYA_LABEL_MAP: dict[str, ElementCategory] = {
    # FLOWING_TEXT: regular text blocks that can be translated and reflowed
    SuryaLabel.TEXT: ElementCategory.FLOWING_TEXT,
    SuryaLabel.LIST_ITEM: ElementCategory.FLOWING_TEXT,
    SuryaLabel.FOOTNOTE: ElementCategory.FLOWING_TEXT,
    # IN_PLACE: text that must be rendered at exact position
    SuryaLabel.SECTION_HEADER: ElementCategory.IN_PLACE,
    SuryaLabel.PAGE_HEADER: ElementCategory.IN_PLACE,
    SuryaLabel.PAGE_FOOTER: ElementCategory.IN_PLACE,
    SuryaLabel.CAPTION: ElementCategory.IN_PLACE,
    SuryaLabel.TABLE_OF_CONTENTS: ElementCategory.IN_PLACE,
    # BYPASS: graphics that should be copied without modification
    SuryaLabel.PICTURE: ElementCategory.BYPASS,
    SuryaLabel.FIGURE: ElementCategory.BYPASS,
    SuryaLabel.FORM: ElementCategory.BYPASS,
    # TABLE: structured data requiring cell-level translation
    SuryaLabel.TABLE: ElementCategory.TABLE,
    # EQUATION: math content to be preserved as-is
    SuryaLabel.EQUATION: ElementCategory.EQUATION,
    # CODE: programming code blocks (treat as IN_PLACE for now)
    SuryaLabel.CODE: ElementCategory.BYPASS,
}

# Default category for unknown Surya labels
DEFAULT_CATEGORY = ElementCategory.FLOWING_TEXT
