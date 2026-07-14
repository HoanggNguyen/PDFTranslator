"""Unit tests for table-cell rendering decisions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf2zh.render.config import RenderConfig
from pdf2zh.render.source_builder import build_typst_source


def test_table_cell_without_source_text_is_not_rendered():
    parsed = {
        "pages": [
            {
                "page_width": 200,
                "page_height": 100,
                "elements": [
                    {
                        "category": "TABLE",
                        "label": "Table",
                        "bbox_pdf": [0, 0, 200, 100],
                        "cells": [
                            {
                                "bbox_pdf": [0, 0, 100, 20],
                                "source_text": "Source",
                                "translated_text": "Translated",
                            },
                            {
                                "bbox_pdf": [100, 0, 200, 20],
                                "translated_text": "Must remain original",
                            },
                        ],
                    }
                ],
            }
        ]
    }

    source = build_typst_source(
        parsed,
        {"p0:e0:c0": 10, "p0:e0:c1": 10},
        {},
        {},
        RenderConfig(),
    )

    assert "Translated" in source
    assert "Must remain original" not in source


def test_native_typst_content_is_embedded_without_eval():
    parsed = {
        "pages": [
            {
                "page_width": 200,
                "page_height": 100,
                "elements": [
                    {
                        "category": "EQUATION",
                        "label": "Equation",
                        "bbox_pdf": [0, 0, 100, 20],
                        "source_text": "Source",
                        "translated_text": "<math>arrow.r.double</math>",
                    }
                ],
            }
        ]
    }

    source = build_typst_source(parsed, {"p0:e0": 10}, {}, {}, RenderConfig())

    assert 'eval(markup, mode: "markup")' not in source
    assert "#let e0_0_tm = [$arrow.r.double$]" in source
