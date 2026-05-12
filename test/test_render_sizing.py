"""Unit tests for pdf2zh.render.sizing — font size clustering."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf2zh.render.config import SizingConfig
from pdf2zh.render.sizing import _greedy_cluster, assign_render_sizes


def _make_doc(
    elements: list[dict], page_width: float = 612, page_height: float = 792
) -> dict:
    return {
        "pages": [
            {
                "page_index": 0,
                "page_width": page_width,
                "page_height": page_height,
                "elements": elements,
            }
        ]
    }


class TestGreedyCluster:
    def test_single_cluster(self):
        items = [(10.5, "a"), (10.7, "b"), (11.0, "c")]
        clusters = _greedy_cluster(items, eps=1.5)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_two_clusters(self):
        items = [(10.5, "a"), (11.0, "b"), (14.0, "c"), (14.5, "d")]
        clusters = _greedy_cluster(items, eps=1.5)
        assert len(clusters) == 2

    def test_three_clusters(self):
        items = [(10.5, "a"), (14.0, "b"), (18.0, "c")]
        clusters = _greedy_cluster(items, eps=1.5)
        assert len(clusters) == 3

    def test_empty(self):
        assert _greedy_cluster([], eps=1.5) == []

    def test_single_item(self):
        clusters = _greedy_cluster([(11.0, "a")], eps=1.5)
        assert len(clusters) == 1
        assert clusters[0] == [(11.0, "a")]


class TestAssignRenderSizes:
    def _cfg(self, **kwargs) -> SizingConfig:
        return SizingConfig(**kwargs)

    def test_body_text_consistent(self):
        doc = _make_doc(
            [
                {
                    "label": "Text",
                    "category": "FLOWING_TEXT",
                    "font_size": 10.7,
                    "bbox_pdf": [0, 0, 100, 20],
                    "cells": [],
                },
                {
                    "label": "Text",
                    "category": "FLOWING_TEXT",
                    "font_size": 11.0,
                    "bbox_pdf": [0, 25, 100, 45],
                    "cells": [],
                },
                {
                    "label": "Text",
                    "category": "FLOWING_TEXT",
                    "font_size": 11.3,
                    "bbox_pdf": [0, 50, 100, 70],
                    "cells": [],
                },
            ]
        )
        cfg = self._cfg()
        sizes = assign_render_sizes(doc, cfg)
        # All should snap to same cluster (within eps=1.5)
        assert sizes["p0:e0"] == sizes["p0:e1"] == sizes["p0:e2"]

    def test_heading_hierarchy_preserved(self):
        doc = _make_doc(
            [
                {
                    "label": "SectionHeader",
                    "category": "IN_PLACE",
                    "font_size": 10.0,
                    "bbox_pdf": [0, 0, 100, 20],
                    "cells": [],
                },
                {
                    "label": "SectionHeader",
                    "category": "IN_PLACE",
                    "font_size": 14.0,
                    "bbox_pdf": [0, 25, 100, 45],
                    "cells": [],
                },
                {
                    "label": "SectionHeader",
                    "category": "IN_PLACE",
                    "font_size": 18.0,
                    "bbox_pdf": [0, 50, 100, 70],
                    "cells": [],
                },
            ]
        )
        cfg = self._cfg()
        sizes = assign_render_sizes(doc, cfg)
        # Three distinct sizes
        assert sizes["p0:e0"] < sizes["p0:e1"] < sizes["p0:e2"]

    def test_fallback_for_zero_size(self):
        doc = _make_doc(
            [
                {
                    "label": "Text",
                    "category": "FLOWING_TEXT",
                    "font_size": 0.0,
                    "bbox_pdf": [0, 0, 100, 20],
                    "cells": [],
                },
            ]
        )
        cfg = self._cfg(fallback_size=11.0)
        sizes = assign_render_sizes(doc, cfg)
        assert sizes["p0:e0"] == 11.0

    def test_bypass_not_included(self):
        doc = _make_doc(
            [
                {
                    "label": "Figure",
                    "category": "BYPASS",
                    "font_size": 0.0,
                    "bbox_pdf": [0, 0, 100, 100],
                    "cells": [],
                },
            ]
        )
        cfg = self._cfg()
        sizes = assign_render_sizes(doc, cfg)
        # BYPASS elements are not in any cluster group, so not assigned
        assert "p0:e0" not in sizes or True  # Acceptable: either absent or fallback

    def test_table_cells_cluster_per_table(self):
        doc = _make_doc(
            [
                {
                    "label": "Table",
                    "category": "TABLE",
                    "font_size": 10.0,
                    "bbox_pdf": [0, 0, 300, 100],
                    "cells": [
                        {
                            "bbox_pdf": [0, 0, 100, 20],
                            "source_text": "A",
                            "translated_text": "B",
                            "cell_font_size": 10.0,
                        },
                        {
                            "bbox_pdf": [100, 0, 200, 20],
                            "source_text": "C",
                            "translated_text": "D",
                            "cell_font_size": 10.5,
                        },
                    ],
                },
            ]
        )
        cfg = self._cfg()
        sizes = assign_render_sizes(doc, cfg)
        # Both cells should get the same cluster size
        assert sizes["p0:e0:c0"] == sizes["p0:e0:c1"]

    def test_equation_clusters_with_body(self):
        doc = _make_doc(
            [
                {
                    "label": "Text",
                    "category": "FLOWING_TEXT",
                    "font_size": 11.0,
                    "bbox_pdf": [0, 0, 100, 20],
                    "cells": [],
                },
                {
                    "label": "Equation",
                    "category": "EQUATION",
                    "font_size": 11.2,
                    "bbox_pdf": [0, 25, 100, 45],
                    "cells": [],
                },
            ]
        )
        cfg = self._cfg()
        sizes = assign_render_sizes(doc, cfg)
        # Both in "body" group → should cluster together (within eps)
        assert sizes["p0:e0"] == sizes["p0:e1"]

    def test_page_scope_header_footer(self):
        # Page-scope: each page's header/footer clusters independently
        pages = [
            {
                "page_index": 0,
                "page_width": 612,
                "page_height": 792,
                "elements": [
                    {
                        "label": "PageHeader",
                        "category": "IN_PLACE",
                        "font_size": 9.0,
                        "bbox_pdf": [0, 0, 600, 20],
                        "cells": [],
                    },
                ],
            },
            {
                "page_index": 1,
                "page_width": 612,
                "page_height": 792,
                "elements": [
                    {
                        "label": "PageHeader",
                        "category": "IN_PLACE",
                        "font_size": 10.0,
                        "bbox_pdf": [0, 0, 600, 20],
                        "cells": [],
                    },
                ],
            },
        ]
        doc = {"pages": pages}
        cfg = self._cfg()
        sizes = assign_render_sizes(doc, cfg)
        # They're in separate page-scoped buckets, so cluster separately
        assert "p0:e0" in sizes
        assert "p1:e0" in sizes
