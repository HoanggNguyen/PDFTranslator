import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from pdf2zh.json_translator import (
    Gateway,
    TranslatorConfig,
    collect_translatables,
    glossary_block_for_chunk,
    is_equation_only,
    is_plain_text,
    segments_to_chunks,
    translate_document,
)

FIXTURE = Path(__file__).parent / "fixtures" / "mini_output.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _mock_cfg() -> TranslatorConfig:
    return TranslatorConfig(
        source_language="English",
        target_language="Vietnamese",
        provider="openrouter",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="google/gemini-2.5-flash-lite",
        concurrent=5,
        chunk_bytes=3000,
        glossary_enabled=False,
        retry=0,
        timeout=10,
    )


def _echo_translations(system: str, user: str, *, force_json: bool = False) -> str:
    """Side-effect for AsyncMock: echoes back each source text as <TR:source>."""
    match = re.search(r"```json\n(.*?)\n```", user, re.DOTALL)
    if not match:
        return "[]"
    chunk = json.loads(match.group(1))
    result = [{"id": k, "t": f"<TR:{v}>"} for k, v in chunk.items()]
    return json.dumps(result)


# ── 1. is_plain_text ────────────────────────────────────────────────────────────

def test_is_plain_text():
    assert is_plain_text("when a = b") is True
    assert is_plain_text("Introduction to Machine Learning") is True
    assert is_plain_text("a = b") is False          # no run of ≥2 letters
    assert is_plain_text("42.5") is False
    assert is_plain_text("<math>x</math>") is False
    assert is_plain_text("<math>x</math> where x is the count") is True
    assert is_plain_text("") is False


# ── 2. is_equation_only ─────────────────────────────────────────────────────────

def test_is_equation_only():
    assert is_equation_only("<math>P(x)</math> (1.1)") is True
    assert is_equation_only("<math>P(x)</math>") is True
    assert is_equation_only("<math>P(x)</math> where P is") is False
    assert is_equation_only("Some plain text") is False
    assert is_equation_only("") is True   # empty → only whitespace after strip


# ── 3. segments_to_chunks ───────────────────────────────────────────────────────

def test_chunking():
    doc = load_fixture()
    tasks = collect_translatables(doc)
    chunks = segments_to_chunks(tasks, max_bytes=200)

    # Every task id appears in exactly one chunk
    all_ids_in_chunks: list[str] = []
    for chunk in chunks:
        all_ids_in_chunks.extend(chunk.keys())
    assert sorted(all_ids_in_chunks) == sorted(t.id for t in tasks)

    # No chunk exceeds budget (unless a single segment is oversized on its own)
    for chunk in chunks:
        if len(chunk) > 1:
            size = len(json.dumps(chunk, ensure_ascii=False).encode())
            assert size <= 200

    # IDs are stringified indices starting from "0"
    expected_ids = [str(i) for i in range(len(tasks))]
    assert all_ids_in_chunks == expected_ids


# ── 4. collect_translatables ─────────────────────────────────────────────────────

def test_collect_translatables_on_sample():
    doc = load_fixture()
    tasks = collect_translatables(doc)

    ids = [t.id for t in tasks]
    write_keys = [t.write_key for t in tasks]
    texts = [t.text for t in tasks]

    # Expected: 6 tasks
    # id=0: source_text of SectionHeader
    # id=1: source_text of Text element
    # id=2: source_text of Caption (BYPASS picture skipped, equation-only skipped)
    # id=3: latex of Caption ("Result table")
    # id=4: cells[0].text of Caption ("Accuracy")
    # id=5: source_text of page 1 Text
    assert len(tasks) == 6
    assert ids == ["0", "1", "2", "3", "4", "5"]

    assert texts[0] == "Introduction to Machine Learning"
    assert write_keys[0] == "translated_text"

    assert texts[2] == "Table showing experimental results"
    assert write_keys[2] == "translated_text"

    assert texts[3] == "Result table"
    assert write_keys[3] == "translated_latex"

    assert texts[4] == "Accuracy"
    assert write_keys[4] == "translated_text"

    assert texts[5] == "CeADAR is a research center located in Dublin Ireland."
    assert write_keys[5] == "translated_text"

    # BYPASS element source_text not included
    assert not any(t.text == "" for t in tasks)
    # Equation-only not included
    assert not any("<math>P(x)" in t.text for t in tasks)
    # "42.5" not included (is_plain_text → False)
    assert not any(t.text == "42.5" for t in tasks)


# ── 5. end-to-end with mocked Gateway.call ──────────────────────────────────────

def test_translate_document_end_to_end_mocked():
    doc = load_fixture()
    cfg = _mock_cfg()

    mock_call = AsyncMock(side_effect=_echo_translations)
    with patch.object(Gateway, "call", mock_call):
        out = translate_document(doc, cfg)

    pages = out["pages"]
    elems0 = pages[0]["elements"]
    elems1 = pages[1]["elements"]

    # Eligible source_text fields are translated
    assert elems0[0]["translated_text"] == "<TR:Introduction to Machine Learning>"
    assert elems0[1]["translated_text"] == "<TR:This is a sample paragraph with enough text to be translatable.>"
    assert elems0[4]["translated_text"] == "<TR:Table showing experimental results>"
    assert elems1[0]["translated_text"] == "<TR:CeADAR is a research center located in Dublin Ireland.>"

    # translated_latex added as new sibling
    assert elems0[4]["translated_latex"] == "<TR:Result table>"
    # Original latex unchanged
    assert elems0[4]["latex"] == "Result table"

    # cells[0] gets translated_text
    assert elems0[4]["cells"][0]["translated_text"] == "<TR:Accuracy>"
    # cells[0] original text unchanged
    assert elems0[4]["cells"][0]["text"] == "Accuracy"
    # "42.5" cell gets no translated_text
    assert "translated_text" not in elems0[4]["cells"][1]

    # BYPASS element unchanged
    bypass = elems0[2]
    assert bypass["category"] == "BYPASS"
    assert bypass["translated_text"] == ""

    # Equation-only element: translated_text not overwritten from ""
    eq = elems0[3]
    assert eq["translated_text"] == ""

    # Structural fields preserved verbatim
    assert out["pdf_path"] == "test.pdf"
    assert elems0[0]["bbox_pdf"] == [72, 720, 540, 740]
    assert elems0[0]["label"] == "SectionHeader"


# ── 6. length violation retry ────────────────────────────────────────────────────

def test_length_violation_retry():
    cfg = _mock_cfg()
    cfg.chunk_bytes = 10000
    cfg.length_tolerance = 0.15

    source = "This is a long enough source string for length checking"
    too_long = source + " extra extra extra extra extra extra extra extra extra extra extra"
    correct = "Đây là một chuỗi nguồn đủ dài để kiểm tra độ dài"

    call_count = 0

    async def _side_effect(system: str, user: str, *, force_json: bool = False) -> str:
        nonlocal call_count
        call_count += 1
        match = re.search(r"```json\n(.*?)\n```", user, re.DOTALL)
        if not match:
            return "[]"
        chunk = json.loads(match.group(1))
        ids = list(chunk.keys())
        # First call returns over-long; subsequent calls return correct
        t = too_long if call_count == 1 else correct
        return json.dumps([{"id": k, "t": t} for k in ids])

    doc = {
        "source_language": "English",
        "target_language": "Vietnamese",
        "pdf_path": "t.pdf",
        "pages": [{
            "page_index": 0,
            "page_width": 612,
            "page_height": 792,
            "elements": [{
                "label": "Text",
                "category": "TEXT",
                "bbox_pdf": [0, 0, 100, 10],
                "source_text": source,
                "translated_text": "",
                "latex": "",
                "cells": [],
            }],
        }],
    }

    mock_call = AsyncMock(side_effect=_side_effect)
    with patch.object(Gateway, "call", mock_call):
        out = translate_document(doc, cfg)

    assert out["pages"][0]["elements"][0]["translated_text"] == correct
    # 1 initial call + 1 length-violation retry
    assert call_count == 2


# ── 7. glossary injection ────────────────────────────────────────────────────────

def test_glossary_injection():
    glossary = {"ceadar": "CEADAR"}

    chunk_with = {"5": "CeADAR is a research center located in Dublin Ireland."}
    chunk_without = {"0": "Introduction to Machine Learning"}

    block_with = glossary_block_for_chunk(chunk_with, glossary)
    block_without = glossary_block_for_chunk(chunk_without, glossary)

    assert "ceadar" in block_with.lower()
    assert "CEADAR" in block_with
    assert block_without == ""
