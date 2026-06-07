from __future__ import annotations

import json

from .models import Task
from .predicates import has_prose_for_equation, is_equation_only, is_plain_text


def collect_translatables(doc: dict) -> list[Task]:
    tasks: list[Task] = []
    idx = 0
    for page in doc.get("pages", []):
        for elem in page.get("elements", []):
            category = elem.get("category", "")
            cells = elem.get("cells", [])
            text_lines = elem.get("equation_words") or []
            # EQUATION with per-line text fragments: translate each line into
            # its own translated_text. Skip elem.source_text + elem.latex —
            # rendering only consumes per-line translations (the rest of the
            # equation image stays untouched).
            is_equation_with_lines = category == "EQUATION" and text_lines
            # TABLE with cells: translate each cell individually; skip elem.source_text
            # (which is just " | ".join(cells) — translating both wastes API calls).
            is_table_with_cells = category == "TABLE" and cells
            if (
                not is_table_with_cells
                and not is_equation_with_lines
                and category != "EQUATION"
            ):
                src = elem.get("source_text", "")
                if src and category != "BYPASS" and not is_equation_only(src):
                    tasks.append(Task(elem, "translated_text", src, str(idx)))
                    idx += 1
            if not is_equation_with_lines:
                latex = elem.get("latex", "")
                if latex and is_plain_text(latex):
                    tasks.append(Task(elem, "translated_latex", latex, str(idx)))
                    idx += 1
            for cell in cells:
                text = cell.get("source_text", "")
                if text and is_plain_text(text):
                    tasks.append(Task(cell, "translated_text", text, str(idx)))
                    idx += 1
            for line in text_lines:
                text = line.get("text", "")
                if text and has_prose_for_equation(text):
                    tasks.append(Task(line, "translated_text", text, str(idx)))
                    idx += 1
    return tasks


def segments_to_chunks(tasks: list[Task], max_bytes: int) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    chunk: dict[str, str] = {}
    for task in tasks:
        candidate = {**chunk, task.id: task.text}
        size = len(json.dumps(candidate, ensure_ascii=False).encode())
        if size > max_bytes and chunk:
            chunks.append(chunk)
            chunk = {task.id: task.text}
        else:
            chunk = candidate
    if chunk:
        chunks.append(chunk)
    return chunks
