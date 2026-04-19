from __future__ import annotations

import json

from .models import Task
from .predicates import is_equation_only, is_plain_text


def collect_translatables(doc: dict) -> list[Task]:
    tasks: list[Task] = []
    idx = 0
    for page in doc.get("pages", []):
        for elem in page.get("elements", []):
            src = elem.get("source_text", "")
            if src and elem.get("category") != "BYPASS" and not is_equation_only(src):
                tasks.append(Task(elem, "translated_text", src, str(idx)))
                idx += 1
            latex = elem.get("latex", "")
            if latex and is_plain_text(latex):
                tasks.append(Task(elem, "translated_latex", latex, str(idx)))
                idx += 1
            for cell in elem.get("cells", []):
                text = cell.get("text", "")
                if text and is_plain_text(text):
                    tasks.append(Task(cell, "translated_text", text, str(idx)))
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
