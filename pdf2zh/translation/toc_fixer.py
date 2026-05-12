"""Post-translation pass: restructure flattened Table-of-Contents entries.

Stage A flattens TOC pages into a single concatenated string — section
numbers, titles, dot leaders, and page numbers all run together. The
regex-based parser in render handles common cases but misses entries
without dot leaders or with unusual spacing.

This pass asks the LLM to:
  1. Identify each TOC entry boundary
  2. Strip dot leaders
  3. Emit '<title>\\t<page_number>' lines (tab-separated)

The render-time `parse_toc_entries` then has clean per-line input to work
with, producing properly aligned TOC layouts.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import NamedTuple

import json_repair

from .config import TranslatorConfig
from .gateway import Gateway
from .prompts import build_toc_fix_prompt

logger = logging.getLogger("json_translator")


class TocTask(NamedTuple):
    elem: dict
    write_key: str
    text: str
    bbox: list
    id: str


def collect_toc_candidates(doc: dict) -> list[TocTask]:
    """Return tasks for TOC elements that need restructuring."""
    tasks: list[TocTask] = []
    idx = 0
    for page in doc.get("pages", []):
        for elem in page.get("elements", []):
            if elem.get("label") != "TableOfContents":
                continue
            text = elem.get("translated_text", "") or ""
            if not text:
                continue
            # Already restructured (has newlines / tabs)
            if "\t" in text or text.count("\n") >= 3:
                continue
            bbox = elem.get("bbox_pdf") or [0, 0, 0, 0]
            tasks.append(TocTask(elem, "translated_text", text, bbox, str(idx)))
            idx += 1
    return tasks


def tasks_to_chunks(tasks: list[TocTask], max_bytes: int) -> list[list[TocTask]]:
    chunks: list[list[TocTask]] = []
    current: list[TocTask] = []
    cur_bytes = 0
    for task in tasks:
        entry = {
            "id": task.id,
            "text": task.text,
            "bbox": [round(c, 1) for c in task.bbox],
        }
        size = len(json.dumps(entry, ensure_ascii=False).encode())
        if cur_bytes + size > max_bytes and current:
            chunks.append(current)
            current = [task]
            cur_bytes = size
        else:
            current.append(task)
            cur_bytes += size
    if current:
        chunks.append(current)
    return chunks


async def _fix_one_chunk(gw: Gateway, chunk: list[TocTask]) -> dict[str, str]:
    payload = [
        {
            "id": t.id,
            "text": t.text,
            "bbox": [round(c, 1) for c in t.bbox],
        }
        for t in chunk
    ]
    system, user = build_toc_fix_prompt(payload)

    def _parse(raw: str) -> dict[str, str]:
        parsed = json_repair.loads(raw)
        if isinstance(parsed, dict):
            parsed = [{"id": k, "t": v} for k, v in parsed.items()]
        if not isinstance(parsed, list):
            return {}
        out: dict[str, str] = {}
        for item in parsed:
            if isinstance(item, dict) and "id" in item and "t" in item:
                out[str(item["id"])] = str(item["t"])
        return out

    result = _parse(await gw.call(system, user))
    valid_ids = {t.id for t in chunk}
    return {k: v for k, v in result.items() if k in valid_ids}


async def _fix_chunks(
    chunks: list[list[TocTask]], cfg: TranslatorConfig
) -> dict[str, str]:
    results: dict[str, str] = {}
    lock = asyncio.Lock()

    async def _process(gw: Gateway, chunk: list[TocTask]) -> None:
        try:
            fixed = await _fix_one_chunk(gw, chunk)
            async with lock:
                results.update(fixed)
            logger.info(f"TOC-fix chunk done: {len(chunk)} elements.")
        except Exception as exc:
            logger.warning(f"TOC-fix chunk failed: {exc}")

    async with Gateway(cfg) as gw:
        await asyncio.gather(*[_process(gw, c) for c in chunks])
    return results


def fix_toc_document(doc: dict, cfg: TranslatorConfig) -> dict:
    """Restructure flat TOC strings into tab-separated entry lines."""
    tasks = collect_toc_candidates(doc)
    if not tasks:
        logger.info("TOC-fix: no TOC candidates found.")
        return doc

    chunks = tasks_to_chunks(tasks, cfg.chunk_bytes)
    logger.info(f"TOC-fix: {len(tasks)} elements, {len(chunks)} chunks")

    fixes = asyncio.run(_fix_chunks(chunks, cfg))

    applied = 0
    for task in tasks:
        new_text = fixes.get(task.id)
        if new_text is not None and new_text != task.text:
            task.elem[task.write_key] = new_text
            applied += 1
    logger.info(f"TOC-fix: applied {applied}/{len(tasks)} fixes")
    return doc
