"""Post-translation pass: fix bare math wrapping + detect multi-column layouts.

After Stage B translation, some elements have:
  - Math expressions outside <math> tags (LLM forgot to wrap)
  - Multi-column reference layouts that got flattened into one line

This module:
  1. Regex-detects elements that look math-y
  2. Sends them in batches to LLM with bbox info
  3. LLM returns fixed text with <math> wrapping and (optionally) <typst>...</typst>
     blocks for grid/column layouts
  4. Writes corrections back into the doc
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import NamedTuple

import json_repair

from .config import TranslatorConfig
from .gateway import Gateway
from .predicates import is_equation_only
from .prompts import build_math_fix_prompt

logger = logging.getLogger("json_translator")


# ---------------------------------------------------------------------------
# Math detection
# ---------------------------------------------------------------------------

# Indicators that an element contains math expressions worth re-checking:
# - Typst math function calls: frac(, sqrt(, sum_, int_
# - LaTeX backslash commands: \frac, \sqrt, \int
# - Math keywords: pi, theta, alpha, sin, cos, etc.
# - Exponent/subscript notation: x^2, a_n, x^{n+1}
# - Existing <math> tags (we still want layout reasoning)
_MATH_INDICATORS = re.compile(
    r"<math\b"  # already-tagged math (re-check layout)
    r"|\\[a-zA-Z]{2,}"  # any LaTeX backslash command
    r"|\b(?:frac|sqrt|sum|int|prod|lim)\s*\("  # Typst math calls
    r"|\b(?:pi|theta|alpha|beta|gamma|delta|sigma|mu|lambda|omega|infty|sin|cos|tan|log|ln)\b"
    r"|[a-zA-Z][\^_]\{?[0-9a-zA-Z+\-]"  # x^2, a_n, x^{n+1}
)


class MathTask(NamedTuple):
    elem: dict  # element dict to write back into
    write_key: str  # "translated_text"
    text: str  # current translated_text
    bbox: list  # [x0, y0, x1, y1] for layout reasoning
    id: str  # numeric id for chunk addressing


def collect_math_candidates(doc: dict) -> list[MathTask]:
    """Return tasks for elements whose translated_text looks math-y."""
    tasks: list[MathTask] = []
    idx = 0
    for page in doc.get("pages", []):
        for elem in page.get("elements", []):
            cat = elem.get("category", "")
            if cat in ("BYPASS", "TABLE"):
                continue
            text = elem.get("translated_text", "") or ""
            if not text:
                continue
            if not _MATH_INDICATORS.search(text):
                continue
            # Pure-math elements: leave them alone — we'll skip rendering and let
            # the original PDF's text layer show through.
            if is_equation_only(text):
                continue
            bbox = elem.get("bbox_pdf") or [0, 0, 0, 0]
            tasks.append(MathTask(elem, "translated_text", text, bbox, str(idx)))
            idx += 1
    return tasks


def tasks_to_chunks(tasks: list[MathTask], max_bytes: int) -> list[list[MathTask]]:
    """Pack math tasks into chunks bounded by serialized JSON size."""
    chunks: list[list[MathTask]] = []
    current: list[MathTask] = []
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


# ---------------------------------------------------------------------------
# Async LLM pipeline
# ---------------------------------------------------------------------------


def _tags_balanced(text: str) -> bool:
    """Reject LLM outputs with unclosed <math>/<typst> blocks."""
    for tag in ("math", "typst"):
        opens = len(re.findall(rf"<{tag}\b", text, re.IGNORECASE))
        closes = len(re.findall(rf"</{tag}\b", text, re.IGNORECASE))
        if opens != closes:
            return False
    return True


def _has_unsafe_latex(text: str) -> bool:
    """Reject outputs where LaTeX leaked outside <math> tags (we can't render it)."""
    s = re.sub(r"<math\b[^>]*>.*?</math>", "", text, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<typst\b[^>]*>.*?</typst>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"\$\$.*?\$\$", "", s, flags=re.DOTALL)
    s = re.sub(r"\$[^$\n]*\$", "", s)
    return bool(re.search(r"\\[a-zA-Z]+", s))


async def _fix_one_chunk(gw: Gateway, chunk: list[MathTask]) -> dict[str, str]:
    payload = [
        {
            "id": t.id,
            "text": t.text,
            "bbox": [round(c, 1) for c in t.bbox],
        }
        for t in chunk
    ]
    system, user = build_math_fix_prompt(payload)

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
    out: dict[str, str] = {}
    for k, v in result.items():
        if k not in valid_ids:
            continue
        if not _tags_balanced(v):
            logger.warning(f"Math-fix dropped id={k}: unbalanced <math>/<typst> tags")
            continue
        if _has_unsafe_latex(v):
            logger.warning(f"Math-fix dropped id={k}: bare LaTeX leaked outside tags")
            continue
        out[k] = v
    return out


async def _fix_chunks(
    chunks: list[list[MathTask]], cfg: TranslatorConfig
) -> dict[str, str]:
    results: dict[str, str] = {}
    lock = asyncio.Lock()

    async def _process(gw: Gateway, chunk: list[MathTask]) -> None:
        try:
            fixed = await _fix_one_chunk(gw, chunk)
            async with lock:
                results.update(fixed)
            logger.info(f"Math-fix chunk done: {len(chunk)} elements.")
        except Exception as exc:
            logger.warning(f"Math-fix chunk failed: {exc}")

    async with Gateway(cfg) as gw:
        await asyncio.gather(*[_process(gw, c) for c in chunks])
    return results


def fix_math_document(doc: dict, cfg: TranslatorConfig) -> dict:
    """Run the math-fix pass over `doc` (mutates in place, returns same doc)."""
    tasks = collect_math_candidates(doc)
    if not tasks:
        logger.info("Math-fix: no math candidates found.")
        return doc

    chunks = tasks_to_chunks(tasks, cfg.chunk_bytes)
    logger.info(f"Math-fix: {len(tasks)} elements, {len(chunks)} chunks")

    fixes = asyncio.run(_fix_chunks(chunks, cfg))

    applied = 0
    for task in tasks:
        new_text = fixes.get(task.id)
        if new_text is not None and new_text != task.text:
            task.elem[task.write_key] = new_text
            applied += 1
    logger.info(f"Math-fix: applied {applied}/{len(tasks)} fixes")
    return doc
