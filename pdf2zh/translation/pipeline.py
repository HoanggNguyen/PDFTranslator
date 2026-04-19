from __future__ import annotations

import asyncio
import json
import logging

import json_repair

from .chunker import collect_translatables, segments_to_chunks
from .config import TranslatorConfig, resolve_provider
from .gateway import Gateway
from .models import Task
from .prompts import (
    build_glossary_prompt,
    build_translation_prompt,
    glossary_block_for_chunk,
)

logger = logging.getLogger("json_translator")


async def extract_glossary(
    chunks: list[dict[str, str]],
    cfg: TranslatorConfig,
) -> dict[str, str]:
    glossary: dict[str, str] = {}
    lock = asyncio.Lock()

    async def _process(gw: Gateway, chunk: dict[str, str]) -> None:
        system, user = build_glossary_prompt(
            chunk, cfg.source_language, cfg.target_language
        )
        try:
            raw = await gw.call(system, user)
            parsed = json_repair.loads(raw)
            if isinstance(parsed, list):
                async with lock:
                    for item in parsed:
                        if isinstance(item, dict) and "src" in item and "dst" in item:
                            key = item["src"].strip().lower()
                            if key not in glossary:
                                glossary[key] = item["dst"]
        except Exception as exc:
            logger.warning(f"Glossary chunk failed: {exc}")

    async with Gateway(cfg) as gw:
        await asyncio.gather(*[_process(gw, c) for c in chunks])
    return glossary


async def translate_chunks(
    chunks: list[dict[str, str]],
    glossary: dict[str, str],
    cfg: TranslatorConfig,
) -> dict[str, str]:
    results: dict[str, str] = {}
    lock = asyncio.Lock()

    async def _process(gw: Gateway, chunk: dict[str, str]) -> None:
        block = glossary_block_for_chunk(chunk, glossary)
        system, user = build_translation_prompt(
            chunk, cfg.source_language, cfg.target_language, block
        )
        translated = await _translate_one_chunk(gw, system, user, chunk, cfg)
        async with lock:
            results.update(translated)
        logger.info(f"Chunk done: {len(chunk)} segments.")

    async with Gateway(cfg) as gw:
        await asyncio.gather(*[_process(gw, c) for c in chunks])
    return results


async def _translate_one_chunk(
    gw: Gateway,
    system: str,
    user: str,
    chunk: dict[str, str],
    cfg: TranslatorConfig,
) -> dict[str, str]:
    original_ids = set(chunk.keys())

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

    # Drop extra ids (hallucinations)
    for k in set(result.keys()) - original_ids:
        del result[k]

    # Retry for missing ids
    missing = original_ids - set(result.keys())
    if missing:
        retry_user = (
            user + "\nDo not omit any IDs; every input ID must appear exactly once."
        )
        retry_result = _parse(await gw.call(system, retry_user))
        for k in missing:
            if k in retry_result:
                result[k] = retry_result[k]
            else:
                logger.warning(f"ID {k} missing after retry; falling back to source.")
                result[k] = chunk[k]

    # Retry if model returned source unchanged (no translation)
    if result and all(result.get(k) == chunk.get(k) for k in original_ids):
        retry_result = _parse(await gw.call(system, user))
        if not all(retry_result.get(k) == chunk.get(k) for k in original_ids):
            result.update({k: v for k, v in retry_result.items() if k in original_ids})

    # Length check — retry violators once, then warn-and-keep
    violators = {
        k
        for k in original_ids
        if _length_violation(result.get(k, ""), chunk[k], cfg.length_tolerance)
    }
    if violators:
        vchunk = {k: chunk[k] for k in violators}
        prev_json = json.dumps({k: result[k] for k in violators}, ensure_ascii=False)
        v_block = glossary_block_for_chunk(vchunk, {})
        v_sys, v_usr = build_translation_prompt(
            vchunk, cfg.source_language, cfg.target_language, v_block
        )
        v_usr += (
            f"\nPrevious attempt violated the length constraint. "
            f"Keep each translation within ±15% of source length. "
            f"Previous (rejected):\n{prev_json}"
        )
        retry2 = _parse(await gw.call(v_sys, v_usr))
        for k in violators:
            t = retry2.get(k, result.get(k, ""))
            if _length_violation(t, chunk[k], cfg.length_tolerance):
                logger.warning(
                    f"Persistent length violation id={k} "
                    f"(src={len(chunk[k])}, out={len(t)})"
                )
            result[k] = t

    return result


def _length_violation(translation: str, source: str, tol: float) -> bool:
    if len(source) < 20:
        return False
    return abs(len(translation) - len(source)) / max(len(source), 1) > tol


async def _pipeline(
    tasks: list[Task],
    chunks: list[dict[str, str]],
    cfg: TranslatorConfig,
) -> dict[str, str]:
    glossary: dict[str, str] = {}
    if cfg.glossary_enabled:
        glossary = await extract_glossary(chunks, cfg)
        logger.info(f"Glossary: {len(glossary)} terms")
    return await translate_chunks(chunks, glossary, cfg)


def translate_document(doc: dict, cfg: TranslatorConfig) -> dict:
    if not cfg.source_language:
        cfg.source_language = doc.get("source_language", "")
    if not cfg.target_language:
        cfg.target_language = doc.get("target_language", "")
    if not cfg.source_language or not cfg.target_language:
        raise ValueError(
            "source_language and target_language must be set via JSON metadata or --src/--tgt flags."
        )

    resolve_provider(cfg)

    tasks = collect_translatables(doc)
    if not tasks:
        logger.info("No translatable segments found.")
        return doc

    chunks = segments_to_chunks(tasks, cfg.chunk_bytes)
    logger.info(f"Segments: {len(tasks)}, chunks: {len(chunks)}")

    translations = asyncio.run(_pipeline(tasks, chunks, cfg))

    for task in tasks:
        t = translations.get(task.id)
        if t is not None:
            task.target[task.write_key] = t

    token_stats = {
        "total_segments": len(tasks),
        "total_chunks": len(chunks),
        "translated": sum(1 for t in tasks if t.id in translations),
    }
    logger.info(f"Token usage stats: {token_stats}")
    return doc
