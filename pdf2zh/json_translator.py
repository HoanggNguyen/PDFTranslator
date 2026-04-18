from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import NamedTuple

import httpx
import json_repair
from dotenv import load_dotenv

logger = logging.getLogger("json_translator")

# ── Provider table ─────────────────────────────────────────────────────────────

PROVIDERS: dict[str, dict[str, str]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "google/gemini-2.5-flash-lite",
        "env_var": "OPENROUTER_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.5-flash-lite",
        "env_var": "GEMINI_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "env_var": "OPENAI_API_KEY",
    },
}

# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class TranslatorConfig:
    source_language: str = ""
    target_language: str = ""
    provider: str = "openrouter"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    concurrent: int = 30
    rpm: int | None = None
    tpm: int | None = None
    chunk_bytes: int = 3000
    glossary_enabled: bool = True
    length_tolerance: float = 0.15
    timeout: int = 300
    retry: int = 2


def resolve_provider(cfg: TranslatorConfig) -> None:
    load_dotenv()
    p = PROVIDERS.get(cfg.provider)
    if p is None:
        raise ValueError(f"Unknown provider '{cfg.provider}'. Choose from {list(PROVIDERS)}.")
    if cfg.base_url is None:
        cfg.base_url = p["base_url"]
    if cfg.model is None:
        cfg.model = p["model"]
    if cfg.api_key is None:
        cfg.api_key = os.environ.get(p["env_var"])
    if not cfg.api_key:
        raise ValueError(f"No API key found. Set {p['env_var']} or pass --api-key.")


# ── Predicates ─────────────────────────────────────────────────────────────────

_MATH_TAG = re.compile(r"<math>.*?</math>", re.DOTALL)
_EQ_LABEL = re.compile(r"\(\d+(\.\d+)*[a-z]?\)")
# Two or more consecutive letters across major Unicode scripts
_LETTER_RUN = re.compile(
    r"[A-Za-z\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF"
    r"\u0600-\u06FF\u0900-\u097F\u0E00-\u0E7F\u2E80-\u9FFF]{2,}"
)


def is_plain_text(s: str) -> bool:
    without_math = _MATH_TAG.sub("", s)
    return bool(_LETTER_RUN.search(without_math))


def is_equation_only(s: str) -> bool:
    without_math = _MATH_TAG.sub("", s)
    without_labels = _EQ_LABEL.sub("", without_math)
    return not without_labels.strip()


# ── Task ───────────────────────────────────────────────────────────────────────

class Task(NamedTuple):
    target: dict    # dict to write into (element or cell dict)
    write_key: str  # key to set on target
    text: str
    id: str


# ── Collect translatables ──────────────────────────────────────────────────────

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


# ── Chunking ───────────────────────────────────────────────────────────────────

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


# ── Prompts ────────────────────────────────────────────────────────────────────

def build_translation_prompt(
    chunk: dict[str, str],
    src_lang: str,
    tgt_lang: str,
    glossary_block: str,
) -> tuple[str, str]:
    system = (
        f"You are a professional, authentic machine translation engine.\n\n"
        f"# Task\nTranslate text from {src_lang} into {tgt_lang}.\n\n"
        f"# Rules\n"
        f"1. Preserve verbatim: content inside <math>...</math> or $...$, URLs, code blocks, "
        f"brand names, and any <ph-xxx> placeholder tags.\n"
        f"2. LENGTH CONSTRAINT (hard): each translation's character length MUST be within "
        f"±15% of its source string. Rephrase compactly if needed.\n"
        f"   Priority order when in tension:\n"
        f"     (1) semantic accuracy\n"
        f"     (2) length preservation\n"
        f"3. Do NOT merge or split entries. Every input id must appear exactly once in the "
        f"output, with the same id string.\n"
        f"4. Return ONLY the JSON array specified below. No prose, no code fences.\n\n"
        f"{glossary_block}"
    )
    input_json = json.dumps(chunk, ensure_ascii=False)
    example_ids = list(chunk.keys())[:2]
    example = ", ".join(f'{{"id":"{k}","t":"<translation {k}>"}}' for k in example_ids)
    user = (
        f"<input>\n```json\n{input_json}\n```\n</input>\n\n"
        f"Return a JSON array in exactly this shape:\n[{example}]"
    )
    return system, user


def build_glossary_prompt(
    chunk: dict[str, str],
    src_lang: str,
    tgt_lang: str,
) -> tuple[str, str]:
    system = "You are a professional glossary extractor."
    input_json = json.dumps(chunk, ensure_ascii=False)
    user = (
        f"Extract proper nouns — people, places, organizations, product names, technical terms "
        f"— from the {src_lang} text below. Provide their {tgt_lang} translations.\n\n"
        f"Rules:\n"
        f"- Do NOT include common nouns.\n"
        f"- Do NOT include content inside <math>...</math> or <ph-xxx> tags.\n"
        f"- Each src appears at most once. No explanations.\n\n"
        f"<input>\n```json\n{input_json}\n```\n</input>\n\n"
        f'Output format — JSON array only:\n[{{"src":"<term>","dst":"<translation>"}}]'
    )
    return system, user


def glossary_block_for_chunk(chunk: dict[str, str], glossary: dict[str, str]) -> str:
    combined = " ".join(chunk.values()).lower()
    matches = [(src, dst) for src, dst in glossary.items() if src.lower() in combined]
    if not matches:
        return ""
    lines = "\n".join(f"{src} => {dst}" for src, dst in matches)
    return f"# Glossary (use these exact translations when the term appears)\n{lines}\n"


# ── Rate limiter ───────────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, rpm: int | None, tpm: int | None):
        self.rpm = rpm
        self.tpm = tpm
        self._req_ts: deque[float] = deque()
        self._tok_ts: deque[tuple[float, int]] = deque()
        self._lock = Lock()

    def _cleanup(self, now: float) -> None:
        cutoff = now - 60.0
        while self._req_ts and self._req_ts[0] <= cutoff:
            self._req_ts.popleft()
        while self._tok_ts and self._tok_ts[0][0] <= cutoff:
            self._tok_ts.popleft()

    def _wait_time(self, now: float, tokens: int) -> float:
        self._cleanup(now)
        wait = 0.0
        if self.rpm and len(self._req_ts) >= self.rpm:
            wait = max(wait, 60.0 - (now - self._req_ts[0]))
        if self.tpm:
            cur = sum(t[1] for t in self._tok_ts)
            if cur + tokens > self.tpm and self._tok_ts:
                wait = max(wait, 60.0 - (now - self._tok_ts[0][0]))
        return wait

    def _record(self, now: float, tokens: int) -> None:
        if self.rpm is not None:
            self._req_ts.append(now)
        if self.tpm is not None:
            self._tok_ts.append((now, tokens))

    async def acquire(self, tokens: int = 0) -> None:
        if self.rpm is None and self.tpm is None:
            return
        import time
        while True:
            with self._lock:
                now = time.time()
                wait = self._wait_time(now, tokens)
                if wait <= 0:
                    self._record(now, tokens)
                    return
            await asyncio.sleep(wait + 0.1)


# ── Gateway ────────────────────────────────────────────────────────────────────

_MAX_CONTINUE = 2
_THINK_RE = re.compile(r"^\s*<think>.*?</think>", re.DOTALL)


class Gateway:
    def __init__(self, cfg: TranslatorConfig):
        self._cfg = cfg
        self._sem = asyncio.Semaphore(cfg.concurrent)
        self._rate = RateLimiter(cfg.rpm, cfg.tpm)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "Gateway":
        limits = httpx.Limits(
            max_connections=self._cfg.concurrent * 2,
            max_keepalive_connections=self._cfg.concurrent,
        )
        timeout = httpx.Timeout(connect=5, read=self._cfg.timeout, write=300, pool=10)
        self._client = httpx.AsyncClient(limits=limits, timeout=timeout, verify=False)
        return self

    async def __aexit__(self, *_) -> None:
        await self._client.aclose()

    async def call(self, system: str, user: str, *, force_json: bool = False) -> str:
        async with self._sem:
            await self._rate.acquire()
            return await self._request(system, user, force_json=force_json)

    async def _request(
        self,
        system: str,
        user: str,
        *,
        force_json: bool,
        retry: int = 0,
        accumulated: str = "",
        cont: int = 0,
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._cfg.api_key}",
        }
        data: dict = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.7,
        }
        if force_json:
            data["response_format"] = {"type": "json_object"}
        try:
            resp = await self._client.post(
                f"{self._cfg.base_url}/chat/completions",
                json=data,
                headers=headers,
            )
            resp.raise_for_status()
            rdata = json.loads(resp.text.lstrip())
            choices = rdata.get("choices", [])
            if not choices:
                raise ValueError("empty choices in response")
            finish = choices[0].get("finish_reason")
            content = choices[0].get("message", {}).get("content", "")
            content = _THINK_RE.sub("", content)
            content = self._merge(accumulated, content) if accumulated else content
            if finish == "length" and cont < _MAX_CONTINUE:
                return await self._request(
                    system, user, force_json=force_json,
                    retry=retry, accumulated=content, cont=cont + 1,
                )
            return content
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                await asyncio.sleep(5)
            if retry < self._cfg.retry:
                await asyncio.sleep(0.5 * (2 ** retry))
                return await self._request(system, user, force_json=force_json, retry=retry + 1)
            raise
        except Exception:
            if retry < self._cfg.retry:
                await asyncio.sleep(0.5 * (2 ** retry))
                return await self._request(system, user, force_json=force_json, retry=retry + 1)
            raise

    @staticmethod
    def _merge(acc: str, add: str) -> str:
        try:
            a = json_repair.loads(acc)
            b = json_repair.loads(add)
            if isinstance(a, list) and isinstance(b, list):
                seen = {x.get("id") for x in a if isinstance(x, dict)}
                for item in b:
                    if isinstance(item, dict) and item.get("id") not in seen:
                        a.append(item)
                        seen.add(item.get("id"))
                return json.dumps(a, ensure_ascii=False)
        except Exception:
            pass
        return acc + add


# ── Glossary pass ──────────────────────────────────────────────────────────────

async def extract_glossary(
    chunks: list[dict[str, str]],
    cfg: TranslatorConfig,
) -> dict[str, str]:
    glossary: dict[str, str] = {}
    lock = asyncio.Lock()

    async def _process(gw: Gateway, chunk: dict[str, str]) -> None:
        system, user = build_glossary_prompt(chunk, cfg.source_language, cfg.target_language)
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


# ── Translation pass ───────────────────────────────────────────────────────────

async def translate_chunks(
    chunks: list[dict[str, str]],
    glossary: dict[str, str],
    cfg: TranslatorConfig,
) -> dict[str, str]:
    results: dict[str, str] = {}
    lock = asyncio.Lock()

    async def _process(gw: Gateway, chunk: dict[str, str]) -> None:
        block = glossary_block_for_chunk(chunk, glossary)
        system, user = build_translation_prompt(chunk, cfg.source_language, cfg.target_language, block)
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
        retry_user = user + "\nDo not omit any IDs; every input ID must appear exactly once."
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
        k for k in original_ids
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


# ── Orchestrator ───────────────────────────────────────────────────────────────

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
        "translated": sum(1 for t in tasks if task.id in translations),
    }
    logger.info(f"Token usage stats: {token_stats}")
    return doc


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Translate PDFTranslator JSON output")
    parser.add_argument("input", help="Input JSON file")
    parser.add_argument("-o", "--output", help="Output JSON file (default: INPUT.translated.json)")
    parser.add_argument("--src", dest="source_language", default="")
    parser.add_argument("--tgt", dest="target_language", default="")
    parser.add_argument("--provider", default="openrouter", choices=list(PROVIDERS))
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", dest="api_key", default=None)
    parser.add_argument("--concurrent", type=int, default=30)
    parser.add_argument("--chunk-bytes", type=int, default=3000)
    parser.add_argument("--no-glossary", action="store_true")
    parser.add_argument("--length-tolerance", type=float, default=0.15)
    parser.add_argument("--rpm", type=int, default=None)
    parser.add_argument("--tpm", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    with open(args.input, encoding="utf-8") as f:
        doc = json.load(f)

    cfg = TranslatorConfig(
        source_language=args.source_language or doc.get("source_language", ""),
        target_language=args.target_language or doc.get("target_language", ""),
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        concurrent=args.concurrent,
        chunk_bytes=args.chunk_bytes,
        glossary_enabled=not args.no_glossary,
        length_tolerance=args.length_tolerance,
        rpm=args.rpm,
        tpm=args.tpm,
    )

    doc = translate_document(doc, cfg)

    from pathlib import Path
    output = args.output or (str(Path(args.input).with_suffix("")) + ".translated.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved to {output}")


if __name__ == "__main__":
    main()
