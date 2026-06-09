"""Framework-agnostic translation runner.

Runs the OCR -> translate -> render pipeline in a worker thread (so its internal
``asyncio.run`` works) and streams per-phase progress through a queue. Knows
nothing about Gradio, so it can be unit-tested in isolation.
"""

from __future__ import annotations

import logging
import queue
import tempfile
import threading
import traceback
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TranslationRequest:
    """One translation job, with the provider already resolved to its key."""

    pdf_path: str | None
    provider: str  # Phase-2 provider key (e.g. "openrouter")
    api_key: str
    model: str | None
    src_lang: str
    tgt_lang: str
    font: str
    pages: list[int] | None


@dataclass(frozen=True)
class Progress:
    """A per-phase progress update streamed while the pipeline runs."""

    frac: float
    msg: str


@dataclass(frozen=True)
class Result:
    """The terminal outcome of a run."""

    status: str  # "ok" | "invalid" | "error"
    out_path: str | None = None
    detail: str = ""


def validate(req: TranslationRequest) -> str | None:
    """Return a user-facing error message if the request can't run, else None."""
    if not req.pdf_path:
        return "Vui lòng tải lên một file PDF."
    if not req.api_key or not req.api_key.strip():
        return "Thiếu API key — nhập API key của provider ở thanh bên."
    if not req.src_lang or not req.tgt_lang:
        return "Chọn ngôn ngữ nguồn và ngôn ngữ đích."
    return None


def stream_translation(req: TranslationRequest) -> Iterator[Progress | Result]:
    """Yield ``Progress`` updates while translating, then one terminal ``Result``."""
    # Imported lazily so the lightweight bits above (dataclasses, validate) stay
    # importable without the heavy ML stack (torch, surya, ...) that e2e pulls in.
    from pdf2zh.e2e import run_pipeline

    q: queue.Queue = queue.Queue()
    work_dir = Path(tempfile.gettempdir()) / f"pdf2zh_{uuid.uuid4().hex}"

    def on_progress(frac: float, msg: str) -> None:
        q.put(Progress(frac, msg))

    def worker() -> None:
        try:
            out = run_pipeline(
                pdf_path=req.pdf_path,
                src_lang=req.src_lang,
                tgt_lang=req.tgt_lang,
                provider=req.provider,
                api_key=req.api_key,
                model=req.model,
                pages=req.pages,
                font=req.font,
                work_dir=work_dir,
                progress=on_progress,
            )
            q.put(Result("ok", out_path=out))
        except ValueError as exc:  # user-facing input error
            q.put(Result("invalid", detail=str(exc)))
        except Exception as exc:  # noqa: BLE001 — surface anything else to the UI
            logger.exception("pipeline failed")
            tail = "".join(traceback.format_exc().splitlines(keepends=True)[-6:])
            q.put(
                Result("error", detail=f"{type(exc).__name__}: {exc}\n```\n{tail}\n```")
            )

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        yield item
        if isinstance(item, Result):
            return
