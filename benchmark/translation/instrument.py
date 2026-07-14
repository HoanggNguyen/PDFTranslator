"""Read-only latency/token instrumentation for the translation core.

Monkeypatches ``httpx.AsyncClient.send`` to time every request and read the API
``usage`` block. It only *observes* — control flow, headers, and bodies are untouched,
so the measured system behaves exactly as in production. Thread-safe (the harness may
run several documents concurrently in quality-gen mode).

Usage:
    with Instrument() as inst:
        start = inst.mark()
        translate_document(doc, cfg)
        stats = inst.since(start)   # RequestStats for just this document
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock

import httpx


@dataclass
class ReqRecord:
    url: str
    status: int
    elapsed_s: float
    tok_in: int | None
    tok_out: int | None


@dataclass
class RequestStats:
    n_req: int
    n_retry: int          # 429 or 5xx responses (each triggers a retry in the gateway)
    tok_in: int
    tok_out: int
    req_latencies: list[float]

    @property
    def p50(self) -> float:
        return _pct(self.req_latencies, 50)

    @property
    def p95(self) -> float:
        return _pct(self.req_latencies, 95)


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class Instrument:
    def __init__(self) -> None:
        self._records: list[ReqRecord] = []
        self._lock = Lock()
        self._orig = None

    def __enter__(self) -> "Instrument":
        self._orig = httpx.AsyncClient.send
        orig = self._orig
        records, lock = self._records, self._lock

        async def patched(client_self, request, **kwargs):
            t0 = time.perf_counter()
            resp = await orig(client_self, request, **kwargs)
            elapsed = time.perf_counter() - t0
            tok_in = tok_out = None
            try:
                # For non-streaming requests (the gateway never streams) the body is
                # already read+cached by the time send() returns, so .json() is safe
                # and does not consume anything the gateway needs later.
                if not kwargs.get("stream", False):
                    usage = resp.json().get("usage") or {}
                    tok_in = usage.get("prompt_tokens")
                    tok_out = usage.get("completion_tokens")
            except Exception:  # noqa: BLE001 — instrumentation must never break a run
                pass
            with lock:
                records.append(ReqRecord(
                    url=str(request.url), status=resp.status_code,
                    elapsed_s=elapsed, tok_in=tok_in, tok_out=tok_out,
                ))
            return resp

        httpx.AsyncClient.send = patched
        return self

    def __exit__(self, *_) -> None:
        if self._orig is not None:
            httpx.AsyncClient.send = self._orig

    def mark(self) -> int:
        """Index snapshot; pass to since() to get stats for work done after it."""
        with self._lock:
            return len(self._records)

    def since(self, mark: int) -> RequestStats:
        with self._lock:
            recs = self._records[mark:]
        return RequestStats(
            n_req=len(recs),
            n_retry=sum(1 for r in recs if r.status == 429 or r.status >= 500),
            tok_in=sum(r.tok_in or 0 for r in recs),
            tok_out=sum(r.tok_out or 0 for r in recs),
            req_latencies=[r.elapsed_s for r in recs],
        )
