from __future__ import annotations

import asyncio
import json
import re
from collections import deque
from threading import Lock

import httpx
import json_repair

from .config import TranslatorConfig

_MAX_CONTINUE = 2
_THINK_RE = re.compile(r"^\s*<think>.*?</think>", re.DOTALL)


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
                    system,
                    user,
                    force_json=force_json,
                    retry=retry,
                    accumulated=content,
                    cont=cont + 1,
                )
            return content
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                await asyncio.sleep(5)
            if retry < self._cfg.retry:
                await asyncio.sleep(0.5 * (2**retry))
                return await self._request(
                    system, user, force_json=force_json, retry=retry + 1
                )
            raise
        except Exception:
            if retry < self._cfg.retry:
                await asyncio.sleep(0.5 * (2**retry))
                return await self._request(
                    system, user, force_json=force_json, retry=retry + 1
                )
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
