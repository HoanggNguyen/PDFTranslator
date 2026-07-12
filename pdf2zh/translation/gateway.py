from __future__ import annotations

import asyncio
import json
import random
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
        # Models that rejected a custom `temperature` (e.g. OpenAI reasoning models
        # like gpt-5-nano/o1/o3, which only accept the default value 1). Learned
        # lazily from a 400 response so we stop sending the param for this model
        # for the rest of the process, instead of retrying it on every call.
        self._no_temp_models: set[str] = set()

    @staticmethod
    def _is_unsupported_temperature(response: httpx.Response) -> bool:
        try:
            err = json.loads(response.text).get("error", {})
        except (ValueError, AttributeError):
            return False
        return err.get("param") == "temperature" and err.get("code") == "unsupported_value"

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

    async def call_vision(self, system: str, prompt: str, image_b64: str) -> str:
        """Send a vision request with a base64-encoded PNG image."""
        async with self._sem:
            await self._rate.acquire()
            return await self._request_vision(system, prompt, image_b64)

    def _retry_delay(self, retry: int, response: httpx.Response | None = None) -> float:
        """Seconds to wait before retrying: honor the server's Retry-After header,
        else capped exponential backoff with jitter (avoids synchronized retries
        all hammering the API at once and re-triggering 429)."""
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    return min(float(retry_after), 60.0)
                except ValueError:
                    pass
        return min(30.0, 2.0**retry) + random.uniform(0, 1)

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
        }
        if self._cfg.model not in self._no_temp_models:
            data["temperature"] = 0.7
        if force_json:
            data["response_format"] = {"type": "json_object"}
        if self._cfg.disable_reasoning and self._cfg.provider == "openrouter":
            data["reasoning"] = {"enabled": False}
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
            # `.get(key, "")` only falls back when the key is absent — some providers
            # send an explicit `"content": null` (e.g. on a filtered/empty completion),
            # which .get() passes through as None and crashes the regex sub below.
            content = choices[0].get("message", {}).get("content") or ""
            content = _THINK_RE.sub("", content)
            # Drop lone UTF-16 surrogates the model occasionally emits;
            # httpx fails to UTF-8 encode them on subsequent retry requests.
            content = content.encode("utf-8", errors="ignore").decode("utf-8")
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
            status = e.response.status_code
            # Some models (OpenAI reasoning models: gpt-5-nano, o1, o3, ...) reject
            # any non-default temperature outright. Learn this once per model and
            # replay the SAME attempt without the param — free (no tokens billed,
            # request was rejected before generation) and doesn't cost a retry slot.
            if (
                status == 400
                and "temperature" in data
                and self._is_unsupported_temperature(e.response)
            ):
                # NOTE: don't gate this on `model not in self._no_temp_models` —
                # concurrent calls for the same model can all be in flight before
                # the first one learns, so every one of them must be allowed to
                # self-heal independently. `"temperature" in data` alone already
                # prevents infinite recursion: the retried call rebuilds `data`
                # from `_no_temp_models`, which by then contains this model.
                self._no_temp_models.add(self._cfg.model)
                return await self._request(
                    system, user, force_json=force_json, retry=retry
                )
            # 429 (rate limit) and 5xx are transient — back off and retry. Other
            # 4xx (401 auth, 400 bad request) are permanent, so fail fast.
            if (status == 429 or status >= 500) and retry < self._cfg.retry:
                await asyncio.sleep(self._retry_delay(retry, e.response))
                return await self._request(
                    system, user, force_json=force_json, retry=retry + 1
                )
            raise
        except Exception:
            if retry < self._cfg.retry:
                await asyncio.sleep(self._retry_delay(retry))
                return await self._request(
                    system, user, force_json=force_json, retry=retry + 1
                )
            raise

    async def _request_vision(
        self,
        system: str,
        prompt: str,
        image_b64: str,
        retry: int = 0,
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._cfg.api_key}",
        }
        data: dict = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                },
            ],
        }
        if self._cfg.model not in self._no_temp_models:
            data["temperature"] = 0.2
        if self._cfg.disable_reasoning and self._cfg.provider == "openrouter":
            data["reasoning"] = {"enabled": False}
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
                raise ValueError("empty choices in vision response")
            content = choices[0].get("message", {}).get("content") or ""
            return _THINK_RE.sub("", content).strip()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if (
                status == 400
                and "temperature" in data
                and self._is_unsupported_temperature(e.response)
            ):
                # See _request(): don't gate on `model not in self._no_temp_models`,
                # concurrent in-flight calls must each be able to self-heal.
                self._no_temp_models.add(self._cfg.model)
                return await self._request_vision(system, prompt, image_b64, retry)
            if (status == 429 or status >= 500) and retry < self._cfg.retry:
                await asyncio.sleep(self._retry_delay(retry, e.response))
                return await self._request_vision(system, prompt, image_b64, retry + 1)
            raise
        except Exception:
            if retry < self._cfg.retry:
                await asyncio.sleep(self._retry_delay(retry))
                return await self._request_vision(system, prompt, image_b64, retry + 1)
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
