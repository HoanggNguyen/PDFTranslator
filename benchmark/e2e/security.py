"""Credential handling for the benchmark (never a guarantee against malicious code)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


def secret_name(name: str) -> bool:
    if name.upper() in {"LITELLM_KEY_ALIAS", "KEY_ALIAS_PREFIX"}:
        return False
    return bool(
        re.search(
            r"(?:TOKEN|SECRET|PASSWORD|API_KEY|AUTH_KEY|LITELLM_KEY_)", name.upper()
        )
    )


def secret_values(env=None) -> list[str]:
    env = os.environ if env is None else env
    return sorted(
        {v for k, v in env.items() if secret_name(k) and v}, key=len, reverse=True
    )


def redact(text: str, values=None) -> str:
    for value in secret_values() if values is None else values:
        text = text.replace(value, "[REDACTED]")
    # Also catch common credentials from earlier runs, no longer in this env.
    text = re.sub(
        r"\b(?:hf_[A-Za-z0-9]{12,}|sk-[A-Za-z0-9_-]{12,})", "[REDACTED]", text
    )
    text = re.sub(r"(?i)(Bearer\s+)[^\s\"']+", r"\1[REDACTED]", text)
    return text


def safe_json(value) -> str:
    def clean(item):
        if isinstance(item, str):
            return redact(item)
        if isinstance(item, dict):
            return {redact(str(k)): clean(v) for k, v in item.items()}
        if isinstance(item, list):
            return [clean(v) for v in item]
        return item

    return json.dumps(clean(value), indent=2, ensure_ascii=False)


def clean_env(env=None, keep=()) -> dict[str, str]:
    env = os.environ if env is None else env
    return {k: v for k, v in env.items() if not secret_name(k) or k in keep}


def scan_file(path: Path) -> None:
    """Fail closed on known secrets in any uploaded file, including binary files."""
    values = [v.encode() for v in secret_values()]
    overlap = max([len(v) for v in values] + [256])
    tail = b""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            data = tail + chunk
            if any(v in data for v in values) or re.search(
                rb"\b(?:hf_[A-Za-z0-9]{12,}|sk-[A-Za-z0-9_-]{12,})", data
            ):
                raise ValueError(f"Credential detected; refusing upload: {path.name}")
            tail = data[-overlap:]


def publishable(relative: Path) -> bool:
    """No raw subprocess directories, credentials, dotfiles or arbitrary binaries."""
    if any(p.startswith(".") or p in {"raw", "__pycache__"} for p in relative.parts):
        return False
    if re.search(r"(?i)(config|credential|secret|token)", relative.name):
        return False
    return relative.suffix.lower() in {
        ".pdf",
        ".json",
        ".jsonl",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".csv",
        ".md",
        ".log",
        ".txt",
    }
