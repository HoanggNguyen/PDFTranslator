from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

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
        raise ValueError(
            f"Unknown provider '{cfg.provider}'. Choose from {list(PROVIDERS)}."
        )
    if cfg.base_url is None:
        cfg.base_url = p["base_url"]
    if cfg.model is None:
        cfg.model = p["model"]
    if cfg.api_key is None:
        cfg.api_key = os.environ.get(p["env_var"])
    if not cfg.api_key:
        raise ValueError(f"No API key found. Set {p['env_var']} or pass --api-key.")
