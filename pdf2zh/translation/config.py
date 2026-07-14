from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

PROVIDERS: dict[str, dict[str, str]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "google/gemini-3.1-flash-lite",
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
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "env_var": "DEEPSEEK_API_KEY",
    },
    "minimax": {
        "base_url": "https://api.minimax.io/v1",
        "model": "MiniMax-Text-01",
        "env_var": "MINIMAX_API_KEY",
    },
    # Anthropic via its OpenAI-compatible endpoint (Bearer auth, /chat/completions).
    # Default to the cheap/fast tier like the other providers; override in the UI
    # model box (e.g. claude-sonnet-4-6, claude-opus-4-8) for higher quality.
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-haiku-4-5",
        "env_var": "ANTHROPIC_API_KEY",
    },
    # LiteLLM proxy — base_url is deployment-specific; set LITELLM_BASE_URL to point
    # at your proxy. The model is whatever your proxy routes.
    "litellm": {
        "base_url": "http://localhost:4000/v1",
        "base_url_env": "LITELLM_BASE_URL",
        "model": "gpt-4o-mini",
        "env_var": "LITELLM_API_KEY",
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
    concurrent: int = 8  # was 30 — 30 bursts most providers straight into 429
    rpm: int | None = None
    tpm: int | None = None
    chunk_bytes: int = 3000
    glossary_enabled: bool = True
    math_fix_enabled: bool = True
    toc_fix_enabled: bool = True
    equation_vision_enabled: bool = True
    table_vision_enabled: bool = True
    length_tolerance: float = 0.15
    timeout: int = 300
    retry: int = 5  # was 2 — transient 429/5xx need a larger budget to ride out
    # OpenRouter's unified `reasoning` request field works across many providers/
    # models it proxies (Qwen, DeepSeek R1, ...). Only meaningful when
    # provider == "openrouter" — other providers get this via _no_temp_models-style
    # per-model quirks instead (see gateway.py).
    disable_reasoning: bool = False


def provider_base_url(provider: str) -> str:
    """Resolve a provider's base URL, honoring its optional ``base_url_env`` override."""
    p = PROVIDERS.get(provider)
    if p is None:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose from {list(PROVIDERS)}."
        )
    env_key = p.get("base_url_env")
    return (os.environ.get(env_key) if env_key else None) or p["base_url"]


def resolve_provider(cfg: TranslatorConfig) -> None:
    load_dotenv()
    p = PROVIDERS.get(cfg.provider)
    if p is None:
        raise ValueError(
            f"Unknown provider '{cfg.provider}'. Choose from {list(PROVIDERS)}."
        )
    if cfg.base_url is None:
        cfg.base_url = provider_base_url(cfg.provider)
    if cfg.model is None:
        cfg.model = p["model"]
    if cfg.api_key is None:
        cfg.api_key = os.environ.get(p["env_var"])
    if not cfg.api_key:
        raise ValueError(f"No API key found. Set {p['env_var']} or pass --api-key.")
