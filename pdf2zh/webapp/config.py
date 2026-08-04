"""UI-facing configuration: provider/page presets and helpers (no Gradio deps)."""

from __future__ import annotations

# UI label -> Phase-2 provider key.
PROVIDER_KEY = {
    "OpenRouter": "openrouter",
    "Gemini": "gemini",
    "OpenAI": "openai",
    "DeepSeek": "deepseek",
    "MiniMax": "minimax",
    "Anthropic": "anthropic",
    "LiteLLM": "litellm",
}
PROVIDER_CHOICES = list(PROVIDER_KEY)

# Default model placeholders (mirror translation/config.py PROVIDERS).
PROVIDER_DEFAULT_MODEL = {
    "OpenRouter": "google/gemini-2.5-flash-lite",
    "Gemini": "gemini-2.5-flash-lite",
    "OpenAI": "gpt-4o-mini",
    "DeepSeek": "deepseek-chat",
    "MiniMax": "MiniMax-Text-01",
    "Anthropic": "claude-haiku-4-5",
    "LiteLLM": "gpt-4o-mini",
}

# Page-selection modes. "All" translates the whole document; "Range" uses the
# 1-based from/to boxes.
PAGE_MODE_ALL = "Toàn bộ"
PAGE_MODE_RANGE = "Khoảng trang"
PAGE_MODES = [PAGE_MODE_ALL, PAGE_MODE_RANGE]
MAX_CUSTOM_PAGES = 50  # guardrail against OOM on a single T4


def resolve_pages(mode: str, from_page, to_page) -> list[int] | None:
    """Map the page mode (+ 1-based from/to) to a 0-based page index list or None.

    ``All`` → None (whole document). ``Range`` → inclusive 1-based ``[from, to]``
    converted to 0-based indices, with the span capped at ``MAX_CUSTOM_PAGES``.
    Out-of-range high values are harmless: the parser and compositor both drop
    indices past the document's last page.
    """
    if mode != PAGE_MODE_RANGE:
        return None
    lo = max(1, int(from_page or 1))
    hi = max(lo, int(to_page or lo))
    hi = min(hi, lo + MAX_CUSTOM_PAGES - 1)  # cap the span
    return list(range(lo - 1, hi))
