"""UI-facing configuration: provider/page presets and helpers (no Gradio deps)."""

from __future__ import annotations

# UI label -> Phase-2 provider key.
PROVIDER_KEY = {"OpenRouter": "openrouter", "Gemini": "gemini", "OpenAI": "openai"}
PROVIDER_CHOICES = list(PROVIDER_KEY)

# Default model placeholders (mirror translation/config.py PROVIDERS).
PROVIDER_DEFAULT_MODEL = {
    "OpenRouter": "google/gemini-2.5-flash-lite",
    "Gemini": "gemini-2.5-flash-lite",
    "OpenAI": "gpt-4o-mini",
}

# Page selection presets. CUSTOM_LABEL -> read the page count from the "First N" box.
CUSTOM_LABEL = "First N…"
PAGE_PRESETS: dict[str, object] = {
    "All": None,
    "First page": [0],
    "First 5 pages": list(range(5)),
    CUSTOM_LABEL: "CUSTOM",
}
MAX_CUSTOM_PAGES = 50  # guardrail against OOM on a single T4


def resolve_pages(page_choice: str, page_n) -> list[int] | None:
    """Map a preset label (+ the custom N) to a 0-based page index list or None."""
    sel = PAGE_PRESETS[page_choice]
    if sel == "CUSTOM":
        n = max(1, min(int(page_n or 1), MAX_CUSTOM_PAGES))
        return list(range(n))
    return sel  # None or a list
