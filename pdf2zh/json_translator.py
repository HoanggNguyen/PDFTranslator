"""
How to run:
    python -m pdf2zh.json_translator input.json --api-key $KEY
"""

import logging
import time

from pdf2zh.translation import (  # noqa: F401
    PROVIDERS,
    Gateway,
    RateLimiter,
    Task,
    TranslatorConfig,
    build_glossary_prompt,
    build_translation_prompt,
    collect_translatables,
    extract_glossary,
    glossary_block_for_chunk,
    is_equation_only,
    is_plain_text,
    resolve_provider,
    segments_to_chunks,
    translate_chunks,
    translate_document,
)
from pdf2zh.translation.cli import main  # noqa: F401

logger = logging.getLogger("json_translator")

if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    logger.info(f"Total time: {time.perf_counter() - t0:.1f}s")
