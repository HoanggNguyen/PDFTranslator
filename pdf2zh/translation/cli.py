from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import PROVIDERS, TranslatorConfig
from .pipeline import translate_document

logger = logging.getLogger("json_translator")


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate PDFTranslator JSON output")
    parser.add_argument("input", help="Input JSON file")
    parser.add_argument(
        "-o", "--output", help="Output JSON file (default: INPUT.translated.json)"
    )
    parser.add_argument("--src", dest="source_language", default="")
    parser.add_argument("--tgt", dest="target_language", default="")
    parser.add_argument("--provider", default="openrouter", choices=list(PROVIDERS))
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", dest="api_key", default=None)
    parser.add_argument("--concurrent", type=int, default=30)
    parser.add_argument("--chunk-bytes", type=int, default=3000)
    parser.add_argument("--no-glossary", action="store_true")
    parser.add_argument("--length-tolerance", type=float, default=0.15)
    parser.add_argument("--rpm", type=int, default=None)
    parser.add_argument("--tpm", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    with open(args.input, encoding="utf-8") as f:
        doc = json.load(f)

    cfg = TranslatorConfig(
        source_language=args.source_language or doc.get("source_language", ""),
        target_language=args.target_language or doc.get("target_language", ""),
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        concurrent=args.concurrent,
        chunk_bytes=args.chunk_bytes,
        glossary_enabled=not args.no_glossary,
        length_tolerance=args.length_tolerance,
        rpm=args.rpm,
        tpm=args.tpm,
    )

    doc = translate_document(doc, cfg)

    output = args.output or (str(Path(args.input).with_suffix("")) + ".translated.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved to {output}")
