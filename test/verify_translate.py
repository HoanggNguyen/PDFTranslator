#!/usr/bin/env python3
"""Translation pipeline verification tool.

Runs the full translation pipeline on a JSON file and prints a detailed
summary of segments found, translations applied, length violations, and
glossary terms — useful for manual debugging and smoke-testing.

Usage:
    python test/verify_translate.py --input test/fixtures/mini_output.json --api-key $KEY
    python test/verify_translate.py --input doc.json --provider gemini --api-key $KEY
    python test/verify_translate.py --input doc.json --no-glossary --verbose
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf2zh.translation import (
    TranslatorConfig,
    collect_translatables,
    segments_to_chunks,
    translate_document,
)
from pdf2zh.translation.config import PROVIDERS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def verify_translation(
    input_path: str,
    output_path: str | None,
    cfg: TranslatorConfig,
) -> None:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")

    with open(path, encoding="utf-8") as f:
        doc = json.load(f)

    tasks = collect_translatables(doc)
    chunks = segments_to_chunks(tasks, cfg.chunk_bytes)

    logger.info(f"Translating {path}  ({len(tasks)} segments, {len(chunks)} chunks)...")
    out = translate_document(doc, cfg)

    out_path = (
        Path(output_path) if output_path else path.with_suffix(".translated.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Collect stats after translation
    translated_count = 0
    untranslated: list[dict] = []
    violations: list[dict] = []

    for task in tasks:
        result = task.target.get(task.write_key, "")
        if result and result != task.text:
            translated_count += 1
            if len(task.text) >= 20:
                ratio = abs(len(result) - len(task.text)) / max(len(task.text), 1)
                if ratio > cfg.length_tolerance:
                    violations.append(
                        {
                            "id": task.id,
                            "src_len": len(task.text),
                            "out_len": len(result),
                            "ratio": f"{ratio:.1%}",
                            "src": task.text[:60],
                            "out": result[:60],
                        }
                    )
        else:
            untranslated.append({"id": task.id, "text": task.text[:60]})

    print("\nTranslation Summary")
    print("=" * 70)
    print(f"Input:      {path}")
    print(f"Output:     {out_path}")
    print(f"Provider:   {cfg.provider}  model={cfg.model}")
    print(f"Segments:   {len(tasks)}")
    print(f"Chunks:     {len(chunks)}")
    print(f"Translated: {translated_count}/{len(tasks)}")

    if violations:
        print(f"\nLength violations ({len(violations)}):")
        for v in violations:
            print(
                f"  id={v['id']}  {v['ratio']}  src={v['src_len']}ch  out={v['out_len']}ch"
            )
            print(f"    src: {v['src']}")
            print(f"    out: {v['out']}")
    else:
        print("\nLength violations: none")

    if untranslated:
        print(f"\nUntranslated ({len(untranslated)}):")
        for u in untranslated:
            print(f"  id={u['id']}: {u['text']}")

    print("\nSegment detail:")
    w = 45
    print(f"  {'ID':<4} {'FIELD':<20} {'SRC':<{w}} {'OUT':<{w}} STATUS")
    print("  " + "-" * (4 + 1 + 20 + 1 + w + 1 + w + 1 + 6))
    for task in tasks:
        result = task.target.get(task.write_key, "")
        status = "OK  " if (result and result != task.text) else "MISS"
        src_display = task.text[: w - 2] + ".." if len(task.text) > w else task.text
        out_display = result[: w - 2] + ".." if len(result) > w else result
        print(
            f"  {task.id:<4} {task.write_key:<20} {src_display:<{w}} {out_display:<{w}} {status}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify translation pipeline with detailed output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test/verify_translate.py --input test/fixtures/mini_output.json --api-key $KEY
    python test/verify_translate.py --input doc.json --provider gemini --api-key $KEY
    python test/verify_translate.py --input doc.json --no-glossary --verbose
        """,
    )
    parser.add_argument("--input", "-i", required=True, help="Input JSON file")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file")
    parser.add_argument("--src", dest="source_language", default="")
    parser.add_argument("--tgt", dest="target_language", default="")
    parser.add_argument("--provider", default="openrouter", choices=list(PROVIDERS))
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", dest="api_key", default=None)
    parser.add_argument("--concurrent", type=int, default=5)
    parser.add_argument("--chunk-bytes", type=int, default=3000)
    parser.add_argument("--no-glossary", action="store_true")
    parser.add_argument("--length-tolerance", type=float, default=0.15)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = TranslatorConfig(
        source_language=args.source_language,
        target_language=args.target_language,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        concurrent=args.concurrent,
        chunk_bytes=args.chunk_bytes,
        glossary_enabled=not args.no_glossary,
        length_tolerance=args.length_tolerance,
    )

    try:
        verify_translation(args.input, args.output, cfg)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
