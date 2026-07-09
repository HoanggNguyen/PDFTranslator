"""Benchmark ``StageAParser.parse_pdf`` over the batched OmniDocBench PDFs.

Workflow (see build_pdfs.py first):
  images -> batched PDFs (32 pages each) -> THIS script -> per-PDF JSON + timing.

To measure parse time accurately, all three models (layout / OCR / table) are
loaded into VRAM *before* the timed loop, and an optional warmup parse is run
on the first PDF (discarded) so CUDA kernel autotuning does not pollute the
first real measurement.

For each PDF we call ``parse_pdf`` and save the ``ParsedDocument`` as
``<pdf_stem>.json``; split back to per-image results later using the
``mapping.json`` produced by build_pdfs.py.

Example
-------
    python run_parser.py \
        --pdfs   /media/lhbac32/OmniDocBench_data/pdfs \
        --out    /media/lhbac32/results/parser_json \
        --timing /media/lhbac32/results/parser_timing.json \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdfs", required=True, type=Path,
                        help="Folder of batched PDFs (from build_pdfs.py).")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output folder for per-PDF ParsedDocument JSON.")
    parser.add_argument("--timing", type=Path, default=None,
                        help="JSON timing report path (default: <out>/_timing.json).")
    parser.add_argument("--pdftranslator", type=Path, default=None,
                        help="PDFTranslator repo root (default: sibling ../PDFTranslator).")
    parser.add_argument("--device", default="auto",
                        help="Torch device for StageAParser (auto|cuda|cpu).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N PDFs (quick test).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run PDFs whose JSON already exists (default: skip/resume).")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false",
                        help="Skip the warmup parse of the first PDF.")
    parser.add_argument("--no-sparse-refine", dest="refine_sparse", action="store_false",
                        help="Tắt is_sparse_text_block -> dump output THÔ (before split).")
    return parser.parse_args()


def resolve_pdftranslator(arg: Path | None) -> Path:
    if arg is not None:
        if (arg / "pdf2zh").is_dir():
            return arg.resolve()
        raise FileNotFoundError(f"pdf2zh not found under --pdftranslator {arg}")
    # Walk up from this file until we find the PDFTranslator repo root (has pdf2zh).
    for parent in Path(__file__).resolve().parents:
        if (parent / "pdf2zh").is_dir():
            return parent
    raise FileNotFoundError(
        "Cannot locate the PDFTranslator repo root (no pdf2zh/ found above "
        f"{__file__}). Pass --pdftranslator explicitly."
    )


def preload_models(parser) -> float:
    """Load all model weights into VRAM. Returns seconds spent."""
    start = time.perf_counter()
    for name in ("layout_model", "ocr_model", "table_model"):
        model = getattr(parser, name, None)
        if model is not None and getattr(model, "model", None) is None:
            print(f"[run_parser] loading {name} ...", flush=True)
            model.load_model()
    return time.perf_counter() - start


def main() -> int:
    args = parse_args()

    pdftranslator = resolve_pdftranslator(args.pdftranslator)
    sys.path.insert(0, str(pdftranslator))

    from pdf2zh.parser.main import StageAParser

    pdfs = sorted(p for p in args.pdfs.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No .pdf files found in {args.pdfs}")
    if args.limit is not None:
        pdfs = pdfs[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    timing_path = args.timing or (args.out / "_timing.json")

    print(f"[run_parser] {len(pdfs)} PDFs from {args.pdfs}", flush=True)
    print(f"[run_parser] device={args.device}  out={args.out}", flush=True)

    # 1) Construct parser (declares models, no weights yet — lazy loading).
    init_start = time.perf_counter()
    parser = StageAParser(
        device=args.device,
        page_batch_size=32,
        layout_batch_size=32,
        detection_batch_size=32,
        ocr_batch_size=512,
        table_batch_size=512,
        detector_blank_threshold=0.5,
        detector_text_threshold=0.6,
        refine_sparse_blocks=args.refine_sparse,
    )
    init_seconds = time.perf_counter() - init_start

    # 2) Preload ALL model weights before timing anything.
    model_load_seconds = preload_models(parser)
    print(f"[run_parser] parser init: {init_seconds:.1f}s  "
          f"model load: {model_load_seconds:.1f}s", flush=True)

    # 3) Warmup parse (discarded) so CUDA autotune doesn't skew the first PDF.
    warmup_seconds = None
    if args.warmup and pdfs:
        w_start = time.perf_counter()
        try:
            parser.parse_pdf(str(pdfs[0]))
            warmup_seconds = time.perf_counter() - w_start
            print(f"[run_parser] warmup parse: {warmup_seconds:.2f}s (discarded)",
                  flush=True)
        except Exception as exc:
            print(f"[run_parser] warmup failed (ignored): {exc!r}", flush=True)

    # 4) Timed loop over all PDFs.
    per_pdf: list[dict] = []
    processed = skipped = failed = 0
    total_parse_seconds = 0.0
    total_pages = 0

    for idx, pdf_path in enumerate(pdfs, start=1):
        out_path = args.out / f"{pdf_path.stem}.json"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        record: dict = {"pdf": pdf_path.name, "index": idx}
        start = time.perf_counter()
        try:
            doc = parser.parse_pdf(str(pdf_path))
            elapsed = time.perf_counter() - start
            doc.save(out_path)

            num_pages = len(doc.pages)
            num_elements = sum(len(p.elements) for p in doc.pages)
            record.update(
                seconds=round(elapsed, 4),
                num_pages=num_pages,
                num_elements=num_elements,
                seconds_per_page=round(elapsed / num_pages, 4) if num_pages else None,
                ok=True,
            )
            total_parse_seconds += elapsed
            total_pages += num_pages
            processed += 1
            print(f"[{idx}/{len(pdfs)}] {pdf_path.name}  {elapsed:.2f}s  "
                  f"pages={num_pages}  elements={num_elements}", flush=True)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            record.update(seconds=round(elapsed, 4), ok=False, error=repr(exc))
            failed += 1
            print(f"[{idx}/{len(pdfs)}] {pdf_path.name}  FAILED: {exc!r}", flush=True)

        per_pdf.append(record)

    report = {
        "device": args.device,
        "pdfs_dir": str(args.pdfs),
        "out_dir": str(args.out),
        "num_pdfs_total": len(pdfs),
        "num_processed": processed,
        "num_skipped_existing": skipped,
        "num_failed": failed,
        "parser_init_seconds": round(init_seconds, 4),
        "model_load_seconds": round(model_load_seconds, 4),
        "warmup_seconds": round(warmup_seconds, 4) if warmup_seconds is not None else None,
        "total_parse_seconds": round(total_parse_seconds, 4),
        "total_pages": total_pages,
        "avg_seconds_per_pdf": round(total_parse_seconds / processed, 4) if processed else None,
        "avg_seconds_per_page": round(total_parse_seconds / total_pages, 4) if total_pages else None,
        "per_pdf": per_pdf,
    }
    timing_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[run_parser] done.", flush=True)
    print(f"  processed={processed}  skipped={skipped}  failed={failed}", flush=True)
    print(f"  model load : {model_load_seconds:.1f}s", flush=True)
    print(f"  total parse: {total_parse_seconds:.1f}s over {total_pages} pages", flush=True)
    if report["avg_seconds_per_page"] is not None:
        print(f"  avg/page   : {report['avg_seconds_per_page']:.3f}s", flush=True)
    print(f"  timing     : {timing_path}", flush=True)

    return 1 if (processed == 0 and skipped == 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
