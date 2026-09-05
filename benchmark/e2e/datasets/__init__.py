"""Corpus builders for the three evaluation tiers (see docs/EVALUATION_PLAN.md §2).

  T1  build_doclaynet.py   DocLayNet-v1.2 test split — born-digital PDFs with
                           human-annotated layout GT. The primary layout tier.
  T2  build_t2_corpus.py   multi-page arXiv / technical docs / patents.
  T3  (no builder needed)  reuse benchmark/parser/evaluation/download_dataset.py
                           + benchmark/parser/run_parser/build_pdfs.py.

``verify_corpus.py`` gates all of them: run it before spending money on a job.

Run these as modules from the repo root so this package does not shadow the
HuggingFace ``datasets`` library::

    python -m benchmark.e2e.datasets.build_doclaynet --help
"""
