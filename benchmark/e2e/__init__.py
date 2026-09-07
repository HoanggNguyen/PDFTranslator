"""End-to-end evaluation harness: PDFTranslator vs BabelDOC / PDFMathTranslate / DeepL.

Where this fits: ``benchmark/parser/`` evaluates Phase 1 (layout/OCR accuracy vs
OmniDocBench GT) and ``benchmark/translation/`` evaluates Phase 2 (COMET/chrF++ on
WMT24++). Neither measures Phase 3 or the document as a whole. This package covers
that gap: it runs whole PDFs through every system and scores the *output PDF* —
layout preservation, visual fidelity, content integrity, translation quality, cost.

Everything here lives OUTSIDE ``pdf2zh/`` and only calls its public entrypoints
(``pdf2zh.e2e``) or shells out to the baselines. ``git diff pdf2zh/`` must stay
empty. Full design: docs/EVALUATION_PLAN.md.
"""

# Environment exports win over .env; remote Jobs never load a local credentials file.
import os
from pathlib import Path

if not os.environ.get("BENCH_REMOTE"):
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass  # Pure stdlib mock tests may run without optional dependencies.
    else:
        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
