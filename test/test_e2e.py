"""End-to-end orchestration tests (pdf2zh.e2e).

Covers the per-phase latency log emitted by ``run_pipeline`` (used only by the
UI's end-to-end button; the stepped flow calls the phases directly).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pdf2zh.e2e as e2e


def test_run_pipeline_logs_per_phase_latency(monkeypatch, caplog):
    # Stub the three phases so no models / network / typst are exercised.
    monkeypatch.setattr(e2e, "run_parse", lambda *a, **k: {"pages": []})
    monkeypatch.setattr(e2e, "run_translate", lambda *a, **k: {"pages": []})
    monkeypatch.setattr(e2e, "run_render", lambda *a, **k: "/tmp/out.pdf")

    with caplog.at_level("INFO", logger="pdf2zh.e2e"):
        out = e2e.run_pipeline(
            pdf_path="in.pdf",
            src_lang="English",
            tgt_lang="Vietnamese",
            provider="openrouter",
            api_key="key",
            model=None,
            pages=None,
            font="Noto Sans",
            work_dir="/tmp/wd",
        )

    assert out == "/tmp/out.pdf"
    latency_lines = [
        r.getMessage() for r in caplog.records if "[latency]" in r.getMessage()
    ]
    assert len(latency_lines) == 1
    # The line reports all four numbers.
    for key in ("parse=", "translate=", "render=", "total="):
        assert key in latency_lines[0]
