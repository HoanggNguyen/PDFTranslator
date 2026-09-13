"""Offline contract/security/orchestration tests. Never calls HF, OCR models or LLMs."""

import importlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import fitz
import pytest

from benchmark.e2e import hf_driver as D, hf_worker as W, protocol as P, sync
from benchmark.e2e.manifest import verify
from benchmark.e2e.metrics import aggregate as A, eval_visual as V
from benchmark.e2e.runners import _common as C, babeldoc
from benchmark.e2e.security import publishable


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "source"
    tier = root / "T1"
    tier.mkdir(parents=True)
    pdf = tier / "sample.pdf"
    gt, mapping = [], []
    with fitz.open() as document:
        for i in range(3):
            page = document.new_page(width=400, height=400)
            page.insert_text(
                (40, 60),
                "The benchmark translates the entire document and preserves 123.",
            )
            gt.append(
                {
                    "page": i,
                    "width": 400,
                    "height": 400,
                    "elements": [{"class": "Text", "bbox_norm": [0.1, 0.1, 0.9, 0.2]}],
                }
            )
            mapping.append(
                {"page": i, "original_filename": f"original-{i}.pdf", "n_chars": 70}
            )
        document.save(pdf)
    (tier / "gt.json").write_text(
        json.dumps({"docs": [{"doc_id": "sample", "pdf": "sample.pdf", "pages": gt}]})
    )
    (tier / "mapping.json").write_text(
        json.dumps(
            {
                "docs": [
                    {
                        "doc_id": "sample",
                        "pdf": "sample.pdf",
                        "pages": mapping,
                        "sha256": P.sha256(pdf),
                    }
                ]
            }
        )
    )
    return root


@pytest.fixture
def prepared(tmp_path, corpus, monkeypatch):
    monkeypatch.setattr(D, "ROOT", tmp_path)
    monkeypatch.setenv("BENCH_MODEL", "mock-model")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://proxy.example/v1")
    args = D.parser().parse_args(
        [
            "prepare-smoke",
            "--source",
            str(corpus),
            "--run-id",
            "smoke-test",
            "--pages",
            "2",
        ]
    )
    D.prepare(args)
    root = D.run_root("smoke-test")
    return root, P.load_contract(root / "corpus")


def fake_translate(cmd, log, env):
    """Three fake translators produce valid PDFs via the REAL worker interface."""
    system = cmd[2].rsplit(".", 1)[-1]
    corpus, out = (
        Path(cmd[cmd.index("--corpus") + 1]),
        Path(cmd[cmd.index("--out") + 1]),
    )
    lang = cmd[cmd.index("--langs") + 1]
    source = next(corpus.rglob("*.pdf"))
    dest = out / system / lang / source.stem
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest / "output.pdf")
    with fitz.open(source) as pdf:
        pages = len(pdf)
    (dest / "meta.json").write_text(
        json.dumps(
            {
                "system": system,
                "lang": lang,
                "doc_id": source.stem,
                "sha256": P.sha256(source),
                "model": "mock-model",
                "error": None,
                "n_pages_in": pages,
                "n_pages_out": pages,
                "wall_seconds": 2,
                "ts": "2026-09-05T00:00:00Z",
            }
        )
    )
    log.write_text("MOCK translator; no API used\n")
    assert "HF_TOKEN" not in env and "HF_JOB_TOKEN" not in env
    return 0, 3.0


def populate(root, contract):
    upload = Mock(return_value=0)
    for system in P.SYSTEMS[:3]:
        assert (
            W.translate(
                root / "corpus",
                root / "out",
                contract,
                system,
                upload,
                executor=fake_translate,
            )
            == 0
        )
    assert upload.call_count == 3


def test_smoke_subset_rewrites_pdf_gt_hash(prepared):
    root, c = prepared
    assert len(list((root / "corpus").rglob("*.pdf"))) == 1
    with fitz.open(root / "corpus/T1/sample.pdf") as pdf:
        assert len(pdf) == 2
    assert (
        len(json.loads((root / "corpus/T1/gt.json").read_text())["docs"][0]["pages"])
        == 2
    )
    assert c["mode"] == "smoke" and c["systems"] == list(P.SYSTEMS[:3])


def test_resume_no_calls_and_changed_output_rejected(prepared):
    root, c = prepared
    populate(root, c)
    assert P.gate(root / "corpus", root / "out", c)["success"] == 3
    execute = Mock(side_effect=AssertionError("resume must not call translation"))
    W.translate(
        root / "corpus",
        root / "out",
        c,
        "babeldoc",
        Mock(return_value=0),
        executor=execute,
    )
    assert execute.call_count == 0
    (root / "out/babeldoc/vi/sample/output.pdf").write_bytes(b"bad")
    with pytest.raises(ValueError, match="output"):
        W.translate(
            root / "corpus",
            root / "out",
            c,
            "babeldoc",
            Mock(return_value=0),
            executor=execute,
        )


def test_missing_system_and_gt_drift_fail(prepared):
    root, c = prepared
    W.translate(
        root / "corpus",
        root / "out",
        c,
        "pdftranslator",
        Mock(return_value=0),
        executor=fake_translate,
    )
    with pytest.raises(ValueError, match="metadata"):
        P.gate(root / "corpus", root / "out", c)
    errors, _ = verify(root / "out", list(P.SYSTEMS[:3]), ["vi"])
    assert errors
    gt = root / "corpus/T1/gt.json"
    data = json.loads(gt.read_text())
    data["docs"][0]["pages"][0]["elements"][0]["bbox_norm"][0] = 0.11
    gt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="changed"):
        P.load_contract(root / "corpus")


def test_failed_runner_checkpoint_and_explicit_failure_gate(prepared):
    root, c = prepared
    called = Mock(return_value=0)

    def fail(*args):
        raise RuntimeError("simulated subprocess startup failure")

    assert (
        W.translate(
            root / "corpus", root / "out", c, "pdftranslator", called, executor=fail
        )
        == 1
    )
    called.assert_called_once()
    with pytest.raises(ValueError, match="failed"):
        P.gate(root / "corpus", root / "out", c, ["pdftranslator"])
    assert (
        P.gate(
            root / "corpus", root / "out", c, ["pdftranslator"], allow_failures=True
        )["failed"]
        == 1
    )


def test_runtime_secret_selection(prepared, monkeypatch):
    _, c = prepared
    for key, value in {
        "HF_TOKEN": "LOCAL_SECRET_123",
        "HF_JOB_TOKEN": "RUNTIME_SECRET_456",
        "HF_EVAL_REPO": "user/eval",
        "HF_BENCH_SPACE": "user/bench",
        **{"LITELLM_KEY_" + s.upper(): "TEST_SECRET_" + s for s in P.SYSTEMS[:3]},
    }.items():
        monkeypatch.setenv(key, value)
    for system in P.SYSTEMS[:3]:
        req = D.submission(c, "translate", system)
        assert req["secrets"] == {
            "HF_TOKEN": "RUNTIME_SECRET_456",
            "LITELLM_API_KEY": "TEST_SECRET_" + system,
        }
        assert "TEST_SECRET" not in json.dumps(req["env"]) + json.dumps(req["command"])
        assert "LOCAL_SECRET" not in json.dumps(req)
    for action in ("check", "warm", "score"):
        assert set(D.submission(c, action)["secrets"]) == {"HF_TOKEN"}


def test_child_logs_redacted_and_no_key_in_babeldoc_command(tmp_path, monkeypatch):
    secret = "TEST_SECRET_LOG_VALUE"
    monkeypatch.setenv("LITELLM_API_KEY", secret)
    monkeypatch.setenv("HF_TOKEN", "TEST_HF_CONTROLLER")
    env = C.child_env({"OPENAI_API_KEY": secret})
    assert "HF_TOKEN" not in env and "LITELLM_API_KEY" not in env
    log = tmp_path / "run.log"
    cmd = [
        sys.executable,
        "-c",
        "import os; print(os.environ['OPENAI_API_KEY'])",
        "--api-key",
        secret,
    ]
    assert C.run_child(cmd, log, 10, env, tmp_path)[0] == 0
    assert secret not in log.read_text() and "[REDACTED]" in log.read_text()
    args = SimpleNamespace(
        model="test", base_url="https://proxy.example/v1", qps=8, no_glossary=False
    )
    cmd = babeldoc.build_cmd(
        ["babeldoc"], Path("in.pdf"), Path("out"), "vi", args, secret
    )
    # babeldoc v0.6.4 requires the key in argv (no env fallback); run_child
    # (verified above) redacts the flagged value from every log line.
    assert cmd[cmd.index("--openai-api-key") + 1] == secret
    assert "--ignore-cache" in cmd


def test_upload_allowlist_scans_exact_staged_files(tmp_path, monkeypatch):
    secret = "TEST_SECRET_UPLOAD_VALUE"
    monkeypatch.setenv("HF_TOKEN", secret)
    root = tmp_path / "out"
    root.mkdir()
    (root / "meta.json").write_text('{"ok":true}')
    (root / ".env").write_text(secret)
    (root / "raw").mkdir()
    (root / "raw/config.json").write_text(secret)
    api = Mock()

    def upload(**kwargs):
        files = sorted(
            str(p.relative_to(kwargs["folder_path"]))
            for p in Path(kwargs["folder_path"]).rglob("*")
            if p.is_file()
        )
        assert files == ["meta.json"]

    api.upload_folder.side_effect = upload
    assert sync.do_push(api, "u/r", tmp_path, [("out", "smoke/out")], None, False) == 0
    api.upload_folder.assert_called_once()
    (root / "run.log").write_text(secret)
    api.reset_mock()
    assert sync.do_push(api, "u/r", tmp_path, [("out", "smoke/out")], None, False) == 1
    api.upload_folder.assert_not_called()


def test_scorer_preflight_blocks_missing_before_any_execution(prepared):
    root, c = prepared
    execute = Mock()
    with pytest.raises(ValueError):
        W.scoring(root / "corpus", root / "out", c, Mock(), executor=execute)
    execute.assert_not_called()


def test_scorer_failure_status_is_uploaded(prepared):
    root, c = prepared
    populate(root, c)
    upload = Mock(return_value=0)
    with pytest.raises(RuntimeError, match="Scoring step"):
        W.scoring(
            root / "corpus", root / "out", c, upload, executor=lambda *args: (1, 1)
        )
    assert not json.loads((root / "out/_run/score-status.json").read_text())[
        "completed"
    ]
    upload.assert_called_once()


def test_real_scoring_io_with_mocked_models(prepared, monkeypatch):
    """Actual rendering/alignment/metrics/report, fake detector+QE+LID models only."""
    root, c = prepared
    populate(root, c)
    from benchmark.e2e.metrics import langid, eval_text, eval_qe

    fake_lid = lambda *a, **k: SimpleNamespace(
        backend="fasttext", predict=lambda text: ("en", 0.99)
    )
    monkeypatch.setattr(langid, "LangID", fake_lid)
    monkeypatch.setattr(eval_text, "LangID", fake_lid)
    monkeypatch.setattr(eval_qe, "load_model", lambda *a: object())
    monkeypatch.setattr(
        eval_qe, "predict", lambda model, data, *a: [0.75 for _ in data]
    )
    from benchmark.e2e.parse import run_detectors

    class Detector:
        def __call__(self, images):
            return [
                [
                    {
                        "class": "Text",
                        "group": "text",
                        "xyxy": [
                            0.1 * im.width,
                            0.1 * im.height,
                            0.9 * im.width,
                            0.2 * im.height,
                        ],
                        "score": 1.0,
                    }
                ]
                for im in images
            ]

    monkeypatch.setitem(run_detectors.DETECTORS, "docling", Detector)
    executed = []

    def execute(cmd, log, env):
        executed.append(cmd[2])
        assert "LITELLM_API_KEY" not in env
        monkeypatch.setattr(sys, "argv", [cmd[2], *cmd[3:]])
        return importlib.import_module(cmd[2]).main(), 0.1

    assert (
        W.scoring(
            root / "corpus", root / "out", c, Mock(return_value=0), executor=execute
        )
        == 0
    )
    report = (root / "out/report/report.md").read_text()
    for system in P.SYSTEMS[:3]:
        assert system in report
        assert (root / f"out/_metrics/qe/{system}.vi.json").exists()
    assert "COMETKiwi" in report and "document-macro" in report
    assert len(executed) == 10
    assert json.loads((root / "out/_run/score-status.json").read_text())["completed"]


def test_layout_aggregation_is_document_macro(tmp_path):
    path = tmp_path / "layout.json"
    path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "doc_id": "patent",
                        "skipped": None,
                        "pages": [
                            {"page": 0, "n_matched": 2, "tau": 1.0},
                            {"page": 1, "n_matched": 2, "tau": 0.0},
                        ],
                    },
                    {"doc_id": "short", "skipped": None,
                     "pages": [{"page": 0, "n_matched": 2, "tau": 1.0}]},
                ]
            }
        )
    )
    values = A.layout_values(path)["tau"]
    assert values == {"patent": 0.5, "short": 1.0}
    assert A.mean(values)["mean"] == 0.75


def test_pixel_preservation_and_harm_are_detector_free(tmp_path):
    import numpy as np
    from PIL import Image

    src = np.full((100, 100), 255, dtype=np.uint8)
    src[50:80, 50:80] = 80
    dst = src.copy()
    # Thay glyph trong GT text là hợp lệ và phải bị loại khỏi hai PPR.
    dst[12:18, 12:35] = 0
    src_path, dst_path = tmp_path / "src.png", tmp_path / "dst.png"
    Image.fromarray(src).save(src_path)
    Image.fromarray(dst).save(dst_path)
    elements = [
        {"class": "Text", "bbox_norm": [0.1, 0.1, 0.4, 0.2]},
        {"class": "Picture", "bbox_norm": [0.5, 0.5, 0.8, 0.8]},
    ]
    args = SimpleNamespace(
        dilate_px=2, pixel_tolerance=8,
        harm_overlap=0.05, harm_change=0.05,
    )
    clean = V.score_page(src_path, dst_path, elements, [], args)
    assert clean["nt_ppr"] == 1.0 and clean["io_ppr"] == 1.0

    changed = np.zeros((100, 100), dtype=bool)
    changed[10:20, 40:50] = True
    overflow_elements = [
        {"class": "Text", "bbox_norm": [0.1, 0.1, 0.4, 0.2]},
        {"class": "Text", "bbox_norm": [0.4, 0.1, 0.7, 0.2]},
    ]
    harmful, assigned = V.harmful_overflow(
        [[0.1, 0.1, 0.5, 0.2]], overflow_elements, changed, 0.05, 0.05
    )
    assert (harmful, assigned) == (1, 1)


def test_pdf_text_bbox_excludes_invisible_spans(tmp_path):
    pdf_path = tmp_path / "visible.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page(width=200, height=200)
        page.insert_text((20, 30), "visible")
        page.insert_text((20, 160), "hidden", render_mode=3)
        pdf.save(pdf_path)
    lines = V.extract_text_lines(pdf_path)[0]
    assert len(lines) == 1
    assert lines[0][1] < 0.5


def test_scope_path_traversal_rejected():
    with pytest.raises(SystemExit):
        sync.resolve_scope(["out/../../.env"])
    with pytest.raises(ValueError):
        D.run_root("../bad")
    assert not publishable(Path("raw/output.pdf"))


def test_mock_hf_submission_is_sequential_and_stops_on_failure(prepared, monkeypatch):
    _, c = prepared
    for key, value in {
        "HF_JOB_TOKEN": "TEST_RUNTIME",
        "HF_EVAL_REPO": "u/r",
        "HF_BENCH_SPACE": "u/s",
        "LITELLM_KEY_BABELDOC": "TEST_BABEL",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("HF_CACHE_BUCKET", raising=False)
    api = Mock()
    api.run_job.return_value = SimpleNamespace(id="job-one")
    api.inspect_job.return_value = SimpleNamespace(
        status=SimpleNamespace(stage="ERROR")
    )
    with pytest.raises(RuntimeError, match="ERROR"):
        D.submit(api, c, "translate", "babeldoc")
    api.run_job.assert_called_once()
    assert (D.run_root(c["run_id"]) / "jobs/job-one.json").exists()
    assert (
        "TEST_RUNTIME"
        not in (D.run_root(c["run_id"]) / "jobs/job-one.json").read_text()
    )


def test_snapshot_acceptance_rejects_stale_report(prepared):
    root, c = prepared
    path = root / "out/_run/score-status.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"completed": False, "run_signature": P.signature(c)}))
    with pytest.raises(ValueError, match="incomplete"):
        P.verify_score(root / "out", c)


def test_synthetic_fixture_valid_but_not_eval(tmp_path):
    from benchmark.e2e.datasets.make_smoke import create

    source = tmp_path / "synthetic"
    create(source)
    assert len(P.validate_corpus(source, ["T1"])) == 3


def test_spend_import_deduplicates_and_requires_coverage(prepared, tmp_path):
    from benchmark.e2e.costs import import_spend

    root, c = prepared
    path = tmp_path / "spend.csv"
    header = "run_id,system,request_id,tokens_in,tokens_out,usd\n"
    rows = "".join(
        f"smoke-test,{s},request-{i},10,20,0.001\n" for i, s in enumerate(P.SYSTEMS[:3])
    )
    path.write_text(header + rows)
    import_spend(path, root / "out", c)
    summary = json.loads((root / "out/_costs/summary.json").read_text())
    assert summary["babeldoc"]["tokens_out"] == 20
    path.write_text(header + rows + rows)
    with pytest.raises(ValueError, match="duplicate"):
        import_spend(path, root / "out", c)
