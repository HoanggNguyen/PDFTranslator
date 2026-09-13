"""Container worker: checkpoint each document, sanitize logs, validate before scoring."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from benchmark.e2e import protocol as P, sync
from benchmark.e2e.manifest import now_iso, _lib_versions
from benchmark.e2e.runners._common import run_child
from benchmark.e2e.security import clean_env, redact, safe_json
from benchmark.e2e.source_bundle import ROOT, identity


def execute(command, log, env):
    rc, seconds, error = run_child(
        command, log, int(os.environ.get("BENCH_STEP_TIMEOUT", "7200")), env, ROOT
    )
    if error:
        print(f"  {log.name}: {error}", flush=True)
    return rc, seconds


def module(name, *args):
    return [sys.executable, "-m", "benchmark.e2e." + name, *map(str, args)]


def qe_module(name, *args):
    """QE steps run in their own venv: unbabel-comet needs numpy<2 while the
    main env must keep numpy>=2 for babeldoc. Locally QE_PYTHON is unset and
    this falls back to the harness interpreter, where both coexist."""
    python = os.environ.get("QE_PYTHON") or sys.executable
    return [python, "-m", "benchmark.e2e." + name, *map(str, args)]


def translate(
    corpus,
    out,
    contract,
    system,
    upload,
    dry=False,
    char_budget=950000,
    executor=execute,
):
    """Per-document subprocesses let completed work survive later crashes/timeouts."""
    sig = P.signature(contract)
    env = clean_env(
        keep=("DEEPL_AUTH_KEY",) if system == "deepl-document" else ("LITELLM_API_KEY",)
    )
    # Prevent a subprocess importing benchmark.e2e from reloading all local keys.
    env["BENCH_REMOTE"] = "1"
    if system != "deepl-document":
        env.update(
            LITELLM_BASE_URL=contract["base_url"],
            BENCH_MODEL=contract["model"],
            LITELLM_KEY_ALIAS=f"{contract['run_id']}-{system}",
        )
    failures = 0
    for doc, source in P.documents(corpus, contract["tiers"]).items():
        for lang in contract["langs"]:
            dest = out / system / lang / doc
            path = dest / "meta.json"
            if path.exists() and not dry:
                previous = json.loads(path.read_text())
                if previous.get("run_signature") != sig or previous.get(
                    "sha256"
                ) != P.sha256(source):
                    raise ValueError(
                        f"Resume refused: {system}/{lang}/{doc} run/corpus drift"
                    )
                if not previous.get("error"):
                    pdf = dest / "output.pdf"
                    if not pdf.is_file() or previous.get("output_sha256") != P.sha256(
                        pdf
                    ):
                        raise ValueError("Resume refused: output changed or missing")
                    print(f"  resume: {system}/{lang}/{doc}", flush=True)
                    continue
            dest.mkdir(parents=True, exist_ok=True)
            if not dry and (dest / "output.pdf").exists():
                (
                    dest / "output.pdf"
                ).unlink()  # Only stale output from a recorded failed attempt.
            name = "deepl_doc" if system == "deepl-document" else system
            meta = {
                "system": system,
                "lang": lang,
                "doc_id": doc,
                "tier": source.parent.name,
                "src": source.name,
                "sha256": P.sha256(source),
                "ts": now_iso(),
                "model": None if system == "deepl-document" else contract["model"],
                "run_signature": sig,
                "key_alias": env.get("LITELLM_KEY_ALIAS", ""),
                "error": None,
            }
            rc = 1
            try:
                with tempfile.TemporaryDirectory(prefix="bench-document-") as tmp:
                    shard = Path(tmp) / source.parent.name
                    shard.mkdir()
                    shutil.copyfile(source, shard / source.name)
                    cmd = module(
                        "runners." + name,
                        "--corpus",
                        tmp,
                        "--out",
                        out,
                        "--tiers",
                        source.parent.name,
                        "--langs",
                        lang,
                        "--no-resume",
                    )
                    if system == "deepl-document":
                        cmd += ["--char-budget", str(char_budget)]
                        if dry:
                            cmd.append("--dry-run")
                    else:
                        cmd += ["--model", contract["model"]]
                    rc, elapsed = executor(
                        cmd, dest / ("forecast.log" if dry else "worker.log"), env
                    )
                    if dry:
                        print((dest / "forecast.log").read_text(), flush=True)
                        if rc:
                            return rc
                        continue
                    if path.exists():
                        meta.update(json.loads(path.read_text()))
                    meta.update(
                        run_signature=sig,
                        key_alias=env.get("LITELLM_KEY_ALIAS", ""),
                        process_wall_seconds=round(elapsed, 2),
                        timing_policy=contract["timing_policy"],
                    )
                    pdf = dest / "output.pdf"
                    if rc or not pdf.exists():
                        meta["error"] = (
                            meta.get("error") or f"runner exit {rc}; see worker.log"
                        )
                    else:
                        import fitz

                        with fitz.open(pdf) as document:
                            if not len(document):
                                raise ValueError("Empty output PDF")
                        meta["output_sha256"] = P.sha256(pdf)
            except Exception as exc:
                meta["error"] = redact(f"{type(exc).__name__}: {exc}")
            finally:
                if not dry:
                    path.write_text(safe_json(meta))
                    if upload():
                        raise RuntimeError(
                            "Checkpoint upload failed; stop before spending more tokens"
                        )
            if meta["error"]:
                failures += 1
            print(
                f"  {system}/{lang}/{doc}: {'FAIL' if meta['error'] else 'OK'}",
                flush=True,
            )
            if rc == 2 and system == "deepl-document":
                return 2
    return 1 if failures else 0


def score_commands(corpus, out, c):
    common = [
        "--corpus",
        str(corpus),
        "--out",
        str(out),
        "--tiers",
        ",".join(c["tiers"]),
        "--langs",
        ",".join(c["langs"]),
    ]
    systems = ",".join(c["systems"] + ["identity"])
    yield module("runners.identity", *common)
    yield module("parse.render_pages", *common, "--systems", systems, "--force")
    yield module(
        "parse.run_detectors",
        "--out",
        out,
        "--langs",
        ",".join(c["langs"]),
        "--systems",
        systems,
        "--detectors",
        c["detector"],
        "--force",
    )
    yield module(
        "metrics.eval_preserve",
        *common,
        "--systems",
        systems,
        "--detector",
        c["detector"],
    )
    yield module("metrics.eval_visual", *common, "--systems", systems)
    yield module("metrics.eval_ink", *common)
    yield module("metrics.eval_text", *common, "--systems", systems)
    yield module("align.extract_pairs", *common, "--systems", systems)
    yield qe_module(
        "metrics.eval_qe",
        "--out",
        out,
        "--langs",
        ",".join(c["langs"]),
        "--systems",
        systems,
        "--model",
        c["qe_model"],
    )
    yield module(
        "metrics.aggregate",
        "--out",
        out,
        "--langs",
        ",".join(c["langs"]),
        "--systems",
        systems,
        "--detector",
        c["detector"],
    )


def scoring(corpus, out, c, upload, allow_failures=False, executor=execute):
    coverage = P.gate(corpus, out, c, allow_failures=allow_failures)
    # No stale derived metrics/report survive a failed scoring rerun.
    for name in ("_render", "_layout", "_metrics", "_pairs", "report"):
        path = out / name
        if path.exists():
            shutil.rmtree(path)
    status = out / "_run" / "score-status.json"
    status.parent.mkdir(parents=True, exist_ok=True)
    record = {"run_signature": P.signature(c), "coverage": coverage, "completed": False}
    env = clean_env(keep=("HF_TOKEN",))
    env["BENCH_REMOTE"] = "1"
    try:
        for i, cmd in enumerate(score_commands(corpus, out, c)):
            record["step"] = cmd[2]
            print(f"Scoring: {cmd[2]}", flush=True)
            rc, _ = executor(cmd, out / "_run" / f"score-{i:02d}.log", env)
            if rc:
                raise RuntimeError(f"Scoring step failed: {cmd[2]}")
        from benchmark.e2e.metrics.langid import LangID

        if LangID().backend != "fasttext":
            raise ValueError("Heuristic LID cannot pass acceptance")
        # Do not let a silently skipped QE system count as completed.
        for system in c["systems"]:
            for lang in c["langs"]:
                qe = out / "_metrics/qe" / f"{system}.{lang}.json"
                if not qe.exists():
                    if not allow_failures:
                        raise ValueError(f"Missing QE: {system}/{lang}")
                elif json.loads(qe.read_text()).get("n_pairs", 0) < 1:
                    raise ValueError(f"Empty QE: {system}/{lang}")
        ceiling_path = out / "_metrics/layout" / f"source_ceiling.{c['detector']}.json"
        ceiling = json.loads(ceiling_path.read_text())
        for lang in c["langs"]:
            baseline = json.loads(
                (
                    out / "_metrics/layout" / f"identity.{lang}.{c['detector']}.json"
                ).read_text()
            )
            if [r["pages"] for r in ceiling["records"]] != [
                r["pages"] for r in baseline["records"]
            ]:
                raise ValueError("Identity layout differs from source ceiling")
            visual_identity = json.loads(
                (out / "_metrics/visual" / f"identity.{lang}.json").read_text()
            )
            identity_pages = [
                page
                for rec in visual_identity["records"] if not rec.get("skipped")
                for page in rec["pages"] if "error" not in page
            ]
            if not identity_pages or any(
                page.get("nt_ppr") != 1.0
                or (page.get("io_ppr") is not None and page["io_ppr"] != 1.0)
                or page.get("of_harm") not in (None, 0.0)
                for page in identity_pages
            ):
                raise ValueError("Identity pixel/harm metrics are not ideal")
            ink_identity = json.loads(
                (out / "_metrics/ink" / f"identity.{lang}.json").read_text()
            )
            ink_pages = [
                page
                for rec in ink_identity["records"] if not rec.get("skipped")
                for page in rec["pages"]
            ]
            if not ink_pages or any(page.get("ic_harm") != 0.0 for page in ink_pages):
                raise ValueError("Identity IC-harm is not zero")
        record["source_ceiling"] = ceiling["summary"]
        if ceiling["summary"].get("n_docs_scored", 0) != len(
            P.documents(corpus, c["tiers"])
        ):
            raise ValueError("Source ceiling has incomplete document coverage")
        record["files"] = {
            str(p.relative_to(out)): P.sha256(p)
            for group in ("report", "_metrics", "_pairs")
            for p in (out / group).rglob("*")
            if p.is_file()
        }
        record["completed"] = True
        return 0
    except Exception as exc:
        record["error"] = redact(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        status.write_text(safe_json(record))
        if upload():
            raise RuntimeError("Scoring checkpoint upload failed")


def preflight(action, out):
    env = clean_env(keep=("HF_TOKEN",))
    env["BENCH_REMOTE"] = "1"
    if action == "check":
        commands = [
            [
                sys.executable,
                "-c",
                "import pdf2zh, fitz, fasttext, docling_ibm_models; print('imports OK')",
            ],
            # Comet lives in its own venv (numpy<2 vs babeldoc's numpy>=2).
            [
                os.environ.get("QE_PYTHON") or sys.executable,
                "-c",
                "import comet; print('comet OK')",
            ],
            [os.environ["BABELDOC_BIN"], "--version"],
            [os.environ["PDFMATHTRANSLATE_BIN"], "--version"],
        ]
    else:
        commands = [
            module("runners.pdftranslator", "--warmup-only"),
            module("runners.babeldoc", "--warmup-only"),
            module("parse.run_detectors", "--warmup-only"),
            [
                sys.executable,
                "-c",
                "from benchmark.e2e.metrics.langid import LangID; assert LangID().backend == 'fasttext'",
            ],
            [
                os.environ.get("QE_PYTHON") or sys.executable,
                "-c",
                "import os; from benchmark.e2e.metrics.eval_qe import load_model; load_model(os.environ['QE_MODEL'])",
            ],
        ]
    for i, command in enumerate(commands):
        log = out / "_run" / f"{action}-{i}.log"
        rc, _ = execute(command, log, env)
        print(f"{action} step {i}: {'OK' if not rc else 'FAIL'}", flush=True)
        if rc:
            raise RuntimeError(f"{action}: see {log.name}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("check", "warm", "probe", "translate", "score"))
    p.add_argument("--system", choices=P.SYSTEMS[:3])
    p.add_argument("--allow-failures", action="store_true")
    a = p.parse_args()
    rid = os.environ["BENCH_RUN_ID"]
    from benchmark.e2e.hf_driver import run_root, scopes

    base = run_root(rid)
    corpus, out = base / "corpus", base / "out"
    repo, token = os.environ["HF_EVAL_REPO"], os.environ["HF_TOKEN"]
    api, _ = sync.client(token)
    if not api.repo_info(repo, repo_type="dataset").private:
        raise ValueError("Runtime dataset must remain private")
    # Pull one immutable snapshot for corpus and all requested existing artifacts.
    revision = api.repo_info(repo, repo_type="dataset").sha
    only = [
        "corpus",
        "out" if a.action == "score" else f"out/{a.system}" if a.system else "out/_run",
    ]
    sync.do_pull(repo, ROOT, scopes(base, rid, only), token, revision, False)
    c = P.load_contract(corpus)
    if P.signature(c) != os.environ["BENCH_SIGNATURE"]:
        raise ValueError("Remote corpus contract does not match submitted run")
    if identity()["sha256"] != c["source_sha256"]:
        raise ValueError(
            "Image source differs from run; rebuild Space and prepare a new run"
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "_run").mkdir(exist_ok=True)
    (out / "_run" / f"input-{a.system or a.action}.json").write_text(
        json.dumps(
            {
                "dataset_revision": revision,
                "run_signature": P.signature(c),
                "source": identity(),
                "libs": _lib_versions(),
                "ts": now_iso(),
                "job_id": os.environ.get("JOB_ID"),
                "accelerator": os.environ.get("ACCELERATOR"),
            },
            indent=2,
        )
    )

    def upload():
        selected = (
            ["out"]
            if a.action == "score"
            else ([f"out/{a.system}"] if a.system else []) + ["out/_run"]
        )
        return sync.do_push(api, repo, ROOT, scopes(base, rid, selected), None, False)

    if a.action == "translate":
        return translate(corpus, out, c, a.system, upload)
    if a.action == "score":
        return scoring(corpus, out, c, upload, a.allow_failures)
    try:
        if a.action == "probe":
            from openai import OpenAI

            client = OpenAI(
                base_url=c["base_url"],
                api_key=os.environ["LITELLM_API_KEY"],
                timeout=60,
                max_retries=0,
            )
            reply = client.chat.completions.create(
                model=c["model"],
                messages=[
                    {"role": "user", "content": "Translate to Vietnamese: Hello."}
                ],
                temperature=0,
                max_tokens=32,
            )
            if not reply.choices or not reply.choices[0].message.content:
                raise ValueError("Probe returned no text")
            print(f"Proxy probe OK for {a.system}; response omitted")
        else:
            preflight(a.action, out)
        return 0
    finally:
        if upload():
            raise RuntimeError("Preflight upload failed")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(redact(f"{type(exc).__name__}: {exc}"), file=sys.stderr)
        raise SystemExit(1)
