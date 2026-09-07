"""Local HF benchmark controller. See docs/HF_GUIDE.md; secrets never enter shell commands."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

from benchmark.e2e import protocol as P, sync
from benchmark.e2e.security import redact
from benchmark.e2e.source_bundle import ROOT, identity, stage


def need(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Missing {name}; see benchmark/e2e/.env.bench.example")
    return value


def run_root(run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Run ID must contain only letters, numbers, '_' or '-'")
    return ROOT / "benchmark/e2e/work" / run_id


def prepare(args):
    """New immutable local snapshot; smoke clips first N pages of one PDF."""
    import fitz
    from urllib.parse import urlsplit

    model, base_url = need("BENCH_MODEL"), need("LITELLM_BASE_URL")
    url = urlsplit(base_url)
    if (
        url.scheme != "https"
        or not url.netloc
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("Proxy URL must be HTTPS without credentials/query/fragment")
    dest = run_root(args.run_id) / "corpus"
    if dest.exists():
        raise ValueError("Run already prepared; choose a new --run-id")
    tiers = ["T1"] if args.action == "prepare-smoke" else P.csv(args.tiers)
    systems, langs = P.csv(args.systems), P.csv(args.langs)
    if (
        not systems
        or set(systems) - set(P.SYSTEMS)
        or not langs
        or set(langs) - {"vi", "zh"}
    ):
        raise ValueError("Unknown/empty systems or languages")
    if args.action == "prepare-smoke" and (
        systems != list(P.SYSTEMS[:3]) or langs != ["vi"]
    ):
        raise ValueError("Smoke uses the three OSS pipelines and vi")
    files = P.validate_corpus(args.source, tiers)
    if args.action == "prepare-eval":
        for tier in tiers:
            if "SYNTHETIC" in json.loads(
                (args.source / tier / "gt.json").read_text()
            ).get("dataset", ""):
                raise ValueError(
                    "Synthetic corpus is for smoke only, not research eval"
                )
    selected = None
    if args.action == "prepare-smoke":
        docs = P.documents(args.source, tiers)
        selected = args.doc_id or min(docs, key=lambda k: docs[k].stat().st_size)
        if selected not in docs or args.pages < 1:
            raise ValueError("Unknown --doc-id or invalid --pages")
    for tier in tiers:
        target = dest / tier
        target.mkdir(parents=True)
        for name in ("gt.json", "mapping.json"):
            data = json.loads((args.source / tier / name).read_text())
            if selected:
                data["docs"] = [d for d in data["docs"] if d["doc_id"] == selected]
                for doc in data["docs"]:
                    doc["pages"] = doc["pages"][: args.pages]
            (target / name).write_text(json.dumps(data, indent=2))
        mapping_path = target / "mapping.json"
        mapping = json.loads(mapping_path.read_text())
        for doc in mapping["docs"]:
            src, dst = args.source / tier / doc["pdf"], target / doc["pdf"]
            if selected:
                with fitz.open(src) as pdf, fitz.open() as subset:
                    subset.insert_pdf(
                        pdf, from_page=0, to_page=min(args.pages, len(pdf)) - 1
                    )
                    subset.save(dst)
                doc["n_pages"] = len(doc["pages"])
                doc["n_chars"] = sum(p.get("n_chars", 0) for p in doc["pages"])
                doc["n_source_docs"] = len(
                    {p.get("original_filename") for p in doc["pages"]}
                )
            else:
                shutil.copyfile(src, dst)
            doc["sha256"] = P.sha256(dst)
        mapping_path.write_text(json.dumps(mapping, indent=2))
    contract = {
        "schema": 1,
        "run_id": args.run_id,
        "mode": "smoke" if selected else "eval",
        "tiers": tiers,
        "langs": langs,
        "systems": systems,
        "model": model,
        "base_url": base_url,
        "source_sha256": identity()["sha256"],
        "flavor": args.flavor,
        "image": os.environ.get("HF_BENCH_IMAGE", "").strip(),
        "qe_model": args.qe_model,
        "detector": "docling",
        "bootstrap_unit": args.unit,
        "temperature_policy": "proxy override temperature=0 (operator verified)",
        "timing_policy": "runner wall time; warmup/cache boundaries differ; diagnostic only",
        "files": P.validate_corpus(dest, tiers),
        "origin_files": files,
    }
    (dest / "run.json").write_text(json.dumps(contract, indent=2))
    print(
        f"Prepared {contract['mode']}: {dest}; signature {P.signature(contract)[:12]}"
    )


def scopes(base, run_id, only):
    return sync.resolve_scope(
        only, {"corpus": str(base / "corpus"), "out": str(base / "out")}, run_id
    )


def submission(contract, action, system=None, allow_failures=False):
    """Pure request builder, mock-tested; secret values never enter env/argv."""
    secrets = {"HF_TOKEN": need("HF_JOB_TOKEN")}
    env = {
        "BENCH_REMOTE": "1",
        "HF_EVAL_REPO": need("HF_EVAL_REPO"),
        "BENCH_RUN_ID": contract["run_id"],
        "BENCH_SIGNATURE": P.signature(contract),
        "BENCH_SOURCE_SHA256": contract["source_sha256"],
        "QE_MODEL": contract["qe_model"],
    }
    if system:
        secrets["LITELLM_API_KEY"] = need("LITELLM_KEY_" + system.upper())
        env.update(
            LITELLM_BASE_URL=contract["base_url"],
            BENCH_MODEL=contract["model"],
            LITELLM_KEY_ALIAS=f"{contract['run_id']}-{system}",
        )
    command = ["python3", "-m", "benchmark.e2e.hf_worker", action]
    if system:
        command += ["--system", system]
    if allow_failures:
        command += ["--allow-failures"]
    return dict(
        image=contract.get("image") or f"hf.co/spaces/{need('HF_BENCH_SPACE')}",
        command=command,
        env=env,
        secrets=secrets,
        flavor="cpu-basic" if action == "check" else contract["flavor"],
        timeout=os.environ.get("TIMEOUT", "3h"),
        name=f"{contract['run_id']}-{system or action}",
    )


def submit(api, contract, action, system=None, allow_failures=False):
    from huggingface_hub import Volume
    import time

    request = submission(contract, action, system, allow_failures)
    bucket = os.environ.get("HF_CACHE_BUCKET", "").strip()
    if bucket:
        request["volumes"] = [Volume(type="bucket", source=bucket, mount_path="/data")]
        request["env"]["BENCH_CACHE_MOUNTED"] = "1"
    job = api.run_job(**request)
    state_dir = run_root(contract["run_id"]) / "jobs"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / f"{job.id}.json").write_text(
        json.dumps(
            {
                "id": job.id,
                "action": action,
                "system": system,
                "image": request["image"],
                "run_signature": P.signature(contract),
            },
            indent=2,
        )
    )
    print(f"Job {job.id}: {action} {system or ''}; hf jobs logs {job.id}", flush=True)
    # Ctrl-C leaves the job running; its ID is persisted before waiting.
    while True:
        info = api.inspect_job(job_id=job.id)
        status = info.status.stage
        print(f"  {job.id}: {status}", flush=True)
        if status == "COMPLETED":
            return
        if status in {"ERROR", "CANCELED", "DELETED"}:
            raise RuntimeError(
                f"Job {job.id} {status}; see sanitized job/artifact logs"
            )
        time.sleep(20)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action",
        choices=(
            "prepare-smoke",
            "prepare-eval",
            "stage-space",
            "push-corpus",
            "check",
            "warm",
            "probe",
            "run",
            "score",
            "pull",
            "verify",
            "deepl-dry-run",
            "deepl",
            "import-costs",
        ),
    )
    p.add_argument("--run-id", default=os.environ.get("BENCH_RUN_ID", "smoke-001"))
    p.add_argument(
        "--source", type=Path, default=ROOT / "benchmark/e2e/datasets/corpus"
    )
    p.add_argument("--doc-id")
    p.add_argument("--pages", type=int, default=2)
    p.add_argument("--tiers", default="T1")
    p.add_argument("--langs", default="vi")
    p.add_argument("--systems", default=",".join(P.SYSTEMS[:3]))
    p.add_argument("--system", choices=P.SYSTEMS[:3])
    p.add_argument("--flavor", default=os.environ.get("FLAVOR", "t4-medium"))
    p.add_argument("--qe-model", default="Unbabel/wmt22-cometkiwi-da")
    p.add_argument("--unit", choices=("doc", "page"), default="doc")
    p.add_argument("--dest", type=Path, default=ROOT / "benchmark/e2e/space-build")
    p.add_argument("--allow-failures", action="store_true")
    p.add_argument(
        "--require-score",
        action="store_true",
        help="Also require a completed scoring snapshot",
    )
    p.add_argument("--revision", help="Exact dataset commit for pull")
    p.add_argument("--char-budget", type=int, default=950000)
    p.add_argument("--spend-csv", type=Path)
    return p


def main():
    a = parser().parse_args()
    if a.action.startswith("prepare-"):
        prepare(a)
        return 0
    if a.action == "stage-space":
        stage(a.dest)
        return 0
    base = run_root(a.run_id)
    if a.action == "verify":
        c = P.load_contract(base / "corpus")
        print(P.gate(base / "corpus", base / "out", c, allow_failures=a.allow_failures))
        if a.require_score:
            P.verify_score(base / "out", c)
            print("Scoring snapshot verified")
        return 0
    from huggingface_hub import HfApi, hf_hub_download

    api, token, repo = (
        HfApi(token=need("HF_TOKEN")),
        need("HF_TOKEN"),
        need("HF_EVAL_REPO"),
    )
    if a.action == "pull":
        revision = a.revision or api.repo_info(repo, repo_type="dataset").sha
        rc = sync.do_pull(
            repo,
            ROOT,
            scopes(base, a.run_id, ["corpus", "out"]),
            token,
            revision,
            False,
        )
        base.mkdir(parents=True, exist_ok=True)
        (base / "download-revision.txt").write_text(revision + "\n")
        print(f"Dataset revision: {revision}; report: {base / 'out/report/report.md'}")
        return rc
    c = P.load_contract(base / "corpus")
    if a.action == "import-costs":
        if not a.spend_csv:
            raise ValueError("import-costs requires --spend-csv")
        from benchmark.e2e.costs import import_spend

        import_spend(a.spend_csv, base / "out", c)
        return sync.do_push(
            api, repo, ROOT, scopes(base, a.run_id, ["out/_costs"]), None, False
        )
    if a.action == "push-corpus":
        if sync.do_init(api, repo, True):
            return 1
        if not api.repo_info(repo, repo_type="dataset").private:
            raise ValueError("Benchmark dataset must be private")
        from huggingface_hub.errors import EntryNotFoundError

        try:
            old = hf_hub_download(
                repo, f"{a.run_id}/corpus/run.json", repo_type="dataset", token=token
            )
        except EntryNotFoundError:
            old = None
        if old and json.loads(Path(old).read_text()) != c:
            raise ValueError("Remote contract differs; choose a new run ID")
        return sync.do_push(
            api, repo, ROOT, scopes(base, a.run_id, ["corpus"]), None, False
        )
    if a.action in {"deepl", "deepl-dry-run"}:
        if "deepl-document" not in c["systems"]:
            raise ValueError("Prepare an eval contract containing deepl-document first")
        from benchmark.e2e.hf_worker import translate

        if a.action == "deepl":
            sync.do_pull(
                repo,
                ROOT,
                scopes(base, a.run_id, ["out/deepl-document"]),
                token,
                None,
                False,
            )
        return translate(
            base / "corpus",
            base / "out",
            c,
            "deepl-document",
            dry=a.action.endswith("dry-run"),
            char_budget=a.char_budget,
            upload=lambda: sync.do_push(
                api,
                repo,
                ROOT,
                scopes(base, a.run_id, ["out/deepl-document", "out/_run"]),
                None,
                False,
            ),
        )
    if identity()["sha256"] != c["source_sha256"]:
        raise ValueError("Source changed; prepare a new run and rebuild Space")
    if a.action in {"run", "probe"}:
        systems = (
            [a.system] if a.system else [s for s in c["systems"] if s in P.SYSTEMS[:3]]
        )
        if set(systems) - set(c["systems"]):
            raise ValueError("System not in run contract")
        # Single shared key allowed by operator decision: cost attribution per
        # pipeline is lost; spend logs mix all systems on one key.
        for system in systems:
            submit(api, c, "translate" if a.action == "run" else "probe", system)
    else:
        submit(api, c, a.action, allow_failures=a.allow_failures)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(redact(f"{type(exc).__name__}: {exc}"), file=sys.stderr)
        raise SystemExit(1)
