"""Build a reviewable Space source tree from an explicit source allowlist."""

import json
import shutil
from pathlib import Path

from benchmark.e2e.manifest import sha256, _git
from benchmark.e2e.protocol import signature
from benchmark.e2e.security import scan_file

ROOT = Path(__file__).resolve().parents[2]


def source_files(root=ROOT):
    files = [
        root / p
        for p in (
            "requirements.txt",
            "pyproject.toml",
            "Dockerfile.bench",
            ".dockerignore",
            "LICENSE",
        )
    ]
    for directory in ("pdf2zh", "benchmark", "script"):
        for p in (root / directory).rglob("*"):
            rel = p.relative_to(root)
            if any(
                part in {"out", "work", "corpus", "__pycache__"}
                or part.startswith((".", "space-build"))
                for part in rel.parts
            ):
                continue
            if p.is_file() and p.suffix in {".py", ".sh", ".toml", ".ttf", ".otf"}:
                files.append(p)
    return sorted(files)


def identity(root=ROOT):
    records = {str(p.relative_to(root)): sha256(p) for p in source_files(root)}
    return {
        "sha256": signature(records),
        "files": records,
        "git_rev": _git("rev-parse", "HEAD"),
    }


def stage(dest: Path):
    dest = dest.resolve()
    if dest.is_relative_to(ROOT) and not (
        dest.parent == ROOT / "benchmark/e2e" and dest.name.startswith("space-build")
    ):
        raise ValueError(
            "Stage outside the repository or into benchmark/e2e/space-build*"
        )
    if dest.exists():
        raise ValueError("Stage destination exists; choose a new empty directory")
    for source in source_files():
        if source.is_symlink():
            raise ValueError("Symlink in source bundle")
        scan_file(source)
    dest.mkdir(parents=True)
    for source in source_files():
        target = dest / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(dest / "Dockerfile.bench", dest / "Dockerfile")
    (dest / "image-source.json").write_text(json.dumps(identity(), indent=2))
    (dest / "README.md").write_text(
        "---\ntitle: PDFTranslator benchmark\nsdk: docker\napp_port: 7860\n---\n"
        "Benchmark image for HF Jobs. No credentials belong in this repository.\n"
    )
    print(f"Review/upload source tree: {dest}")
