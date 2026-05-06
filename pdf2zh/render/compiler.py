from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def compile_typst(
    source: str,
    font_paths: list[str],
    output_pdf: Path,
    typst_bin: str = "typst",
    work_dir: Path | None = None,
) -> Path:
    """Compile a Typst source string to PDF.

    Args:
        source: Complete Typst source code.
        font_paths: Directories or file paths passed to --font-path.
        output_pdf: Destination PDF path.
        typst_bin: Path/name of the typst binary.
        work_dir: Directory to write the intermediate .typ file (defaults to
                  output_pdf.parent).

    Returns:
        output_pdf path.

    Raises:
        RuntimeError: If the typst process exits non-zero.
    """
    work_dir = work_dir or output_pdf.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    typ_path = work_dir / (output_pdf.stem + ".typ")
    typ_path.write_text(source, encoding="utf-8")

    cmd = [typst_bin, "compile"]
    for fp in font_paths:
        cmd += ["--font-path", fp]
    cmd += [str(typ_path), str(output_pdf)]

    logger.debug("typst: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        tail = "\n".join((result.stderr or "").splitlines()[-50:])
        raise RuntimeError(
            f"typst compile failed (exit {result.returncode}):\n{tail}"
        )

    logger.debug("typst compiled → %s", output_pdf)
    return output_pdf
