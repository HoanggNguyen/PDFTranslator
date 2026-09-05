"""Shared plumbing cho các baseline chạy ngoài process.

BabelDOC và PDFMathTranslate KHÔNG thể import chung interpreter với PDFTranslator:

* PDFMathTranslate cũng đặt tên package của nó là ``pdf2zh``. Import cả hai trong một
  interpreter thì cái nào vào ``sys.modules`` trước sẽ che cái kia — và lỗi này im
  lặng: benchmark vẫn chạy, chỉ là đo sai hệ thống. Vì vậy ``child_env()`` xoá sạch
  ``PYTHONPATH`` và runner đặt ``cwd`` ra thư mục tạm, không phải repo root.
* Ràng buộc dep vênh nhau: PDFMathTranslate cần ``pymupdf<1.25.3`` còn BabelDOC cần
  ``pymupdf>=1.26.7``. Không có một venv nào thoả cả hai.

Nên mọi baseline đều đi qua subprocess + filesystem: runner dựng câu lệnh, gọi
console script trong venv riêng, rồi đọc PDF nó đẻ ra. Trao đổi duy nhất là file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from benchmark.e2e.manifest import now_iso

# Biến môi trường không được rò từ process cha sang con: PYTHONPATH/PYTHONHOME làm
# lẫn package (xem docstring), VIRTUAL_ENV làm một số tool đoán sai interpreter.
_SCRUB = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONSTARTUP")


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _SCRUB}
    env["PYTHONUNBUFFERED"] = "1"
    if extra:
        env.update({k: v for k, v in extra.items() if v is not None})
    return env


def peak_rss_children_mb() -> float:
    """Peak RSS của toàn bộ process con đã kết thúc.

    ``ru_maxrss`` của RUSAGE_CHILDREN là max cộng dồn cho cả run, không phải số của
    riêng document vừa chạy — đọc nó như trần bộ nhớ mức run, y như
    ``peak_rss_mb()`` trong runner của PDFTranslator.
    """
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return round(rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024, 1)


def page_count(path: Path) -> int:
    import fitz  # PyMuPDF

    try:
        with fitz.open(path) as doc:
            return doc.page_count
    except Exception:  # noqa: BLE001 — file có thể không tồn tại hoặc hỏng
        return 0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def absolutize(*paths: Path | None) -> tuple[Path | None, ...]:
    """Đưa mọi đường dẫn về tuyệt đối TRƯỚC khi truyền cho baseline.

    Child chạy với ``cwd`` là thư mục tạm (xem docstring module), nên đường dẫn
    tương đối kiểu ``benchmark/e2e/datasets/corpus/...`` sẽ không phân giải được ở
    phía nó. Baseline thì báo "không tìm thấy file" theo cách riêng của mỗi cái —
    hoặc rc=0 mà chẳng đẻ ra gì — nên lỗi này rất dễ bị đọc thành "baseline thất
    bại trên document này".
    """
    return tuple(p.expanduser().resolve() if p is not None else None for p in paths)


def discover(corpus: Path, tiers: list[str], system: str) -> list[tuple[str, Path]]:
    """Giống hệt ``runners/pdftranslator.discover`` — doc_id LUÔN là ``pdf.stem``.

    Trùng khớp là điều kiện để metric ghép được artifact của các hệ thống với nhau;
    lệch một ký tự là bảng so sánh mất dòng mà không báo lỗi.
    """
    jobs: list[tuple[str, Path]] = []
    for tier in tiers:
        tier_dir = corpus / tier
        if not tier_dir.is_dir():
            print(f"[{system}] {tier}: chưa dựng ({tier_dir}) — bỏ qua", flush=True)
            continue
        for pdf in sorted(tier_dir.glob("*.pdf")):
            jobs.append((tier, pdf))
    return jobs


def resolve_bin(raw: str | None, script_name: str, module_expr: str,
                env_var: str, allow_path_fallback: bool = True) -> list[str] | None:
    """``--bin`` nhận cả console script và interpreter.

    Trỏ vào ``.../bin/babeldoc`` thì dùng thẳng; trỏ vào ``.../bin/python`` thì gọi
    ``python -c "<module_expr>"``. Chấp nhận cả hai vì venv do uv tạo có script, còn
    checkout cài ``-e`` đôi khi chỉ có interpreter.

    ``allow_path_fallback=False`` cho hệ mà cái tên trong PATH gần như chắc chắn là
    thứ khác: env của PDFTranslator có sẵn console script ``pdf2zh`` **của chính nó**
    (cùng tên package), nên rơi về PATH là lặng lẽ benchmark nhầm hệ thống.
    """
    if not raw:
        found = shutil.which(script_name) if allow_path_fallback else None
        if found:
            return [found]
        why = ("PATH không được dùng cho hệ này (trùng tên với chính PDFTranslator)"
               if not allow_path_fallback else f"không thấy '{script_name}' trong PATH")
        print(f"!! {why}. Đặt {env_var}=<đường dẫn tới {script_name} hoặc python "
              f"của venv baseline> hoặc dùng --bin.")
        return None

    path = Path(raw).expanduser()
    if not path.exists():
        print(f"!! --bin không tồn tại: {path}")
        return None
    if path.name.startswith("python"):
        return [str(path), "-c", module_expr]
    return [str(path)]


_VER_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def probe_version(base: list[str], env: dict[str, str],
                  cwd: Path) -> tuple[str, tuple[int, int, int] | None]:
    """Hỏi baseline nó là phiên bản nào, TRƯỚC khi chạy một document nào.

    Bắt buộc phải làm, và đây là lý do cụ thể: env của PDFTranslator cài sẵn
    ``babeldoc`` (nó là dependency, bản ``<0.3``) trong khi baseline là checkout
    ``0.6.x``. Hai bản này khác nhau về cả thuật toán typesetting lẫn cờ dòng lệnh.
    Chạy nhầm thì không có lỗi nào cả — chỉ là cột BabelDOC trong bảng thuộc về một
    phần mềm khác. Phiên bản đo được luôn được ghi vào ``meta.json``.
    """
    try:
        proc = subprocess.run(base + ["--version"], capture_output=True, text=True,
                              timeout=120, env=env, cwd=str(cwd))
    except Exception as exc:  # noqa: BLE001
        return f"(không hỏi được: {type(exc).__name__})", None
    raw = (proc.stdout + proc.stderr).strip().splitlines()
    text = raw[0] if raw else ""
    m = _VER_RE.search(text)
    return text, (tuple(int(g) for g in m.groups()) if m else None)  # type: ignore[return-value]


def run_child(cmd: list[str], log_path: Path, timeout_s: int,
              env: dict[str, str], cwd: Path) -> tuple[int, float, str | None]:
    """Chạy baseline, ghi log ra file, trả (returncode, giây, lỗi-nếu-có).

    Timeout giết cả process group: baseline nào cũng spawn worker (BabelDOC dùng
    thread pool + có thể process pool), ``proc.kill()`` trần để lại con mồ côi giữ
    GPU và làm document sau đo sai.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                env=env, cwd=str(cwd), start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout_s)
            err = None
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            rc, err = -9, f"timeout sau {timeout_s}s"
    return rc, round(time.perf_counter() - t0, 2), err


def pick_mono(raw_dir: Path) -> Path | None:
    """Chọn PDF *mono* (chỉ bản dịch) trong thư mục output của baseline.

    Phải là mono, không phải dual: metric hình học đối chiếu từng trang đích với
    trang nguồn tương ứng, còn dual xen trang gốc nên ``n_pages_out`` gấp đôi và
    mọi chỉ số layout thành vô nghĩa.
    """
    pdfs = [p for p in raw_dir.rglob("*.pdf")]
    if not pdfs:
        return None
    mono = [p for p in pdfs if "mono" in p.name.lower()]
    if mono:
        return max(mono, key=lambda p: p.stat().st_size)
    single = [p for p in pdfs if "dual" not in p.name.lower()]
    if single:
        return max(single, key=lambda p: p.stat().st_size)
    return None


def write_meta(meta_path: Path, meta: dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                         encoding="utf-8")


def base_meta(system: str, tier: str, lang: str, pdf: Path, model: str | None,
              base_url: str | None) -> dict:
    """Các khoá trùng schema của ``runners/pdftranslator.py`` để aggregate đọc chung.

    ``tokens_in``/``tokens_out`` để None khi baseline không tự đếm: không quan sát
    được từ ngoài process. Quy về LiteLLM proxy — mỗi hệ thống một virtual key
    (``key_alias``), sau đó truy ``/spend/logs`` để quy tiền và token. ``key_alias``
    ghi vào meta chính là mối nối đó.
    """
    return {
        "system": system, "tier": tier, "lang": lang,
        "ts": now_iso(),          # 4 hệ chạy cách nhau hàng tuần — xem manifest.py
        "doc_id": pdf.stem, "src": pdf.name, "sha256": sha256(pdf),
        "provider": "litellm", "model": model,
        "base_url": base_url,
        "key_alias": os.environ.get("LITELLM_KEY_ALIAS", ""),
        "wall_seconds": None, "n_pages_in": page_count(pdf), "n_pages_out": 0,
        "page_inflation": None, "peak_rss_mb": None,
        "accelerator": os.environ.get("ACCELERATOR", ""),
        "tokens_in": None, "tokens_out": None, "error": None,
    }


def finalize(meta: dict, out_pdf: Path) -> dict:
    meta["n_pages_out"] = page_count(out_pdf)
    meta["peak_rss_mb"] = peak_rss_children_mb()
    meta["page_inflation"] = (round(meta["n_pages_out"] / meta["n_pages_in"], 4)
                              if meta["n_pages_in"] else None)
    return meta
