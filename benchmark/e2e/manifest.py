"""Truy vết lượt chạy — vì **4 pipeline không bao giờ chạy cùng lúc**.

Đây là vấn đề riêng của benchmark này, không phải chuyện vệ sinh code chung. Bốn hệ
có dep xung đột nên phải chạy tuần tự, ở các venv khác nhau, và DeepL còn bị hạn mức
1M ký tự/tháng nên lượt EN→ZH của nó rơi sang **tháng sau**. Nghĩa là bốn cột trong
bảng so sánh được sinh ra cách nhau hàng tuần.

Trong khoảng đó, ba thứ có thể trôi mà **không để lại dấu vết nào trong PDF đầu ra**:

1. **Corpus bị dựng lại** với seed khác ⇒ hệ A chấm trên 120 trang này, hệ B chấm
   trên 120 trang khác. Bảng vẫn ra số đẹp và hoàn toàn vô nghĩa.
2. **Model đổi.** Runner baseline bắt buộc ``--model``, nhưng PDFTranslator để None
   thì rơi về default của provider. Chạy lại sau khi sửa default là lệch model.
3. **Alias key ở LiteLLM trùng nhau** ⇒ không tách được token/tiền của từng hệ.

Module này làm hai việc:

* ``write_manifest`` — chụp lại trạng thái đầu mỗi lượt vào ``out/_run/<run_id>.json``:
  git rev, model, corpus + sha256 từng file, phiên bản các thư viện đo.
* ``verify`` — đối chiếu ``meta.json`` của **mọi** hệ trước khi chấm điểm, và chặn
  nếu ba thứ trên đã trôi. Đây là cửa chặn thứ hai, song song với
  ``datasets/verify_corpus.py`` (cửa đó chặn corpus bẩn TRƯỚC khi tiêu tiền; cửa này
  chặn corpus/model đã trôi TRƯỚC khi tin vào bảng).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Hệ không dùng LLM ⇒ không có gì để đối chiếu về model.
NO_LLM = {"deepl-document"}

# Thư viện mà phiên bản của nó ảnh hưởng trực tiếp đến CON SỐ, không chỉ đến việc
# code chạy được hay không.
TRACKED_LIBS = ("pymupdf", "numpy", "scipy", "scikit-image", "fasttext",
                "docling", "unbabel-comet", "surya-ocr", "babeldoc")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _git(*args: str) -> str:
    try:
        return subprocess.run(("git", *args), capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001 — không phải git repo cũng không sao
        return ""


def _lib_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in TRACKED_LIBS:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "—"
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def corpus_fingerprint(corpus: Path, tiers: list[str]) -> dict:
    """sha256 từng PDF + một sha tổng. Đây là thứ phát hiện corpus bị dựng lại."""
    files = {}
    for tier in tiers:
        for pdf in sorted((corpus / tier).glob("*.pdf")) if (corpus / tier).is_dir() else []:
            files[pdf.stem] = {"tier": tier, "sha256": sha256(pdf),
                               "bytes": pdf.stat().st_size}
    combined = hashlib.sha256(
        "".join(f"{k}:{v['sha256']}" for k, v in sorted(files.items())).encode()
    ).hexdigest()
    return {"corpus_sha256": combined, "n_files": len(files), "files": files}


def write_manifest(out: Path, corpus: Path, tiers: list[str], langs: list[str],
                   systems: list[str], provider: str, model: str | None) -> Path:
    rid = run_id()
    manifest = {
        "run_id": rid,
        "ts": now_iso(),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_rev": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "pdf2zh")),
        "systems": systems, "tiers": tiers, "langs": langs,
        "provider": provider, "model": model,
        "key_alias_prefix": os.environ.get("KEY_ALIAS_PREFIX", ""),
        "accelerator": os.environ.get("ACCELERATOR", ""),
        "libs": _lib_versions(),
        "corpus_root": str(corpus),
        **corpus_fingerprint(corpus, tiers),
    }
    path = out / "_run" / f"{rid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_metas(out: Path, systems: list[str], langs: list[str]) -> list[dict]:
    metas = []
    for system in systems:
        for lang in langs:
            base = out / system / lang
            if not base.is_dir():
                continue
            for meta_path in sorted(base.glob("*/meta.json")):
                try:
                    m = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    continue
                m.setdefault("system", system)
                m.setdefault("lang", lang)
                m.setdefault("doc_id", meta_path.parent.name)
                metas.append(m)
    return metas


def verify(out: Path, systems: list[str], langs: list[str]) -> tuple[list[str], list[str]]:
    """Trả (errors, warnings). Errors = bảng so sánh KHÔNG dùng được."""
    errors: list[str] = []
    warns: list[str] = []
    metas = load_metas(out, systems, langs)
    if not metas:
        return ["không thấy meta.json nào — chưa chạy runner?"], []

    # 1. Cùng doc_id mà sha256 nguồn khác nhau ⇒ corpus đã bị dựng lại giữa 2 lượt.
    by_doc: dict[str, dict[str, str]] = {}
    for m in metas:
        if m.get("sha256"):
            by_doc.setdefault(m["doc_id"], {})[f"{m['system']}/{m['lang']}"] = m["sha256"]
    for doc, seen in sorted(by_doc.items()):
        if len(set(seen.values())) > 1:
            who = ", ".join(f"{k}={v[:8]}" for k, v in sorted(seen.items()))
            errors.append(f"corpus đã trôi: '{doc}' có sha256 khác nhau giữa các hệ ({who})")

    # 2. Model phải giống nhau ở mọi hệ dùng LLM.
    models: dict[str, set[str]] = {}
    for m in metas:
        if m["system"] in NO_LLM or not m.get("model"):
            continue
        models.setdefault(str(m["model"]), set()).add(f"{m['system']}/{m['lang']}")
    if len(models) > 1:
        detail = "; ".join(f"{k} <- {sorted(v)}" for k, v in sorted(models.items()))
        errors.append(f"model không đồng nhất giữa các hệ: {detail}")

    # 3. key_alias trùng ⇒ không tách được token/tiền của từng hệ ở LiteLLM.
    alias: dict[str, set[str]] = {}
    for m in metas:
        a = (m.get("key_alias") or "").strip()
        if a:
            alias.setdefault(a, set()).add(m["system"])
    for a, owners in sorted(alias.items()):
        if len(owners) > 1:
            warns.append(f"key_alias '{a}' dùng chung bởi {sorted(owners)} — "
                         f"không quy được token/USD riêng từng hệ")

    # 4. Phủ tài liệu lệch nhau giữa các hệ.
    cover: dict[str, set[str]] = {}
    for m in metas:
        cover.setdefault(f"{m['system']}/{m['lang']}", set()).add(m["doc_id"])
    if cover:
        full = set().union(*cover.values())
        for key, docs in sorted(cover.items()):
            missing = sorted(full - docs)
            if missing:
                warns.append(f"{key} thiếu {len(missing)} doc: {', '.join(missing[:4])}"
                             + (" …" if len(missing) > 4 else ""))

    # 5. Khoảng thời gian giữa lượt sớm nhất và muộn nhất.
    ts = sorted(m["ts"] for m in metas if m.get("ts"))
    if ts and ts[0][:10] != ts[-1][:10]:
        warns.append(f"artifact trải từ {ts[0][:10]} đến {ts[-1][:10]} — kiểm lại "
                     f"model/corpus không đổi trong khoảng đó (mục 1 và 2 ở trên)")
    elif not ts:
        warns.append("meta.json chưa có trường 'ts' — chạy lại runner để có mốc thời gian")

    return errors, warns


def print_report(errors: list[str], warns: list[str]) -> None:
    for w in warns:
        print(f"  [warn]  {w}", flush=True)
    for e in errors:
        print(f"  [ERROR] {e}", flush=True)
    if not errors and not warns:
        print("  OK — corpus, model và phủ tài liệu đồng nhất giữa các hệ.", flush=True)


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=("write", "verify", "list"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--tiers", default="T1")
    p.add_argument("--langs", default="vi")
    p.add_argument("--systems", default="pdftranslator,babeldoc,pdfmathtranslate,deepl-document")
    p.add_argument("--provider", default="litellm")
    p.add_argument("--model", default=None)
    a = p.parse_args()

    tiers = [x.strip() for x in a.tiers.split(",") if x.strip()]
    langs = [x.strip() for x in a.langs.split(",") if x.strip()]
    systems = [x.strip() for x in a.systems.split(",") if x.strip()]

    if a.action == "write":
        if a.corpus is None:
            print("!! 'write' cần --corpus")
            return 1
        path = write_manifest(a.out, a.corpus, tiers, langs, systems, a.provider, a.model)
        m = json.loads(path.read_text(encoding="utf-8"))
        print(f"  manifest: {path}")
        print(f"  corpus_sha256 {m['corpus_sha256'][:16]}…  ({m['n_files']} file)  "
              f"model={m['model'] or '(default provider)'}  git={m['git_rev'][:8] or '—'}")
        return 0

    if a.action == "list":
        runs = sorted((a.out / "_run").glob("*.json"))
        if not runs:
            print(f"  chưa có manifest nào dưới {a.out}/_run/")
            return 0
        for r in runs:
            m = json.loads(r.read_text(encoding="utf-8"))
            print(f"  {m['run_id']}  {m['ts']}  {','.join(m['systems']):40} "
                  f"model={m['model'] or '—'}  corpus={m['corpus_sha256'][:8]}")
        return 0

    errors, warns = verify(a.out, systems, langs)
    print_report(errors, warns)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
