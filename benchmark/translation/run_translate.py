"""Bước A — dịch WMT24++ qua ĐÚNG core (translate_document) + đo latency.

Chạy 1 doc để test, hoặc full pair/sweep. Ghi:
  <out>/hypotheses.jsonl  — 1 dòng/segment (từ repeat 0), để chấm COMET/judge
  <out>/latency.jsonl     — 1 dòng/doc/repeat, để thống kê latency

Hai chế độ:
  * latency-measure (mặc định, --doc-workers 1): tuần tự, concurrency=8 chuẩn
    production, N repeats → số s/doc trung thực.
  * quality-gen (--doc-workers N>1): chạy nhiều doc song song để lấy hypotheses
    nhanh; latency KHÔNG còn production-faithful (đánh dấu mode="quality-gen").

Ví dụ:
  # test 1 doc:
  python -m benchmark.translation.run_translate --pair vi_VN --provider gemini --limit-docs 1 --repeats 1
  # full pair, đo latency:
  python -m benchmark.translation.run_translate --pair vi_VN --provider gemini --repeats 5
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Allow `python benchmark/translation/run_translate.py` as well as `-m`.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv  # noqa: E402

from pdf2zh.translation.config import PROVIDERS, TranslatorConfig, resolve_provider  # noqa: E402
from pdf2zh.translation.pipeline import translate_document  # noqa: E402

from benchmark.translation.instrument import Instrument  # noqa: E402
from benchmark.translation.wmt24pp_adapter import (  # noqa: E402
    DocBundle,
    extract_hypotheses,
    load_pair,
)

logger = logging.getLogger("benchmark.translation.run_translate")


def build_config(args, api_key: str) -> TranslatorConfig:
    cfg = TranslatorConfig(
        source_language="English",
        provider=args.provider,
        model=args.model,
        api_key=api_key,
    )
    if args.no_post_fix:
        cfg.toc_fix_enabled = False
        cfg.math_fix_enabled = False
    if args.no_reasoning:
        cfg.disable_reasoning = True
    return cfg


def vertex_auth(cfg: TranslatorConfig, model: str | None):
    """Cấu hình cfg gọi thẳng Vertex AI (OpenAI-compat) bằng ADC + project/region.

    Trả về hàm refresh() lấy access-token mới (token GCP hết hạn ~1h; harness gọi
    refresh trước mỗi doc để token luôn tươi cho run dài). KHÔNG đụng core: chỉ set
    base_url/api_key/model trên cfg và đặt provider='litellm' (một key hợp lệ trong
    PROVIDERS) để resolve_provider(cfg) không raise khi base_url/model/key đã có.

    .env cần: VERTEX_PROJECT (và tùy chọn VERTEX_LOCATION, mặc định us-central1).
    Xác thực: `gcloud auth application-default login` (ADC) hoặc
    GOOGLE_APPLICATION_CREDENTIALS trỏ tới service-account JSON.
    """
    try:
        import google.auth
        import google.auth.transport.requests as gart
    except ImportError:
        raise SystemExit("Cần google-auth: pip install google-auth")
    project = os.environ.get("VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    region = os.environ.get("VERTEX_LOCATION", "us-central1")
    if not project:
        raise SystemExit("Đặt VERTEX_PROJECT (và tùy chọn VERTEX_LOCATION) trong .env")
    if not model:
        raise SystemExit("Vertex cần --model, ví dụ: google/gemini-2.5-flash")
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    req = gart.Request()

    def refresh() -> str:
        creds.refresh(req)  # chỉ gọi mạng khi token gần hết hạn (google-auth tự cache)
        return creds.token

    # location 'global' dùng host không có tiền tố region.
    host = "aiplatform.googleapis.com" if region == "global" else f"{region}-aiplatform.googleapis.com"
    cfg.base_url = (
        f"https://{host}/v1beta1/"
        f"projects/{project}/locations/{region}/endpoints/openapi"
    )
    cfg.provider = "litellm"   # để resolve_provider(cfg) chấp nhận (đã set base_url/model/key)
    cfg.model = model
    cfg.api_key = refresh()
    return refresh


def _fresh(bundle: DocBundle) -> DocBundle:
    """Copy so a repeat translates a clean, untranslated doc."""
    return DocBundle(
        pair=bundle.pair, document_id=bundle.document_id,
        segs=bundle.segs, doc=copy.deepcopy(bundle.doc),
    )


def translate_one(bundle: DocBundle, cfg: TranslatorConfig,
                  inst: Instrument | None) -> tuple[DocBundle, dict]:
    """Translate one document; return the (mutated) bundle + a latency record.

    Never raises: a translate_document failure (network glitch, malformed
    provider response, ...) must not lose the OTHER documents in the same
    batch. Failures are recorded (rec["error"]) and yield an empty translation
    for that doc so hypotheses/latency for the rest of the pair are preserved.
    """
    b = _fresh(bundle)
    src_chars = sum(len(s.source) for s in b.segs)
    src_words = sum(len(s.source.split()) for s in b.segs)
    mark = inst.mark() if inst else 0
    t0 = time.perf_counter()
    try:
        translate_document(b.doc, cfg)
        error = None
    except Exception as exc:  # noqa: BLE001 — isolate one bad doc from the rest
        logger.warning("doc %s failed: %s: %s", b.document_id, type(exc).__name__, exc)
        error = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - t0
    # Độ dài BẢN DỊCH (destination) — để đo tỉ lệ giãn & ràng buộc ±15% theo từng LLM.
    dst_texts = [e.get("translated_text", "") for e in b.doc["pages"][0]["elements"]]
    dst_chars = sum(len(t) for t in dst_texts)
    dst_words = sum(len(t.split()) for t in dst_texts)
    rec = {
        "document_id": b.document_id,
        "n_segments": len(b.segs),
        "src_words": src_words,   # để chuẩn hoá latency theo độ dài (từ-nguồn/s)
        "src_chars": src_chars,
        "dst_words": dst_words,
        "dst_chars": dst_chars,
        "wall_s": round(wall, 4),
        "error": error,
    }
    if inst:
        st = inst.since(mark)
        rec.update({
            "n_req": st.n_req, "n_retry": st.n_retry,
            "tok_in": st.tok_in, "tok_out": st.tok_out,
            "req_p50": round(st.p50, 4), "req_p95": round(st.p95, 4),
        })
    return b, rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", required=True, help="Target locale, e.g. vi_VN, de_DE, zh_CN")
    ap.add_argument("--provider", default="gemini", choices=list(PROVIDERS) + ["vertex"])
    ap.add_argument("--model", default=None, help="Override model id (else provider default)")
    ap.add_argument("--api-key", default=None, help="Else read provider env var")
    ap.add_argument("--limit-docs", type=int, default=None, help="First N documents only (test)")
    ap.add_argument("--repeats", type=int, default=5, help="Latency repeats per doc")
    ap.add_argument("--doc-workers", type=int, default=1, help=">1 = quality-gen (latency not faithful)")
    ap.add_argument("--no-post-fix", action="store_true", help="Disable toc_fix/math_fix (for the no-op check)")
    ap.add_argument("--no-reasoning", action="store_true",
                    help="OpenRouter only: send reasoning={enabled: false} to skip the "
                         "model's thinking pass (faster/cheaper on reasoning models it proxies, "
                         "e.g. Qwen, DeepSeek R1). No-op for other providers.")
    ap.add_argument("--out", default="benchmark/translation/out", help="Output dir")
    ap.add_argument("--resume", action="store_true",
                    help="Skip docs already succeeded (found in existing latency.jsonl "
                         "for this system+pair, error=null) — retries only missing/failed docs")
    ap.add_argument("--system-label", default=None,
                    help="Ghi đè nhãn 'system' dùng để log VÀ để so khớp --resume — dùng khi "
                         "gọi qua route khác (vd vertex) để BÙ dữ liệu cho cùng một model đã "
                         "chạy qua route cũ (vd litellm), coi là cùng một nguồn kết quả. "
                         "VD: --provider vertex --model google/gemini-3.1-flash-lite "
                         "--system-label litellm/gemini-3.1-flash-lite")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.verbose:
        # --verbose chỉ để xem log CỦA HARNESS (benchmark.translation.*, json_translator,
        # pdf2zh.*) ở mức
        # DEBUG — httpx/httpcore/google-auth/urllib3 quá ồn (log từng gói TCP/TLS) nên
        # giữ ở INFO, đủ để vẫn thấy "HTTP Request: POST ... 200/400/429" của mỗi call.
        for noisy in ("httpcore", "httpx", "urllib3", "google", "google.auth"):
            logging.getLogger(noisy).setLevel(logging.INFO)

    load_dotenv()  # nạp .env (repo root) trước khi đọc key/base_url
    refresh = None  # token-refresh callback (chỉ dùng cho vertex)
    if args.provider == "vertex":
        cfg = build_config(args, api_key="")   # api_key sẽ do vertex_auth cấp
        refresh = vertex_auth(cfg, args.model)
    else:
        api_key = args.api_key or os.environ.get(PROVIDERS[args.provider]["env_var"], "")
        if not api_key:
            raise SystemExit(f"No API key: pass --api-key or set {PROVIDERS[args.provider]['env_var']}")
        cfg = build_config(args, api_key)
    resolve_provider(cfg)  # fill model/base_url now so the system label is accurate
    system = args.system_label or f"{args.provider}/{cfg.model}"
    if args.system_label:
        logger.warning("system-label override: gọi %s/%s nhưng ghi log là %r",
                       args.provider, cfg.model, system)
    mode = "quality-gen" if args.doc_workers > 1 else "latency-measure"
    logger.info("System=%s pair=en-%s mode=%s", system, args.pair, mode)

    bundles = load_pair(args.pair)
    if args.limit_docs:
        bundles = bundles[: args.limit_docs]
    logger.info("Documents: %d", len(bundles))

    if args.resume:
        lat_path = Path(args.out) / "latency.jsonl"
        done_ids: set[str] = set()
        if lat_path.exists():
            with open(lat_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if (r.get("system") == system and r.get("pair") == f"en-{args.pair}"
                            and not r.get("error")):
                        done_ids.add(r["document_id"])
        before = len(bundles)
        skipped_in_scope = sum(1 for b in bundles if b.document_id in done_ids)
        bundles = [b for b in bundles if b.document_id not in done_ids]
        logger.info(
            "Resume: %d known-OK trong cả cặp en-%s | phạm vi hiện tại %d doc "
            "-> %d đã OK (skip), %d còn lại (missing/failed) sẽ dịch",
            len(done_ids), args.pair, before, skipped_in_scope, len(bundles),
        )
        if not bundles:
            logger.info("Không còn doc nào cần dịch cho en-%s — bỏ qua.", args.pair)
            return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hyp_f = open(out / "hypotheses.jsonl", "a", encoding="utf-8")
    lat_f = open(out / "latency.jsonl", "a", encoding="utf-8")

    n_hyp = n_err = 0

    def _write(b: DocBundle, rec: dict, rep: int, is_canonical: bool) -> None:
        """Ghi + flush NGAY cho một doc — không đợi cả lô 170 doc xong. Một doc
        lỗi (rec['error'] set) không mất dữ liệu của các doc khác đã hoàn tất."""
        nonlocal n_hyp, n_err
        rec.update({"system": system, "pair": f"en-{args.pair}",
                    "repeat": rep, "mode": mode})
        if rec.get("error"):
            n_err += 1
        lat_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        lat_f.flush()
        if is_canonical and not rec.get("error"):
            for h in extract_hypotheses(b):
                h["system"] = system
                hyp_f.write(json.dumps(h, ensure_ascii=False) + "\n")
                n_hyp += 1
            hyp_f.flush()

    with Instrument() as inst:
        for rep in range(args.repeats):
            is_canonical = rep == 0  # write hypotheses only from the first pass
            if args.doc_workers > 1:
                if refresh:
                    cfg.api_key = refresh()  # 1 lần/batch (token đủ dùng ~1h)
                with ThreadPoolExecutor(max_workers=args.doc_workers) as ex:
                    futures = [ex.submit(translate_one, b, cfg, None) for b in bundles]
                    for fut in as_completed(futures):
                        b, rec = fut.result()  # translate_one never raises (see docstring)
                        _write(b, rec, rep, is_canonical)
            else:
                for b in bundles:
                    if refresh:
                        cfg.api_key = refresh()  # trước mỗi doc -> token luôn tươi cho run dài
                    bb, rec = translate_one(b, cfg, inst)
                    _write(bb, rec, rep, is_canonical)

            logger.info("repeat %d/%d done (%d errors)", rep + 1, args.repeats, n_err)

    hyp_f.close(); lat_f.close()
    if n_err:
        logger.warning("%d/%d documents failed this run (see 'error' field in latency.jsonl)",
                       n_err, len(bundles) * args.repeats)

    # Quick console summary.
    walls, words = [], 0.0
    with open(out / "latency.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["system"] == system and r["pair"] == f"en-{args.pair}":
                walls.append(r["wall_s"])
                words += r.get("src_words", 0)
    walls.sort()
    med = walls[len(walls) // 2] if walls else 0.0
    wps = words / sum(walls) if walls else 0.0
    print(f"\n=== {system} | en-{args.pair} ===")
    print(f"  docs×repeats measured : {len(walls)}")
    print(f"  s/doc  median         : {med:.2f}s   (min {min(walls, default=0):.2f} / max {max(walls, default=0):.2f})")
    print(f"  throughput            : {wps:.2f} source-words/s")
    print(f"  hypotheses written    : {n_hyp}  -> {out/'hypotheses.jsonl'}")
    print(f"  latency records       : -> {out/'latency.jsonl'}")


if __name__ == "__main__":
    main()
