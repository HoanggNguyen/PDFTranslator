"""Đánh giá công thức bằng CDM (Character Detection Matching) — chuẩn vàng.

CDM render LaTeX (GT và pred) ra ảnh rồi khớp từng ký hiệu -> precision/recall/F1,
KHÔNG bị nhiễu bởi khác ký hiệu như edit distance (``\\left[`` vs ``\\left\\lbrack``…).

Script này TÁI DÙNG lớp ``CDM`` trong OmniDocBench/src/metrics/cdm và bộ ghép cặp
công thức của ``eval_formula.py`` (match Equation ↔ equation_isolated theo bbox).
Chỉ chạy trên các công thức GT ĐÃ được parser phủ (matched); công thức sót detect
đã phản ánh ở ``coverage`` của eval_formula.

YÊU CẦU HỆ THỐNG (CDM render bằng LaTeX + ImageMagick):
  * pdflatex, kpsewhich   (texlive)   -> apt install texlive-latex-extra texlive-latex-base
  * magick / convert      (ImageMagick) -> apt install imagemagick
  * python: numpy, Pillow  (đã thêm vào requirements-eval.txt)
Nếu thiếu, script báo rõ và thoát (không chạy dở).

Ví dụ (chạy nơi có texlive + imagemagick, vd Colab/Docker)
-----
    python eval_formula_cdm.py --gt ../../OmniDocBench.json --pred ../../parser_json \
        --mapping ../../mapping.json --omnidocbench ../../../OmniDocBench \
        --out ../../eval_report_formula_cdm.json --limit 200
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import eval_layout as E
import eval_formula as F


def ensure_magick() -> None:
    """CDM gọi hardcode lệnh ``magick`` (ImageMagick 7). Trên ImageMagick 6 chỉ có
    ``convert`` -> tạo shim ``magick`` trỏ về ``convert`` và nhét vào PATH.
    (build_tex_env của CDM copy os.environ nên shim này được thấy.)"""
    if shutil.which("magick"):
        return
    convert = shutil.which("convert")
    if not convert:
        return
    shim_dir = Path(tempfile.gettempdir()) / "cdm_magick_shim"
    shim_dir.mkdir(exist_ok=True)
    shim = shim_dir / "magick"
    shim.write_text(f'#!/bin/sh\nexec "{convert}" "$@"\n')
    shim.chmod(0o755)
    os.environ["PATH"] = str(shim_dir) + os.pathsep + os.environ.get("PATH", "")


def check_system_deps() -> list[str]:
    missing = []
    if not (shutil.which("pdflatex") and shutil.which("kpsewhich")):
        missing.append("pdflatex/kpsewhich (texlive)")
    if not (shutil.which("magick") or shutil.which("convert")):
        missing.append("magick/convert (ImageMagick)")
    for mod in ("numpy", "PIL", "scipy"):
        try:
            __import__(mod)
        except Exception:
            missing.append(f"python:{mod}")
    return missing


def resolve_omnidocbench(arg: Path | None) -> Path:
    if arg is not None:
        if (arg / "src" / "metrics" / "cdm").is_dir():
            return arg.resolve()
        raise FileNotFoundError(f"Không thấy src/metrics/cdm dưới {arg}")
    for cand in (Path(__file__).resolve().parents[3] / "OmniDocBench",
                 Path("../../../OmniDocBench"), Path("OmniDocBench")):
        if (cand / "src" / "metrics" / "cdm").is_dir():
            return cand.resolve()
    raise FileNotFoundError("Không định vị được OmniDocBench (truyền --omnidocbench).")


def collect_pairs(gt_pages, pred_index, member_thr):
    """Trả list (img_id, gt_latex, pred_latex, slice_keys) cho công thức đã match."""
    pairs = []
    for img_name, pred_page in pred_index.items():
        gt_page = gt_pages.get(img_name)
        if gt_page is None:
            continue
        gts = F.gt_equations(gt_page, drop_ignore=True)
        preds = F.pred_equations(pred_page)
        keys = ["all"]
        for k in F.SLICE_KEYS:
            v = gt_page["attr"].get(k)
            if isinstance(v, list):
                keys += [f"{k}={x}" for x in v]
            elif v is not None:
                keys.append(f"{k}={v}")
        for gi, g in enumerate(gts):
            members = [p for p in preds if E.contain_ratio(p["box"], g["box"]) >= member_thr]
            if not members:
                continue
            members.sort(key=lambda m: (round(m["box"][1], 3), m["box"][0]))
            pred_latex = F.norm_formula(" ".join(m["text"] for m in members))
            if g["latex"] or pred_latex:
                pairs.append((f"{img_name}#{gi}", g["latex"], pred_latex, keys))
    return pairs


class Acc:
    def __init__(self):
        self.tp = self.gt_tok = self.pred_tok = 0
        self.f1_list = []

    def add(self, m):
        self.tp += m.get("tp", 0)
        self.gt_tok += m.get("gt_tokens", 0)
        self.pred_tok += m.get("pred_tokens", 0)
        self.f1_list.append(m.get("F1_score", 0.0))

    def summary(self):
        r = self.tp / self.gt_tok if self.gt_tok else None
        p = self.tp / self.pred_tok if self.pred_tok else None
        f = (2 * p * r / (p + r)) if (p and r and (p + r)) else None
        return {
            "n": len(self.f1_list),
            "CDM_recall_micro": round(r, 4) if r is not None else None,
            "CDM_precision_micro": round(p, 4) if p is not None else None,
            "CDM_F1_micro": round(f, 4) if f is not None else None,
            "CDM_F1_mean": round(sum(self.f1_list) / len(self.f1_list), 4) if self.f1_list else None,
        }


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument("--omnidocbench", type=Path, default=None,
                    help="Đường dẫn repo OmniDocBench (để import CDM).")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--member-thr", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None, help="Chỉ chấm N cặp đầu (test).")
    ap.add_argument("--result-dir", type=Path, default=Path("./cdm_work"),
                    help="Thư mục tạm CDM render (mặc định ./cdm_work).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    ensure_magick()
    missing = check_system_deps()
    if missing:
        print("[cdm] THIẾU dependency, không chạy được CDM:", flush=True)
        for m in missing:
            print(f"    - {m}", flush=True)
        print("\nCài (Ubuntu/Colab):\n"
              "    apt-get install -y texlive-latex-base texlive-latex-extra "
              "texlive-fonts-recommended imagemagick\n"
              "    .venv/bin/pip install numpy Pillow", flush=True)
        return 2

    odb = resolve_omnidocbench(args.omnidocbench)
    # Import THẲNG gói cdm (thêm src/metrics vào path) để né src/__init__.py
    # vốn kéo theo cli/yaml/evaluate... rất nặng.
    sys.path.insert(0, str(odb / "src" / "metrics"))
    from cdm.cdm import cdm_metrics  # noqa: E402

    gt_pages = E.load_gt(args.gt, "merged")
    mapping = json.load(open(args.mapping, encoding="utf-8"))
    pred_index = E.build_pred_index(args.pred, mapping)
    pairs = collect_pairs(gt_pages, pred_index, args.member_thr)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"[cdm] {len(pairs)} cặp công thức đã match sẽ chấm bằng CDM "
          f"(render LaTeX -> có thể chậm)...", flush=True)

    args.result_dir.mkdir(parents=True, exist_ok=True)
    slices = defaultdict(Acc)
    for i, (img_id, gt_latex, pred_latex, keys) in enumerate(pairs, 1):
        try:
            m = cdm_metrics(gt_latex, pred_latex, save_vis=False,
                            tmp_dir=str(args.result_dir))
        except Exception as exc:
            print(f"  [{i}] lỗi CDM {img_id}: {exc!r}", flush=True)
            continue
        for key in keys:
            slices[key].add(m)
        if i % 50 == 0:
            print(f"  ...{i}/{len(pairs)}", flush=True)

    report = {"config": {"member_thr": args.member_thr, "n_pairs": len(pairs)},
              "slices": {k: a.summary() for k, a in slices.items()}}
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    a = slices["all"].summary()
    print("\n===== FORMULA CDM (all, trên công thức đã match) =====")
    print(f"  n = {a['n']}")
    print(f"  CDM F1 micro = {a['CDM_F1_micro']}  (precision={a['CDM_precision_micro']}, "
          f"recall={a['CDM_recall_micro']})")
    print(f"  CDM F1 mean  = {a['CDM_F1_mean']}")
    if args.out:
        print(f"\n[cdm] report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
