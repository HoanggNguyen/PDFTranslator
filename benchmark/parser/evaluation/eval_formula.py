"""Đánh giá nhận dạng CÔNG THỨC (isolated formula) parser-vs-GT trên OmniDocBench.

Vấn đề "không đồng nhất": GT lưu LaTeX bọc ``$$...$$``; parser xuất
``<math display="block"> ...thân LaTeX... </math>``. Thân hai bên đều là LaTeX
nên chỉ cần BÓC LỚP BỌC rồi so:
  * GT  : bỏ ``$$ $ \\[ \\] \\( \\)`` ngoài cùng, gom khoảng trắng/newline.
  * pred: bỏ thẻ ``<math ...>`` / ``</math>`` (và delimiter nếu có), gom trắng.

Matching theo CONTAINMENT + UNION (giống eval_layout): với mỗi công thức GT, gom
mọi Equation của parser nằm gọn trong nó rồi NỐI lại theo thứ tự đọc — chịu được
trường hợp parser over-split 1 công thức thành nhiều mảnh.

Metric = Normalized Edit Distance (Levenshtein/max(len), đúng công thức
OmniDocBench). Báo:
  * edit_all      : tính trên MỌI công thức GT (GT không match -> pred rỗng -> phạt
                    hết) => phản ánh cả nhận dạng LẪN sót detect.
  * edit_matched  : chỉ trên công thức GT có ít nhất 1 Equation phủ => chất lượng
                    nhận dạng thuần, tách khỏi lỗi detect.
  * coverage      : tỉ lệ công thức GT được phủ (recall detect).
  * pred_unmatched: số Equation của parser không rơi vào công thức GT nào (FP).
Chia theo language / layout / subset / data_source (chú ý subset=equation_hard).

LƯU Ý: đây là edit distance trên LaTeX -> nhạy với khác ký hiệu
(``\\left[`` vs ``\\left\\lbrack``, ``dx`` vs ``\\partial x``). Chuẩn vàng là
**CDM** (render ra ảnh rồi so) nhưng cần môi trường KaTeX/texlive riêng — để dành.

Ví dụ
-----
    # chạy từ benchmark/parser/
    python evaluation/eval_formula.py \
        --gt data/OmniDocBench.json --pred parser_results \
        --mapping parser_results/mapping.json --out eval_results/eval_report_formula.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import module cạnh bên
import eval_layout as E   # tái dùng geometry, _dist, load_gt, build_pred_index

SLICE_KEYS = ("language", "layout", "subset", "data_source")
_MATH_TAG = re.compile(r"</?math[^>]*>", re.IGNORECASE)
_WS = re.compile(r"\s+")
_DELIMS = (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$"))


def norm_formula(s: str) -> str:
    """Bóc lớp bọc (math tag + delimiter) + thẻ định dạng -> LaTeX trần.

    Quan trọng: parser hay nhét thẻ định dạng (``<b>49</b>`` = số bài tập, ``<sub>``,
    ``<sup>``…) vào Equation. Nếu không bóc, thẻ lọt vào LaTeX -> CDM render ra ký
    hiệu rác -> điểm ≈ 0 oan. Bóc thẻ (giữ nội dung) + unescape như norm_ocr.
    """
    s = E._html.unescape(s or "")
    s = _MATH_TAG.sub(" ", s).strip()
    s = E._FMT_TAGS.sub("", s)          # <b><i><u><sub><sup>... (giữ nội dung)
    changed = True
    while changed:                      # bóc nhiều lớp delimiter lồng nhau nếu có
        changed = False
        for a, b in _DELIMS:
            if len(s) >= len(a) + len(b) and s.startswith(a) and s.endswith(b):
                s = s[len(a):len(s) - len(b)].strip()
                changed = True
    return _WS.sub(" ", s).strip()


def gt_equations(page: dict, drop_ignore: bool) -> list[dict]:
    out = []
    for d in page["dets"]:
        if d.get("category_type") != "equation_isolated":
            continue
        if drop_ignore and d.get("ignore"):
            continue
        out.append({
            "box": E.norm_box(E.poly_to_xyxy(d["poly"]), page["w"], page["h"]),
            "latex": norm_formula(d.get("latex") or d.get("text") or ""),
        })
    return out


def pred_equations(page: dict) -> list[dict]:
    W, H = page.get("page_width"), page.get("page_height")
    out = []
    for e in page["elements"]:
        if e.get("label") != "Equation":
            continue
        out.append({
            "box": E.norm_box(e["bbox_pdf"], W, H),
            "text": e.get("source_text") or "",
        })
    return out


class Acc:
    def __init__(self):
        self.n_gt = self.n_matched = 0
        self.n_pred = self.n_pred_unmatched = 0
        self.num_all = self.den_all = 0.0
        self.num_m = self.den_m = 0.0
        self.ratio_all = []        # per-formula normalized edit dist (over all GT)

    def summary(self):
        return {
            "gt_formulas": self.n_gt,
            "coverage": round(self.n_matched / self.n_gt, 4) if self.n_gt else None,
            "pred_formulas": self.n_pred,
            "pred_unmatched": self.n_pred_unmatched,
            "edit_all_micro": round(self.num_all / self.den_all, 4) if self.den_all else None,
            "edit_all_sample": round(sum(self.ratio_all) / len(self.ratio_all), 4) if self.ratio_all else None,
            "edit_matched_micro": round(self.num_m / self.den_m, 4) if self.den_m else None,
            "score_matched": round(1 - self.num_m / self.den_m, 4) if self.den_m else None,
        }


def score_page(gts, preds, member_thr):
    used = [False] * len(preds)
    ps = {"n_gt": len(gts), "n_matched": 0, "n_pred": len(preds),
          "pairs": []}  # (dist, maxlen, matched_bool)
    for g in gts:
        members, idxs = [], []
        for i, p in enumerate(preds):
            if E.contain_ratio(p["box"], g["box"]) >= member_thr:
                members.append(p)
                idxs.append(i)
        gt_norm = g["latex"]
        if members:
            for i in idxs:
                used[i] = True
            members.sort(key=lambda m: (round(m["box"][1], 3), m["box"][0]))
            pred_norm = norm_formula(" ".join(m["text"] for m in members))
            ps["n_matched"] += 1
            matched = True
        else:
            pred_norm = ""
            matched = False
        if gt_norm or pred_norm:
            # Bỏ HẲN whitespace khi so edit distance: GT chèn dấu cách quanh mọi token
            # (`\mathbb { R }`) còn parser xuất gọn (`\mathbb{R}`) — khác biệt spacing
            # vô nghĩa về ngữ nghĩa/hiển thị. Chỉ bỏ ở ĐÂY (bước đo); norm_formula giữ
            # nguyên spacing để CDM render an toàn (không nối `\in A` -> `\inA`).
            pe, ge = _WS.sub("", pred_norm), _WS.sub("", gt_norm)
            d = E._dist(pe, ge)
            ps["pairs"].append((d, max(len(pe), len(ge)), matched))
    ps["n_pred_unmatched"] = sum(1 for u in used if not u)
    return ps


def evaluate(gt_pages, pred_index, member_thr, drop_ignore):
    slices = defaultdict(Acc)
    for img_name, pred_page in pred_index.items():
        gt_page = gt_pages.get(img_name)
        if gt_page is None:
            continue
        gts = gt_equations(gt_page, drop_ignore)
        preds = pred_equations(pred_page)
        if not gts and not preds:
            continue
        ps = score_page(gts, preds, member_thr)

        keys = ["all"]
        for k in SLICE_KEYS:
            v = gt_page["attr"].get(k)
            if isinstance(v, list):
                keys += [f"{k}={x}" for x in v]
            elif v is not None:
                keys.append(f"{k}={v}")

        for key in keys:
            a = slices[key]
            a.n_gt += ps["n_gt"]
            a.n_matched += ps["n_matched"]
            a.n_pred += ps["n_pred"]
            a.n_pred_unmatched += ps["n_pred_unmatched"]
            for d, mlen, matched in ps["pairs"]:
                a.num_all += d
                a.den_all += mlen
                a.ratio_all.append(d / mlen if mlen else 0.0)
                if matched:
                    a.num_m += d
                    a.den_m += mlen
    return slices


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--member-thr", type=float, default=0.5)
    ap.add_argument("--keep-ignore", action="store_true")
    ap.add_argument("--min-slice", type=int, default=20,
                    help="Chỉ in lát cắt có >= N công thức GT.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    gt_pages = E.load_gt(args.gt, "merged")
    mapping = json.load(open(args.mapping, encoding="utf-8"))
    pred_index = E.build_pred_index(args.pred, mapping)
    print(f"[formula] GT trang={len(gt_pages)}  pred trang={len(pred_index)}  "
          f"member_thr={args.member_thr}")

    slices = evaluate(gt_pages, pred_index, args.member_thr, not args.keep_ignore)
    report = {
        "config": {"member_thr": args.member_thr, "drop_ignore": not args.keep_ignore},
        "slices": {k: a.summary() for k, a in slices.items()},
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    a = slices["all"].summary()
    print("\n===== FORMULA (all) =====")
    print(f"  GT công thức = {a['gt_formulas']}   coverage(recall detect) = {a['coverage']}")
    print(f"  pred Equation = {a['pred_formulas']}   không khớp GT nào = {a['pred_unmatched']}")
    print(f"  edit_all      micro = {a['edit_all_micro']}   sample = {a['edit_all_sample']}   (gồm cả sót detect)")
    print(f"  edit_matched  micro = {a['edit_matched_micro']}   -> score = {a['score_matched']}  (nhận dạng thuần)")

    print("\n===== THEO LÁT CẮT (coverage / edit_matched / edit_all) =====")
    for key in sorted(slices):
        if key == "all":
            continue
        s = slices[key].summary()
        if s["gt_formulas"] < args.min_slice:
            continue
        print(f"  {key:28s} cov={str(s['coverage']):>6}  "
              f"editM={str(s['edit_matched_micro']):>6}  "
              f"editAll={str(s['edit_all_micro']):>6}  (gt={s['gt_formulas']})")

    if args.out:
        print(f"\n[formula] report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
