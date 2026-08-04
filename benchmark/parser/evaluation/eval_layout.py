"""Đánh giá output của StageAParser với ground-truth OmniDocBench (parser-vs-GT).

Thiết kế để chịu được lệch granularity (many-to-one / one-to-many) sinh ra khi
parser gộp nhiều thứ rời rạc vào một bbox rồi split (is_sparse_text_block), hoặc
ngược lại parser gộp nhiều box GT vào một:

  * Matching theo CONTAINMENT + UNION, không phải IoU 1-1:
      - recall (neo theo GT) : gom mọi pred nằm gọn (>= member_thr) trong 1 GT,
                               union lại rồi so IoU với GT.  -> gộp nhiều GT vào
                               1 pred sẽ bị phạt recall (đúng: parser thiếu tách).
      - precision (neo theo pred): đối xứng, gom mọi GT nằm gọn trong 1 pred.
                               -> over-split sẽ bị phạt precision (đúng).
  * Tách LOCALIZATION (class-agnostic) và CLASSIFICATION (label-agreement) để mẹo
    "label con = label mẹ" không bị phạt kép.
  * Chuẩn hoá bbox về [0,1] theo kích thước từng trang -> khỏi lệch DPI.
  * Chọn granularity GT: 'merged' (dùng box top-level như OmniDocBench) hoặc
    'fine' (bung merge_list ra sub-box; hợp với output đã split).
  * Báo cáo tổng + theo lát cắt: language / layout / subset / data_source.
  * OCR edit distance (chuẩn hoá) trên các nhóm văn bản; reading-order edit dist.

Để so BEFORE vs AFTER split: chạy 2 lần trên 2 thư mục parser_results khác nhau
(một bản dump trước split, một bản sau split) rồi đối chiếu report.

Ví dụ
-----
    # chạy từ benchmark/parser/
    python evaluation/eval_layout.py \
        --gt      data/OmniDocBench.json \
        --pred    parser_results \
        --mapping parser_results/mapping.json \
        --gt-granularity fine \
        --out     eval_results/eval_report_fine.json
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

# --------------------------------------------------------------------------- #
# Ánh xạ nhãn về NHÓM THÔ (ổn định hơn map chi tiết 28<->14).                  #
# Sửa trực tiếp ở đây nếu bạn muốn gộp/tách nhóm khác đi.                      #
# --------------------------------------------------------------------------- #
GT_GROUP = {
    "text_block": "text", "reference": "text", "list_group": "text",
    "code_txt": "code", "equation_explanation": "text",
    "title": "section",
    "equation_isolated": "formula", "equation_semantic": "formula",
    "table": "table",
    "figure": "figure",
    "figure_caption": "caption", "table_caption": "caption",
    "equation_caption": "caption", "code_txt_caption": "caption",
    "header": "header_footer", "footer": "header_footer", "page_number": "header_footer",
    "figure_footnote": "text", "table_footnote": "text", "page_footnote": "footnote",
    "abandon": "abandon",
    # *_mask, unknown_mask, ... -> "other" (mặc định)
}
PRED_GROUP = {
    "Text": "text", "ListItem": "text", "TableOfContents": "text", "Code": "code",
    "SectionHeader": "section",
    "Equation": "formula",
    "Table": "table",
    "Figure": "figure", "Picture": "figure",
    "Caption": "caption",
    "PageHeader": "header_footer", "PageFooter": "header_footer",
    "Footnote": "footnote",
    "Form": "text",
}

# Nhóm không đưa vào matching (không phải mục tiêu detect / không có đối ứng).
EXCLUDE_GROUPS = {"abandon", "other"}
# Nhóm tính OCR edit distance (văn bản thuần; formula/table so nội dung riêng).
OCR_GROUPS = {"text", "section", "caption", "footnote"}
# Các chiều lát cắt lấy từ page_attribute.
SLICE_KEYS = ("language", "layout", "subset", "data_source")
IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # .50 .. .95


# --------------------------------------------------------------------------- #
# Hình học                                                                     #
# --------------------------------------------------------------------------- #
def poly_to_xyxy(poly: list[float]) -> list[float]:
    xs, ys = poly[0::2], poly[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def norm_box(b: list[float], w: float, h: float) -> list[float]:
    if not w or not h:
        return [0.0, 0.0, 0.0, 0.0]
    return [b[0] / w, b[1] / h, b[2] / w, b[3] / h]


def area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter_area(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def iou(a: list[float], b: list[float]) -> float:
    inter = inter_area(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def contain_ratio(inner: list[float], outer: list[float]) -> float:
    """Tỉ lệ diện tích của `inner` nằm trong `outer` (0..1)."""
    a = area(inner)
    return inter_area(inner, outer) / a if a > 0 else 0.0


def union_box(boxes: list[list[float]]) -> list[float]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


# --------------------------------------------------------------------------- #
# Edit distance. Ưu tiên python-Levenshtein (C). Không có thì dùng Myers        #
# bit-parallel thuần Python (exact, ~300x nhanh hơn DP trên chuỗi dài).         #
# --------------------------------------------------------------------------- #
def _myers(a, b) -> int:
    """Levenshtein (Hyyrö/Myers bit-parallel). Dùng được cho str và list."""
    if len(a) == 0 or len(b) == 0:
        return max(len(a), len(b))
    if len(a) > len(b):
        a, b = b, a
    m = len(a)
    mask = (1 << m) - 1
    Peq = {}
    for i, c in enumerate(a):
        Peq[c] = Peq.get(c, 0) | (1 << i)
    Pv, Mv, score, last = mask, 0, m, 1 << (m - 1)
    for c in b:
        Eq = Peq.get(c, 0)
        Xv = Eq | Mv
        Xh = (((Eq & Pv) + Pv) ^ Pv) | Eq
        Ph = (Mv | ~(Xh | Pv)) & mask
        Mh = (Pv & Xh) & mask
        if Ph & last:
            score += 1
        elif Mh & last:
            score -= 1
        Ph = ((Ph << 1) | 1) & mask
        Mh = (Mh << 1) & mask
        Pv = (Mh | ~(Xv | Ph)) & mask
        Mv = (Ph & Xv) & mask
    return score


try:
    import Levenshtein as _Lev

    def _dist(a, b) -> int:
        if isinstance(a, str) and isinstance(b, str):
            return _Lev.distance(a, b)
        return _myers(a, b)
except Exception:  # pragma: no cover
    def _dist(a, b) -> int:
        return _myers(a, b)


def norm_edit(a, b) -> float:
    """Normalized edit distance = dist / max(len). 0 = giống hệt."""
    m = max(len(a), len(b))
    return _dist(a, b) / m if m else 0.0


import html as _html
import re as _re

_MATH_BLOCK = _re.compile(r"<math\b[^>]*>(.*?)</math>", _re.IGNORECASE | _re.DOTALL)
_INLINE_DOLLAR = _re.compile(r"\$\$?(.+?)\$\$?", _re.DOTALL)   # $...$ hoặc $$...$$
_FMT_TAGS = _re.compile(
    r"</?(?:b|i|u|s|em|strong|del|mark|sub|sup|small|big|tt|code|span|font|br|p|h[1-6])\b[^>]*>",
    _re.IGNORECASE)
_ANY_TAG = _re.compile(r"</?[a-zA-Z][^>]*>")
_WS = _re.compile(r"\s+")
_PLACEHOLDER = "░"   # token thay cho công thức khi --mask-math


def norm_ocr(s: str, mask_math: bool = False) -> str:
    """Chuẩn hoá text cho đo OCR, tránh phạt oan do thẻ HTML / lớp bọc công thức.

    - Hợp nhất inline math: pred ``<math>BODY</math>`` và GT ``$BODY$`` -> BODY
      (hoặc placeholder nếu mask_math) để hai bên so được.
    - Bóc các thẻ định dạng (<b>,<i>,<sub>,<sup>,...) giữ lại nội dung bên trong.
    - Unescape HTML entity, gom khoảng trắng.
    """
    s = _html.unescape(s or "")
    if mask_math:
        s = _MATH_BLOCK.sub(f" {_PLACEHOLDER} ", s)
        s = _INLINE_DOLLAR.sub(f" {_PLACEHOLDER} ", s)
    else:
        s = _MATH_BLOCK.sub(lambda m: " " + m.group(1).strip() + " ", s)
        s = _INLINE_DOLLAR.sub(lambda m: " " + m.group(1).strip() + " ", s)
    s = _FMT_TAGS.sub("", s)
    s = _ANY_TAG.sub("", s)
    return _WS.sub(" ", s).strip()


def norm_text(s: str) -> str:                     # giữ tương thích (chỉ gom trắng)
    return " ".join((s or "").split())


# --------------------------------------------------------------------------- #
# Nạp dữ liệu                                                                  #
# --------------------------------------------------------------------------- #
def load_gt(path: Path, granularity: str) -> dict:
    data = json.load(open(path, encoding="utf-8"))
    pages = {}
    for p in data:
        info = p["page_info"]
        dets = []
        for d in p["layout_dets"]:
            if granularity == "fine" and d.get("merge_list"):
                dets.extend(d["merge_list"])
            else:
                dets.append(d)
        pages[info["image_path"]] = {
            "w": info["width"], "h": info["height"],
            "attr": info.get("page_attribute", {}), "dets": dets,
        }
    return pages


def prep_gt(page: dict, drop_ignore: bool) -> list[dict]:
    out = []
    for d in page["dets"]:
        if drop_ignore and d.get("ignore"):
            continue
        grp = GT_GROUP.get(d.get("category_type"), "other")
        if grp in EXCLUDE_GROUPS:
            continue
        txt = d.get("text") or d.get("latex") or d.get("html") or ""
        out.append({
            "box": norm_box(poly_to_xyxy(d["poly"]), page["w"], page["h"]),
            "group": grp, "cat": d.get("category_type"),
            "text": txt, "order": d.get("order"),
        })
    return out


def prep_pred(page: dict) -> list[dict]:
    out = []
    W, H = page.get("page_width"), page.get("page_height")
    for i, e in enumerate(page["elements"]):
        grp = PRED_GROUP.get(e.get("label"), "other")
        if grp in EXCLUDE_GROUPS:
            continue
        out.append({
            "box": norm_box(e["bbox_pdf"], W, H),
            "group": grp, "label": e.get("label"),
            "text": e.get("source_text") or "",
            "order": i,               # thứ tự parser xuất (đã là reading order)
        })
    return out


# --------------------------------------------------------------------------- #
# Matching containment + union (bất đối xứng theo thiết kế)                     #
# --------------------------------------------------------------------------- #
def match_side(anchors: list[dict], others: list[dict], member_thr: float):
    """Với mỗi anchor, gom `others` nằm gọn trong nó (>= member_thr), union lại.

    Trả list record: {anchor, members, union_iou, group_ok}.
    """
    records = []
    for a in anchors:
        members = [o for o in others if contain_ratio(o["box"], a["box"]) >= member_thr]
        if members:
            u = union_box([m["box"] for m in members])
            u_iou = iou(u, a["box"])
            grp_counts = Counter(m["group"] for m in members)
            group_ok = grp_counts.most_common(1)[0][0] == a["group"]
        else:
            u_iou, group_ok = 0.0, False
        records.append({"anchor": a, "members": members,
                        "union_iou": u_iou, "group_ok": group_ok})
    return records


def read_order_seq(matched_pairs: list[tuple]) -> float:
    """Normalized edit distance giữa thứ tự đọc GT và thứ tự GỐC parser xuất.

    matched_pairs: list of (gt, pred_union_box, pred_order) với gt['order'] hợp lệ.
    pred_order = chỉ số element nhỏ nhất của các pred thành viên = vị trí parser đặt
    vùng này trong luồng đọc (parser vốn xuất theo reading order, xử lý đa cột đúng —
    KHÔNG suy lại từ (y, x) vì sort (y,x) đọc ngang qua cột sẽ sai ở layout đa cột).
    """
    pairs = [(g, ub, po) for g, ub, po in matched_pairs if g.get("order") is not None]
    if len(pairs) < 2:
        return None
    ids = list(range(len(pairs)))
    gt_seq = [i for i, _ in sorted(zip(ids, pairs), key=lambda t: t[1][0]["order"])]
    pred_seq = [i for i, _ in sorted(zip(ids, pairs), key=lambda t: t[1][2])]
    return norm_edit(gt_seq, pred_seq)


# --------------------------------------------------------------------------- #
# Localization matchers (tính TP theo từng ngưỡng IoU).                        #
#  - COCO 1-1  : ghép tham lam theo IoU giảm dần, mỗi GT/pred dùng 1 lần       #
#                (chuẩn detection; không có confidence nên xếp theo IoU).      #
#  - COMPONENT : đồ thị chồng lấn -> thành phần liên thông; cụm 'khớp' nếu     #
#                IoU(union_GT, union_pred) >= t -> mọi GT/pred trong cụm là TP #
#                (một phép ghép nhất quán, xử lý N-M).                          #
# Trả dict {t: (tp_gt, tp_pred)} cho từng ngưỡng.                              #
# --------------------------------------------------------------------------- #
def coco_tp_by_threshold(gts, preds, thresholds):
    pairs = []
    for i, g in enumerate(gts):
        for j, p in enumerate(preds):
            v = iou(g["box"], p["box"])
            if v > 0:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    out = {}
    for t in thresholds:
        ug, up, tp = set(), set(), 0
        for v, i, j in pairs:
            if v < t:
                break                       # pairs xếp giảm dần
            if i in ug or j in up:
                continue
            ug.add(i); up.add(j); tp += 1
        out[t] = (tp, tp)                    # ghép 1-1: tp_gt == tp_pred
    return out


def component_tp_by_threshold(gts, preds, thresholds):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for i in range(len(gts)):
        find(("g", i))
    for j in range(len(preds)):
        find(("p", j))
    for i, g in enumerate(gts):
        for j, p in enumerate(preds):
            gb, pb = g["box"], p["box"]
            if (iou(gb, pb) >= 0.5 or contain_ratio(gb, pb) >= 0.5
                    or contain_ratio(pb, gb) >= 0.5):
                union(("g", i), ("p", j))
    comps = defaultdict(lambda: {"g": [], "p": []})
    for i, g in enumerate(gts):
        comps[find(("g", i))]["g"].append(g["box"])
    for j, p in enumerate(preds):
        comps[find(("p", j))]["p"].append(p["box"])
    comp_list = []
    for c in comps.values():
        if c["g"] and c["p"]:
            uiou = iou(union_box(c["g"]), union_box(c["p"]))
            comp_list.append((uiou, len(c["g"]), len(c["p"])))
    out = {}
    for t in thresholds:
        tg = sum(ng for uiou, ng, _ in comp_list if uiou >= t)
        tp = sum(npd for uiou, _, npd in comp_list if uiou >= t)
        out[t] = (tg, tp)
    return out


MATCHERS = ("coco", "component")


# --------------------------------------------------------------------------- #
# Tổng hợp                                                                     #
# --------------------------------------------------------------------------- #
class Acc:
    """Bộ đếm tích luỹ cho 1 lát cắt."""
    def __init__(self):
        self.gt_total = 0
        self.pred_total = 0
        # loc[matcher][t] = [tp_gt, tp_pred] cho từng matcher (coco / component)
        self.loc = {m: {t: [0, 0] for t in IOU_THRESHOLDS} for m in MATCHERS}
        self.cls_matched = 0     # cặp recall khớp @0.5
        self.cls_correct = 0     # trong đó đúng nhóm
        self.edit_num = 0.0      # normalized-by-max (kiểu OmniDocBench Edit_dist)
        self.edit_den = 0.0
        self.cer_num = 0.0       # CER: char edits / độ dài GT
        self.cer_den = 0.0
        self.wer_num = 0.0       # WER: word edits / số từ GT
        self.wer_den = 0.0
        self.ocr_pairs = 0
        self.ro_scores = []

    def add(self, ps: dict):
        """Cộng dồn contribution của 1 trang (đã tính sẵn một lần)."""
        self.gt_total += ps["gt_total"]
        self.pred_total += ps["pred_total"]
        for m in MATCHERS:
            for t in IOU_THRESHOLDS:
                self.loc[m][t][0] += ps["loc"][m][t][0]
                self.loc[m][t][1] += ps["loc"][m][t][1]
        self.cls_matched += ps["cls_matched"]
        self.cls_correct += ps["cls_correct"]
        self.edit_num += ps["edit_num"]
        self.edit_den += ps["edit_den"]
        self.cer_num += ps["cer_num"]
        self.cer_den += ps["cer_den"]
        self.wer_num += ps["wer_num"]
        self.wer_den += ps["wer_den"]
        self.ocr_pairs += ps["ocr_pairs"]
        if ps["ro"] is not None:
            self.ro_scores.append(ps["ro"])

    def prf(self, m, t):
        tg, tp = self.loc[m][t]
        r = tg / self.gt_total if self.gt_total else 0.0
        p = tp / self.pred_total if self.pred_total else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    def _loc_block(self, m):
        p50, r50, f50 = self.prf(m, 0.5)
        p75, r75, f75 = self.prf(m, 0.75)
        mf = sum(self.prf(m, t)[2] for t in IOU_THRESHOLDS) / len(IOU_THRESHOLDS)
        return {
            "precision@0.5": round(p50, 4), "recall@0.5": round(r50, 4),
            "f1@0.5": round(f50, 4), "f1@0.75": round(f75, 4),
            "mF1@[.5:.95]": round(mf, 4),
        }

    def summary(self):
        return {
            "gt_boxes": self.gt_total, "pred_boxes": self.pred_total,
            # localization báo cả 2 matcher: coco (chuẩn, nghiêm) + component (độ phủ)
            "localization": {m: self._loc_block(m) for m in MATCHERS},
            "classification": {
                "label_accuracy_on_matched@0.5":
                    round(self.cls_correct / self.cls_matched, 4) if self.cls_matched else None,
                "matched_pairs": self.cls_matched,
            },
            "ocr": {
                "edit_distance_micro":
                    round(self.edit_num / self.edit_den, 4) if self.edit_den else None,
                "CER":
                    round(self.cer_num / self.cer_den, 4) if self.cer_den else None,
                "WER":
                    round(self.wer_num / self.wer_den, 4) if self.wer_den else None,
                "text_pairs": self.ocr_pairs,
            },
            "reading_order": {
                "edit_distance_mean":
                    round(sum(self.ro_scores) / len(self.ro_scores), 4) if self.ro_scores else None,
                "pages": len(self.ro_scores),
            },
        }


def score_page(gts, preds, member_thr, mask_math=False) -> dict:
    """Tính TẤT CẢ contribution của 1 trang MỘT LẦN (edit distance không lặp lại)."""
    # rec (union-gather phía GT) chỉ dùng cho classification / OCR / reading-order.
    rec = match_side(gts, preds, member_thr)

    ps = {
        "gt_total": len(gts), "pred_total": len(preds),
        # Localization: tính TP theo 2 matcher độc lập (KHÔNG dùng cho OCR/cls/RO).
        "loc": {
            "coco": coco_tp_by_threshold(gts, preds, IOU_THRESHOLDS),
            "component": component_tp_by_threshold(gts, preds, IOU_THRESHOLDS),
        },
        "cls_matched": 0, "cls_correct": 0,
        "edit_num": 0.0, "edit_den": 0.0,
        "cer_num": 0.0, "cer_den": 0.0, "wer_num": 0.0, "wer_den": 0.0,
        "ocr_pairs": 0, "ro": None,
    }

    matched_pairs = []
    for r in rec:
        if r["union_iou"] < 0.5 or not r["members"]:
            continue
        ps["cls_matched"] += 1
        if r["group_ok"]:
            ps["cls_correct"] += 1
        g = r["anchor"]
        matched_pairs.append((g, union_box([m["box"] for m in r["members"]]),
                              min(m["order"] for m in r["members"])))
        if g["group"] in OCR_GROUPS and g["text"]:
            members = sorted(r["members"], key=lambda m: (round(m["box"][1], 3), m["box"][0]))
            pred_text = norm_ocr(" ".join(m["text"] for m in members), mask_math)
            gt_text = norm_ocr(g["text"], mask_math)
            if pred_text or gt_text:
                ps["edit_num"] += _dist(pred_text, gt_text)
                ps["edit_den"] += max(len(pred_text), len(gt_text))
                ps["ocr_pairs"] += 1
                if gt_text:                       # CER chuẩn hoá theo GT (reference)
                    ps["cer_num"] += _dist(pred_text, gt_text)
                    ps["cer_den"] += len(gt_text)
                    # WER chỉ có nghĩa với ngôn ngữ tách từ bằng dấu cách.
                    # CJK (không dấu cách) -> dùng CER, bỏ khỏi WER để tránh nhiễu.
                    if " " in gt_text:
                        gw, pw = gt_text.split(), pred_text.split()
                        ps["wer_num"] += _dist(pw, gw)
                        ps["wer_den"] += len(gw)

    ps["ro"] = read_order_seq(matched_pairs)
    return ps


def evaluate(gt_pages, pred_index, member_thr, drop_ignore, mask_math=False):
    slices = defaultdict(Acc)  # key -> Acc ; key "all" luôn có

    for img_name, pred_page in pred_index.items():
        gt_page = gt_pages.get(img_name)
        if gt_page is None:
            continue
        gts = prep_gt(gt_page, drop_ignore)
        preds = prep_pred(pred_page)
        ps = score_page(gts, preds, member_thr, mask_math)   # <-- tính 1 lần

        keys = ["all"]
        for k in SLICE_KEYS:
            v = gt_page["attr"].get(k)
            if isinstance(v, list):
                keys += [f"{k}={x}" for x in v]
            elif v is not None:
                keys.append(f"{k}={v}")

        for key in keys:                            # <-- chỉ cộng số học
            slices[key].add(ps)

    return slices


# --------------------------------------------------------------------------- #
def build_pred_index(pred_dir: Path, mapping: dict) -> dict:
    """image_name -> pred page dict, dựa trên mapping (page_index k -> images[k])."""
    by_stem = {Path(e["pdf"]).stem: e["images"] for e in mapping["pdfs"]}
    index = {}
    missing = 0
    for f in sorted(glob.glob(str(pred_dir / "*.json"))):
        stem = Path(f).stem
        images = by_stem.get(stem)
        if images is None:
            continue
        doc = json.load(open(f, encoding="utf-8"))
        for page in doc["pages"]:
            k = page.get("page_index")
            if k is None or k >= len(images):
                missing += 1
                continue
            index[images[k]] = page
    if missing:
        print(f"[warn] {missing} trang pred không map được về ảnh.")
    return index


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True, help="OmniDocBench.json")
    ap.add_argument("--pred", type=Path, required=True, help="Thư mục parser_results/")
    ap.add_argument("--mapping", type=Path, required=True, help="mapping.json")
    ap.add_argument("--out", type=Path, default=None, help="File report JSON.")
    ap.add_argument("--gt-granularity", choices=["merged", "fine"], default="fine",
                    help="merged=box top-level (như OmniDocBench); fine=bung merge_list.")
    ap.add_argument("--member-thr", type=float, default=0.5,
                    help="Ngưỡng containment để coi 1 box là thành viên (0..1).")
    ap.add_argument("--keep-ignore", action="store_true",
                    help="Giữ cả box GT ignore=true (mặc định bỏ).")
    ap.add_argument("--mask-math", action="store_true",
                    help="Thay công thức inline bằng 1 token khi đo OCR (đo text thuần).")
    ap.add_argument("--min-slice", type=int, default=30,
                    help="Chỉ in lát cắt có >= N trang-box (đỡ nhiễu).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    gt_pages = load_gt(args.gt, args.gt_granularity)
    mapping = json.load(open(args.mapping, encoding="utf-8"))
    pred_index = build_pred_index(args.pred, mapping)
    print(f"[eval] GT trang={len(gt_pages)}  pred trang map được={len(pred_index)}  "
          f"granularity={args.gt_granularity}  member_thr={args.member_thr}")

    slices = evaluate(gt_pages, pred_index, args.member_thr,
                      not args.keep_ignore, args.mask_math)

    report = {
        "config": {
            "gt_granularity": args.gt_granularity, "member_thr": args.member_thr,
            "drop_ignore": not args.keep_ignore, "mask_math": args.mask_math,
            "iou_thresholds": IOU_THRESHOLDS,
            "excluded_groups": sorted(EXCLUDE_GROUPS), "ocr_groups": sorted(OCR_GROUPS),
        },
        "slices": {k: acc.summary() for k, acc in slices.items()},
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    a = slices["all"].summary()
    print("\n===== TỔNG (all) =====")
    print(f"  GT boxes={a['gt_boxes']}  pred boxes={a['pred_boxes']}")
    for m in MATCHERS:
        loc = a["localization"][m]
        print(f"  LOCALIZATION[{m:9s}] P@.5={loc['precision@0.5']}  R@.5={loc['recall@0.5']}  "
              f"F1@.5={loc['f1@0.5']}  F1@.75={loc['f1@0.75']}  mF1={loc['mF1@[.5:.95]']}")
    print(f"  CLASSIFY      label_acc@.5={a['classification']['label_accuracy_on_matched@0.5']} "
          f"(trên {a['classification']['matched_pairs']} cặp)")
    print(f"  OCR           edit_dist={a['ocr']['edit_distance_micro']}  "
          f"CER={a['ocr']['CER']}  WER={a['ocr']['WER']}  "
          f"(trên {a['ocr']['text_pairs']} vùng)")
    print(f"  READING ORDER edit_dist={a['reading_order']['edit_distance_mean']} "
          f"({a['reading_order']['pages']} trang)")

    print("\n===== THEO LÁT CẮT (F1@.5 / labelAcc / OCRedit) =====")
    for key in sorted(slices):
        if key == "all":
            continue
        s = slices[key].summary()
        if s["gt_boxes"] < args.min_slice:
            continue
        print(f"  {key:28s} F1@.5 coco={s['localization']['coco']['f1@0.5']:.3f} "
              f"comp={s['localization']['component']['f1@0.5']:.3f}  "
              f"labelAcc={str(s['classification']['label_accuracy_on_matched@0.5']):>6}  "
              f"CER={str(s['ocr']['CER']):>6}  (gt={s['gt_boxes']})")

    if args.out:
        print(f"\n[eval] report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
