"""Dẫn xuất GT cell của PubTables-1M bằng CHÍNH code của microsoft/table-transformer.

Vì sao không tự viết: XML PubTables-1M **không có object nào tên ``cell``** (đã đếm trên
93.834 file split test, chỉ có 6 class). Cell phải dẫn xuất từ ``table row`` × ``table
column`` × ``table spanning cell``. Bản tự viết đầu tiên bỏ mất bước ``align_supercells()``
— snap ô span vào chỉ số hàng/cột TRƯỚC khi hấp thụ subcell — nên lệch upstream ở các bảng
có ô span, tức **42,0%** split test. Nay gọi thẳng ``vendor/tatr_postprocess.py``
(bản sao nguyên văn, MIT).

Đường đi tái hiện đúng ``eval.py`` của upstream (hàm ``eval_tsr_sample``, dòng ~456):

    XML -> [{bbox, label, score=1.0}]  (GT nên score = 1.0, giống ``true_scores``)
        -> apply_class_thresholds
        -> objects_to_cells(table, objects, tokens, class_names, class_thresholds)
             -> objects_to_table_structures  (refine/nms/align rows, columns, supercells)
             -> table_structure_to_cells     (cell = row ∩ column, gộp supercell)

HAI CHẾ ĐỘ, khác nhau ở chỗ có truyền word token hay không:

``--mode tatr`` (mặc định, KHÔNG cần file Words)
    Truyền ``tokens = []``. Đây là **code path có sẵn trong upstream**, không phải hack:
    ``refine_rows``/``refine_columns`` rẽ nhánh tường minh theo ``if len(tokens) > 0``,
    và khi rỗng thì dùng NMS thuần. Khối cuối của ``table_structure_to_cells`` (co box
    hàng/cột về phạm vi text) cũng tự thành no-op vì mọi danh sách ``min_x_values_by_*``
    đều rỗng.
    => Cell giữ **vùng ô đầy đủ**, có cả ô rỗng. Đúng convention mà cell detector xuất ra.

``--mode tatr-words`` (cần ``PubTables-1M-Structure_Table_Words.tar.gz``, 3,89 GiB)
    Tái hiện **byte-for-byte** đường đi của ``eval.py``: có token thì thêm
    ``nms_by_containment`` + ``remove_objects_without_content`` (bỏ hàng/cột KHÔNG có
    chữ), và khối cuối co box hàng/cột về phạm vi text.
    => Số so được trực tiếp với GriTS mà table-transformer công bố, NHƯNG box trở nên
    bám text hơn, và **hàng/cột rỗng bị loại** — cần cân nhắc vì đó chính là thứ khiến
    PubTabNet/FinTabNet không dùng được cho detector này.

Khuyến nghị: báo ``tatr`` làm số chính (khớp convention của hệ thống), và ``tatr-words``
làm số phụ nếu cần so với literature. Nói rõ trong bài đã dùng chế độ nào.

Ví dụ
-----
    # chạy từ benchmark/parser/
    python evaluation/pubtables_gt.py --ann data/pubtables1m/test/PMC2268688_table_0.xml
    python evaluation/pubtables_gt.py --ann ...xml --mode tatr-words \
        --words data/pubtables1m/words
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vendor import tatr_postprocess as P  # noqa: E402

Box = list[float]

# Giống hệt eval.py của upstream (dòng 32-46). Thứ tự QUAN TRỌNG: label là chỉ số.
STRUCTURE_CLASS_NAMES = [
    "table", "table column", "table row", "table column header",
    "table projected row header", "table spanning cell", "no object",
]
STRUCTURE_CLASS_MAP = {k: v for v, k in enumerate(STRUCTURE_CLASS_NAMES)}
STRUCTURE_CLASS_THRESHOLDS = {
    "table": 0.5, "table column": 0.5, "table row": 0.5,
    "table column header": 0.5, "table projected row header": 0.5,
    "table spanning cell": 0.5, "no object": 10,
}


def _bbox(obj: ET.Element) -> Box:
    bb = obj.find("bndbox")
    return [float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]


def read_voc(path: Path) -> dict:
    """XML -> {'name','w','h','bboxes','labels'} theo đúng dạng eval.py cần."""
    root = ET.parse(path).getroot()
    size = root.find("size")
    bboxes, labels = [], []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in STRUCTURE_CLASS_MAP:
            continue
        bboxes.append(_bbox(obj))
        labels.append(STRUCTURE_CLASS_MAP[name])
    return {
        "name": path.stem,
        "w": int(size.find("width").text),
        "h": int(size.find("height").text),
        "bboxes": bboxes,
        "labels": labels,
    }


def quick_stats(path: Path) -> dict | None:
    """Đếm hàng/cột/ô-span THÔ từ XML, không chạy pipeline upstream.

    Dùng cho phân tầng lấy mẫu: nhanh hơn ~100x và không cần refine. Lưu ý số hàng/cột
    ở đây là số **chú thích gốc**; sau khi upstream refine (NMS, align) con số có thể
    khác chút. Với mục đích thiết kế mẫu thì số thô mới là số đúng, vì ta phân tầng theo
    đặc tính của dữ liệu nguồn.
    """
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return None
    n_row = n_col = n_span = 0
    has_table = False
    for obj in root.findall("object"):
        n = obj.find("name").text
        if n == "table row":
            n_row += 1
        elif n == "table column":
            n_col += 1
        elif n in ("table spanning cell", "table projected row header"):
            n_span += 1
        elif n == "table":
            has_table = True
    if not has_table or n_row == 0 or n_col == 0:
        return None
    return {"name": path.stem, "xml": path.name, "n_row": n_row, "n_col": n_col,
            "n_slots": n_row * n_col, "has_span": n_span > 0}


def table_bbox(path: Path) -> Box | None:
    """Bbox của object ``table`` trong XML. Dùng để crop ảnh trước khi chạy detector."""
    root = ET.parse(path).getroot()
    for obj in root.findall("object"):
        if obj.find("name").text == "table":
            return _bbox(obj)
    return None


def load_words(words_dir: Path, name: str) -> list[dict]:
    """Đọc file token của một bảng (chế độ tatr-words)."""
    for cand in (words_dir / f"{name}_words.json", words_dir / f"{name}.json"):
        if cand.exists():
            return json.loads(cand.read_text(encoding="utf-8"))
    return []


def objects_to_cells_like_eval(bboxes: list[Box], labels: list[int],
                               scores: list[float], page_tokens: list[dict]):
    """Bản sao của ``eval.py::objects_to_cells`` (dòng 54-81) — không đổi logic."""
    bboxes, scores, labels = P.apply_class_thresholds(
        bboxes, labels, scores, STRUCTURE_CLASS_NAMES, STRUCTURE_CLASS_THRESHOLDS)

    table_objects = [{"bbox": b, "score": s, "label": l}
                     for b, s, l in zip(bboxes, scores, labels)]
    table = {"objects": table_objects, "page_num": 0}

    table_class_objects = [o for o in table_objects
                           if o["label"] == STRUCTURE_CLASS_MAP["table"]]
    if len(table_class_objects) > 1:
        table_class_objects = sorted(table_class_objects,
                                     key=lambda x: x["score"], reverse=True)
    try:
        table_bbox = list(table_class_objects[0]["bbox"])
    except Exception:
        table_bbox = (0, 0, 1000, 1000)

    tokens_in_table = [t for t in page_tokens
                       if P.iob(t["bbox"], table_bbox) >= 0.5]

    return P.objects_to_cells(table, table_objects, tokens_in_table,
                              STRUCTURE_CLASS_NAMES, STRUCTURE_CLASS_THRESHOLDS)


def _normalise(cells: list[dict]) -> list[dict]:
    """Cell của upstream -> dạng eval_cells.py dùng."""
    out = []
    for c in cells:
        rows = sorted(c["row_nums"])
        cols = sorted(c["column_nums"])
        out.append({
            "box": [float(v) for v in c["bbox"]],
            "row": rows[0], "col": cols[0],
            "rowspan": rows[-1] - rows[0] + 1,
            "colspan": cols[-1] - cols[0] + 1,
            "is_spanning": len(rows) > 1 or len(cols) > 1,
            "is_header": bool(c.get("header", False)),
        })
    out.sort(key=lambda c: (c["row"], c["col"]))
    return out


def gt_for(path: Path, mode: str = "tatr", words_dir: Path | None = None) -> dict:
    """GT cell cho một bảng. Trả về cả hai biến thể span để báo cáo song song."""
    ann = read_voc(path)
    tokens: list[dict] = []
    if mode == "tatr-words":
        if words_dir is None:
            raise SystemExit("--mode tatr-words cần --words <thư mục>")
        tokens = load_words(words_dir, ann["name"])

    scores = [1.0] * len(ann["labels"])          # GT: score = 1.0, giống true_scores
    structures, cells, _ = objects_to_cells_like_eval(
        ann["bboxes"], ann["labels"], scores, tokens)

    merged = _normalise(cells)
    # Biến thể "unmerged": mọi grid cell, không gộp ô span. Dùng chính rows/columns đã
    # được upstream refine, nên hai biến thể chỉ khác nhau ở bước gộp.
    rows = structures.get("rows", [])
    cols = structures.get("columns", [])
    unmerged = []
    for ri, row in enumerate(rows):
        for ci, col in enumerate(cols):
            r = P.Rect(list(row["bbox"])).intersect(P.Rect(list(col["bbox"])))
            if r.get_area() <= 0:
                continue
            unmerged.append({"box": [float(v) for v in list(r)],
                             "row": ri, "col": ci, "rowspan": 1, "colspan": 1,
                             "is_spanning": False,
                             "is_header": bool(row.get("header", False))})

    n_span_src = sum(1 for lb in ann["labels"]
                     if STRUCTURE_CLASS_NAMES[lb] in
                     ("table spanning cell", "table projected row header"))
    return {
        "name": ann["name"], "w": ann["w"], "h": ann["h"],
        "mode": mode,
        "n_row": len(rows), "n_col": len(cols),
        "n_spanning": n_span_src,
        "has_span": n_span_src > 0,
        "cells_merged": merged,
        "cells_unmerged": unmerged,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann", type=Path, required=True)
    ap.add_argument("--mode", choices=("tatr", "tatr-words"), default="tatr")
    ap.add_argument("--words", type=Path, default=None)
    args = ap.parse_args()

    g = gt_for(args.ann, args.mode, args.words)
    print(f"{g['name']}  ảnh {g['w']}x{g['h']}  mode={g['mode']}")
    print(f"  {g['n_row']} hàng x {g['n_col']} cột  |  spanning nguồn = {g['n_spanning']}")
    print(f"  cells_unmerged = {len(g['cells_unmerged'])}"
          f"   cells_merged = {len(g['cells_merged'])}"
          f"   (span sau gộp = {sum(1 for c in g['cells_merged'] if c['is_spanning'])})")
    print(json.dumps(g["cells_merged"][:3], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
