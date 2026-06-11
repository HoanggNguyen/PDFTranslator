from typing import Any

from pdf2zh.parser.utils.bbox import polygon_to_bbox
from pdf2zh.parser.utils.ocr_text import extract_text_for_region, sort_text_lines


def is_sparse_text_block(
    ocr_result: Any,
    block_bbox: list[float],
    is_equation: bool,
) -> tuple[bool, list[Any]]:
    text_lines = extract_text_for_region(ocr_result, block_bbox)

    if is_equation and len(text_lines) > 1:
        return True, text_lines

    if len(text_lines) < 2:
        return False, text_lines

    rows = cluster_text_lines_into_rows(text_lines)

    GAP_MULTIPLIER = 1.0

    for row in rows:
        if len(row) < 2:
            continue

        row_boxes = []
        for line in row:
            box = get_line_bbox(line)
            if box is not None:
                row_boxes.append(box)

        row_boxes = sort_text_lines(row_boxes)

        for i in range(len(row_boxes) - 1):
            prev_box = row_boxes[i]
            curr_box = row_boxes[i + 1]

            gap = curr_box[0] - prev_box[2]

            prev_height = prev_box[3] - prev_box[1]
            curr_height = curr_box[3] - curr_box[1]
            avg_height = (prev_height + curr_height) / 2.0

            if gap > (avg_height * GAP_MULTIPLIER):
                return True, text_lines

    return False, text_lines


def cluster_text_lines_into_rows(lines: list[Any]) -> list[list[Any]]:
    _ROW_Y_OVERLAP_RATIO = 0.4
    rows: list[dict[str, Any]] = []

    for line in lines:
        line_bbox = get_line_bbox(line)
        if line_bbox is None:
            continue

        _, y0, _, y1 = line_bbox
        placed = False
        for row in rows:
            row_y0 = row["y0"]
            row_y1 = row["y1"]
            overlap = max(0.0, min(y1, row_y1) - max(y0, row_y0))
            line_height = max(1.0, y1 - y0)
            row_height = max(1.0, row_y1 - row_y0)
            overlap_ratio = overlap / min(line_height, row_height)
            if overlap_ratio >= _ROW_Y_OVERLAP_RATIO:
                row["lines"].append(line)
                row["y0"] = min(row_y0, y0)
                row["y1"] = max(row_y1, y1)
                placed = True
                break

        if not placed:
            rows.append({"y0": y0, "y1": y1, "lines": [line]})

    rows.sort(key=lambda row: (row["y0"], row["y1"]))
    return [row["lines"] for row in rows]


def get_line_bbox(line: Any) -> list[float] | None:
    line_bbox = getattr(line, "bbox", None)
    if line_bbox is not None:
        return list(line_bbox)
    if hasattr(line, "polygon"):
        return polygon_to_bbox(line.polygon)
    return None
