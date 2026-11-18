"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401


def run(ds, cfg):
    c = cfg["label_format"]
    issues = []

    for lp in ds.label_files():
        stem = lp.stem
        lines = ds.read_label_lines(stem)
        if not lines:
            continue

        seen = []  # các box hợp lệ trước đó trong file: (cls_id, (cx, cy, w, h))
        size = ds.image_size(stem)

        for line_no, text in lines:
            # 1. Định dạng 5 cột số
            try:
                cls_id, cx, cy, w, h = parse_yolo_line(text)
            except ValueError as e:
                issues.append(Issue("label_format", ERROR, "BAD_FORMAT", stem, str(e), line=line_no))
                continue

            # 2. ID lớp hợp lệ (0 .. len(classes)-1)
            if not (0 <= cls_id < len(ds.classes)):
                issues.append(
                    Issue("label_format", ERROR, "BAD_CLASS_ID", stem, f"id lớp {cls_id} ngoài 0..{len(ds.classes) - 1}", line=line_no)
                )
                continue

            # 3. Box rộng/cao <= 0
            if w <= 0 or h <= 0:
                issues.append(Issue("label_format", ERROR, "BOX_ZERO", stem, f"box rỗng w={w}, h={h}", line=line_no))
                continue

            # 4. Box tràn ra ngoài [0, 1] vượt ngưỡng overflow_tol
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            overflow = max(0.0, -x1, -y1, x2 - 1.0, y2 - 1.0)
            if overflow > c["overflow_tol"]:
                issues.append(
                    Issue(
                        "label_format",
                        ERROR,
                        "BOX_OUT_OF_IMAGE",
                        stem,
                        f"box tràn {overflow:.4f}",
                        line=line_no,
                        extra={"overflow": round(overflow, 6)},
                    )
                )

            # 5. Box trùng lặp cùng lớp với một dòng trước đó
            box = (cx, cy, w, h)
            for prev_cls, prev_box in seen:
                if prev_cls == cls_id and iou_xywh(box, prev_box) > c["dup_box_iou"]:
                    issues.append(Issue("label_format", ERROR, "DUP_BOX", stem, f"trùng box cùng lớp {cls_id}", line=line_no))
                    break
            seen.append((cls_id, box))

            # 6. Cảnh báo box quá nhỏ (px)
            if size is not None:
                W, H = size
                short_px = min(w * W, h * H)
                if short_px < c["min_box_px"]:
                    issues.append(Issue("label_format", WARN, "BOX_TINY", stem, f"cạnh ngắn {short_px:.1f}px < {c['min_box_px']}px", line=line_no))

            # 7. Cảnh báo box phủ gần hết ảnh
            if w * h > c["max_box_frac"]:
                issues.append(Issue("label_format", WARN, "BOX_HUGE", stem, f"diện tích {w * h:.2f} > {c['max_box_frac']}", line=line_no))

    return issues