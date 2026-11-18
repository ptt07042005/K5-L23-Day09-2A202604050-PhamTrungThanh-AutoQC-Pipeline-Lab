"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401
from ..model_cache import get_preds  # noqa: F401


def run(ds, cfg, preds=None):
    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    c = cfg["second_opinion"]
    conf_thresh = c["conf"]
    min_area_px = c["min_area_px"]
    fp_conf = c["fp_conf"]
    iou_missing = c["iou_missing"]
    iou_match = c["iou_match"]
    iou_loose = c["iou_loose"]

    issues = []

    for p in ds.image_files():
        stem = p.stem
        lines = ds.read_label_lines(stem)
        if lines is None or stem not in preds:
            continue

        size = ds.image_size(stem)
        if size is None:
            continue
        W, H = size

        # Giữ số dòng gốc và bỏ qua dòng hỏng / id lớp ngoài phạm vi
        valid_labels = []
        for line_no, text in lines:
            try:
                cls_id, cx, cy, w, h = parse_yolo_line(text)
            except ValueError:
                continue
            if 0 <= cls_id < len(ds.classes):
                valid_labels.append((line_no, cls_id, (cx, cy, w, h)))

        img_preds = preds[stem]

        # Pha 1: kiểm từ dự đoán model -> nhãn
        for p_cls, p_conf, px, py, pw, ph in img_preds:
            if p_conf < conf_thresh:
                continue
            if (pw * W) * (ph * H) < min_area_px:
                continue

            p_box = (px, py, pw, ph)
            cands = []
            for _, l_cls, l_box in valid_labels:
                iou = iou_xywh(p_box, l_box)
                if iou >= iou_missing:
                    cands.append((iou, l_cls))

            cls_name = ds.classes[p_cls] if 0 <= p_cls < len(ds.classes) else str(p_cls)

            if not cands:
                issues.append(
                    Issue(
                        "second_opinion",
                        WARN,
                        "POSSIBLE_MISSING",
                        stem,
                        f"model thấy {cls_name} (conf={p_conf:.2f}) nhưng không có nhãn",
                        extra={"conf": round(p_conf, 4)},
                    )
                )
                continue

            same_ious = [iou for iou, l_cls in cands if l_cls == p_cls]
            if same_ious:
                best_same = max(same_ious)
                if best_same < iou_loose:
                    issues.append(
                        Issue(
                            "second_opinion",
                            WARN,
                            "POSSIBLE_LOOSE_BOX",
                            stem,
                            f"box {cls_name} lỏng (IoU={best_same:.2f} < {iou_loose})",
                            extra={"conf": round(p_conf, 4), "iou": round(best_same, 4)},
                        )
                    )
            else:
                best_other_iou, best_other_cls = max(cands, key=lambda x: x[0])
                if best_other_iou >= iou_match:
                    lbl_name = ds.classes[best_other_cls]
                    issues.append(
                        Issue(
                            "second_opinion",
                            WARN,
                            "POSSIBLE_WRONG_CLASS",
                            stem,
                            f"nhãn ghi {lbl_name} nhưng model thấy {cls_name} (IoU={best_other_iou:.2f})",
                            extra={"conf": round(p_conf, 4), "iou": round(best_other_iou, 4)},
                        )
                    )

        # Pha 2: kiểm từ nhãn -> dự đoán (dùng fp_conf, KHÔNG lọc min_area_px)
        fp_boxes = [(px, py, pw, ph) for _, p_conf, px, py, pw, ph in img_preds if p_conf >= fp_conf]
        for line_no, l_cls, l_box in valid_labels:
            max_iou = max((iou_xywh(l_box, pb) for pb in fp_boxes), default=0.0)
            if max_iou < iou_missing:
                issues.append(
                    Issue(
                        "second_opinion",
                        WARN,
                        "POSSIBLE_FALSE_POS",
                        stem,
                        f"box {ds.classes[l_cls]} không khớp dự đoán nào (max_IoU={max_iou:.2f})",
                        line=line_no,
                        extra={"iou": round(max_iou, 4)},
                    )
                )

    return issues
