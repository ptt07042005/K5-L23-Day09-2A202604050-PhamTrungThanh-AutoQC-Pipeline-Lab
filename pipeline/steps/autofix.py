"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401


REMOVE_CODES = {"DUP_IMAGE_EXACT", "DUP_IMAGE_NEAR", "LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}
DROP_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}


def fix(ds, issues, cfg):
    fixes = []
    removed_stems = set()
    needs_label_seen = set()
    renamed_yaml = False
    line_ops = defaultdict(dict)  # {stem: {line_no: "drop" | "clip"}}

    for iss in issues:
        if iss.severity != ERROR:
            continue

        code = iss.code
        stem = iss.image

        if code in REMOVE_CODES:
            if stem not in removed_stems:
                removed_stems.add(stem)
                img_p = ds.image_path(stem)
                lbl_p = ds.label_path(stem)
                before_name = img_p.name if img_p else stem
                if img_p and img_p.exists():
                    img_p.unlink()
                if lbl_p.exists():
                    lbl_p.unlink()
                fixes.append({
                    "image": stem,
                    "line": "",
                    "action": "remove_image",
                    "before": before_name,
                    "after": "(đã xoá)",
                })

        elif code == "BAD_CLASS_NAME":
            if not renamed_yaml:
                renamed_yaml = True
                data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) if ds.yaml_path.exists() else {}
                data = data or {}
                before = str(data.get("names", ""))
                data["names"] = list(cfg["classes"])
                data["nc"] = len(cfg["classes"])
                ds.yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
                fixes.append({
                    "image": "data.yaml",
                    "line": "",
                    "action": "rename_classes",
                    "before": before,
                    "after": str(data["names"]),
                })

        elif code == "MISSING_LABEL_FILE":
            if stem not in needs_label_seen:
                needs_label_seen.add(stem)
                fixes.append({
                    "image": stem,
                    "line": "",
                    "action": "needs_label",
                    "before": "(không có file)",
                    "after": "chuyển người gán",
                })

        elif code in DROP_CODES and iss.line is not None:
            line_ops[stem][iss.line] = "drop"

        elif code == "BOX_OUT_OF_IMAGE" and iss.line is not None:
            # drop_line luôn ưu tiên hơn clip_box nếu cùng một dòng có 2 lỗi
            if line_ops[stem].get(iss.line) != "drop":
                line_ops[stem][iss.line] = "clip"

    # Áp dụng sửa/xoá theo từng file nhãn (ghi lại 1 lần mỗi file)
    for stem, ops in sorted(line_ops.items()):
        if stem in removed_stems:
            continue
        lp = ds.label_path(stem)
        if not lp.exists():
            continue

        raw_lines = lp.read_text(encoding="utf-8", errors="replace").splitlines()
        new_lines = []
        changed = False

        for idx, raw_text in enumerate(raw_lines, 1):
            op = ops.get(idx)
            if op is None:
                new_lines.append(raw_text)
            elif op == "drop":
                changed = True
                fixes.append({
                    "image": stem,
                    "line": idx,
                    "action": "drop_line",
                    "before": raw_text,
                    "after": "(đã xoá)",
                })
            elif op == "clip":
                changed = True
                try:
                    cls_id, cx, cy, w, h = parse_yolo_line(raw_text)
                except ValueError:
                    fixes.append({
                        "image": stem,
                        "line": idx,
                        "action": "drop_line",
                        "before": raw_text,
                        "after": "(đã xoá: lỗi định dạng)",
                    })
                    continue

                x1 = max(0.0, min(1.0, cx - w / 2))
                y1 = max(0.0, min(1.0, cy - h / 2))
                x2 = max(0.0, min(1.0, cx + w / 2))
                y2 = max(0.0, min(1.0, cy + h / 2))
                nw, nh = x2 - x1, y2 - y1

                if nw <= 0 or nh <= 0:
                    fixes.append({
                        "image": stem,
                        "line": idx,
                        "action": "drop_line",
                        "before": raw_text,
                        "after": "(đã xoá: ngoài ảnh)",
                    })
                else:
                    ncx, ncy = (x1 + x2) / 2, (y1 + y2) / 2
                    clipped = f"{cls_id} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}"
                    new_lines.append(clipped)
                    fixes.append({
                        "image": stem,
                        "line": idx,
                        "action": "clip_box",
                        "before": raw_text,
                        "after": clipped,
                    })

        if changed:
            content = "\n".join(new_lines) + ("\n" if new_lines else "")
            lp.write_text(content, encoding="utf-8")

    return fixes
