# Báo cáo Day 09 — Phạm Trung Thành, 2A202604050

## 1. Kết luận trước
Pipeline kết luận **Hold** (mã thoát 10), xuất bản nháp vào `submission/export/clean_v1/` để thử nghiệm train và chưa phát hành chính thức. Lô đầu vào có 224 ảnh, qua bước cửa vào loại 21 ảnh (ảnh hỏng, nhỏ, trùng và rò rỉ tập test), giữ lại 203/224 ảnh. Sau khi `autofix` xử lý xong các lỗi cú pháp và người duyệt mô phỏng xem hết ngân sách 80/125 ảnh trong hàng đợi, lỗi bắt buộc còn 0 nhưng vẫn còn 38/203 ảnh (18.72% >= ngưỡng 10%) mang cảnh báo chưa được duyệt nên kết luận dừng ở Hold.

## 2. Lỗi và phân việc
Tổng hợp từ `01_intake.csv`, `02_error_list_v1.txt`, `03_fixes.csv` và `06_review_log.csv`:

| Mã lỗi | Số lượng | Ai sửa | Lý do |
|---|---:|---|---|
| `CORRUPT_IMAGE` | 2 / 224 ảnh | Máy (`remove_image`) | Ảnh `img_0010`, `img_0136` không mở được, không thể dùng để train. |
| `BAD_SIZE` | 3 / 224 ảnh | Máy (`remove_image`) | Cạnh ngắn dưới 250px, không đạt độ phân giải tối thiểu. |
| `DUP_IMAGE_EXACT` | 8 / 224 ảnh | Máy (`remove_image`) | Trùng từng byte (`md5`), giữ lại ảnh đầu và bỏ bản lặp. |
| `DUP_IMAGE_NEAR` | 3 / 224 ảnh | Máy (`remove_image`) | Trùng gần đúng với `phash <= 8` (`img_0076`, `img_0143`, `img_0223`). |
| `LEAK_TEST` | 5 / 224 ảnh | Máy (`remove_image`) | Trùng ảnh trong `data/test/` (`test_009`, `test_033`, `test_039`, `test_044`, `test_085`), để lại sẽ làm lệch kết quả đánh giá. |
| `BAD_CLASS_NAME` | 2 lỗi / `data.yaml` | Máy (`rename_classes`) | Sai viết hoa (`Person`, `Bus`), chuẩn hoá trực tiếp theo `cfg["classes"]`. |
| `BAD_CLASS_ID` | 6 dòng / 203 ảnh | Máy (`drop_line`) | ID lớp ngoài 0..3 (`93`, `-1`, `7`, `4`), không tự suy đoán được lớp đúng. |
| `BAD_FORMAT` | 3 dòng / 203 ảnh | Máy (`drop_line`) | Dòng thiếu cột hoặc có `nan`, không khôi phục được toạ độ. |
| `BOX_ZERO` | 4 dòng / 203 ảnh | Máy (`drop_line`) | Box có chiều rộng hoặc cao bằng 0. |
| `DUP_BOX` | 7 dòng / 203 ảnh | Máy (`drop_line`) | Hai box cùng lớp chồng gần khít (`IoU > 0.9`), xoá dòng lặp phía sau. |
| `BOX_OUT_OF_IMAGE` | 8 dòng / 203 ảnh | Máy (`clip_box`: 7, `drop_line`: 1) | Cắt các mép tràn nhẹ về đoạn `[0, 1]` (1 dòng vừa tràn vừa trùng box nên bị `drop_line`). |
| `MISSING_LABEL_FILE` | 6 / 203 ảnh | Người (`needs_label`) | Máy không được tự bịa nhãn, chuyển lên đầu hàng đợi cho người gán. |
| `POSSIBLE_MISSING` | 53 dòng / 203 ảnh | Người (hàng đợi) | Model kiểm tra thấy vật nhưng file nhãn không có box, cần người xác nhận. |
| `POSSIBLE_FALSE_POS` | 29 dòng / 203 ảnh | Người (hàng đợi) | Box nhãn không khớp dự đoán nào của model (`conf >= 0.1`), tránh xoá nhầm vật nhỏ. |
| `POSSIBLE_LOOSE_BOX` | 28 dòng / 203 ảnh | Người (hàng đợi) | Đúng lớp nhưng `IoU < 0.5`, cần người căn lại mép box. |
| `POSSIBLE_WRONG_CLASS` | 18 dòng / 203 ảnh | Người (hàng đợi) | Khớp vị trí (`IoU >= 0.5`) nhưng lệch lớp giữa `car`, `person`, `truck`, `bus`. |
| `EMPTY_LABEL`, `BOX_TINY`, `BOX_HUGE`, `CLASS_RARE` | 12 / 6 / 3 / 4 dòng | Người (hàng đợi) | Cần người kiểm tra ảnh thực tế để phân biệt ảnh nền thật với thiếu nhãn hoặc vật quá nhỏ/lớn. |

**Thứ tự ưu tiên của `05_review_queue.csv` và ngân sách 80 ảnh:**
Hàng đợi gồm 125 ảnh được xếp theo tổng trọng số rủi ro của các mã lỗi nhân với độ tự tin. Đứng đầu là `img_0020` (điểm 11.524 vì vừa bị `FIXED_DROP_LINE`, vừa có `POSSIBLE_MISSING` và `POSSIBLE_WRONG_CLASS`), tiếp theo là 6 ảnh `NEEDS_LABEL` (điểm 10.0). Với ngân sách 80 ảnh, người duyệt đã xử lý hết cả 6 ảnh thiếu file nhãn cùng phần lớn các ảnh `POSSIBLE_WRONG_CLASS` và ảnh nhiều cảnh báo gộp lại, còn 45 ảnh cuối hàng đợi chưa tới lượt xem.

## 3. Calibration (2 ngưỡng)

| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước -> sau) | Lý do (đánh đổi) |
|---|---:|---:|---|---|
| `duplicates.phash_dist` | 2 | 8 | `DUP_IMAGE_EXACT`: 1.00 / 1.00 -> 1.00 / 1.00<br>`DUP_IMAGE_NEAR`: 1.00 / 0.50 -> 1.00 / 0.67 | Thao tác `remove_image` xoá thẳng ảnh nên ưu tiên precision = 1.00. Chọn 8 giúp tăng recall ảnh gần trùng lên 0.67 mà vẫn an toàn. Mở xem `work/viz_dup/` thấy `img_0077` và `img_0223` (`phash = 6`, cùng cảnh tháp đồng hồ và xe buýt nhưng lệch sáng, crop nhẹ) cùng `img_0042` và `img_0076` (`phash = 0`, cùng xe buýt `103 ROJA` ban đêm) đều đúng là ảnh trùng. |
| `second_opinion.conf` | 0.25 | 0.6 | `POSSIBLE_MISSING`: 0.50 / 1.00 -> 0.57 / 1.00<br>`POSSIBLE_WRONG_CLASS`: 0.56 / 0.83 -> 0.62 / 0.83<br>`POSSIBLE_LOOSE_BOX`: 0.29 / 0.40 -> 0.20 / 0.20<br>`POSSIBLE_FALSE_POS`: 0.31 / 1.00 -> 0.31 / 1.00 | Nâng ngưỡng lên 0.6 giúp giảm bớt báo động giả cho hàng đợi 80 ảnh nhưng vẫn giữ nguyên recall cho `MISSING` (1.00) và `WRONG_CLASS` (0.83). Nếu tăng lên 0.8 thì recall của `MISSING` tụt xuống 0.75. |

**Điểm calibration có thể đánh lừa:**
Tập `data/calib/` chỉ gồm 39 ảnh nên mỗi mẫu chèn làm precision/recall nhảy 17–50%, dễ gây chọn nhầm ngưỡng cực đoan nếu chỉ chạy theo F1:
- Với `phash_dist`, mức 16 đạt F1 = 0.91 trên `calib`, nhưng khi tăng lên 20 thì recall của `DUP_IMAGE_EXACT` tụt xuống còn 0.33 do ảnh trùng bị ghép nhầm sang kho `test/gold_val` thành `LEAK_TEST` trước khi kiểm trùng nội bộ.
- Với `second_opinion.conf`, mức 0.8 đạt F1 = 0.91 cho `WRONG_CLASS` trên `calib` (chủ yếu là 6 ca `person`/`car`), nhưng trên lô `raw` có nhiều cặp dễ nhầm hơn như `car`/`truck`/`bus` và đặt `conf = 0.8` sẽ bỏ sót 25% lỗi thiếu nhãn.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
- **(a) Chứng minh gate chặn Rework:** Chạy `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb` (không có người gán 6 ảnh `MISSING_LABEL_FILE`). Pipeline kết luận **Rework** với lý do còn 6 lỗi bắt buộc chưa xử lý, trả về mã thoát **20** và không tạo thư mục xuất dataset (`Test-Path work/exp_nb` ra `False`).
- **(b) Điểm mù pipeline không bắt được:**
  1. **Box lỏng nhẹ (`0.5 <= IoU < 0.75`) hoặc gộp nhiều người đứng sát nhau:** `second_opinion` chỉ cảnh báo `POSSIBLE_LOOSE_BOX` khi `IoU < 0.5`. Xem bằng mắt ở `img_0077.png` (`work/viz_dup/`) thấy hai người đi bộ sát nhau ở góc phải dưới bị gộp chung vào một box `person` nhưng không bị báo lỗi; tương tự ở `img_0001.png` (`work/viz/`) có box `person` bao hơi rộng phần tường phía trên và bên trái.
  2. **Lỗi mà cả nhãn gốc lẫn model `yolo26l` đều sai hoặc cùng bỏ sót:** Vì `second_opinion` dựa vào `yolo26l` để đối chiếu, nếu cả hai cùng bỏ qua xe đỗ khuất góc (như góc phải của `img_0004.png` đang có nhãn rỗng) hoặc cùng gán nhầm một chiếc SUV/bán tải sang `car`, pipeline sẽ không phát hiện ra.
  3. **Ảnh trùng biến đổi mạnh vượt `phash_dist = 8`:** Trên tập `calib`, 2/6 ảnh `DUP_IMAGE_NEAR` bị biến đổi mạnh vẫn lọt qua ngưỡng 8.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
Thí nghiệm chạy trên Google Colab T4 với cùng cấu hình: model `yolo26n.pt`, 20 epoch, `batch=16`, `imgsz=640`, `val` là `data/gold_val` (40 ảnh), `test` là `data/test` (100 ảnh), lặp lại 2 seed (`0`, `1`).

| biến thể | #ảnh train | số seed | map50 (TB ± độ lệch) | map50_95 (TB ± độ lệch) | precision (TB ± độ lệch) | recall (TB ± độ lệch) |
|---|---|---|---|---|---|---|
| raw | 224 | 2 | 0.726 ± 0.000 | 0.512 ± 0.004 | 0.764 ± 0.061 | 0.649 ± 0.028 |
| clean | 203 | 2 | 0.699 ± 0.025 | 0.486 ± 0.013 | 0.796 ± 0.050 | 0.596 ± 0.049 |

Chênh `mAP50 (clean − raw) = -0.027`.

**So sánh với nhiễu giữa các seed và kết luận trung thực:**
Độ lệch chuẩn giữa 2 seed trên `clean` ở `mAP50` là `±0.025` (seed 0 đạt `0.6813`, seed 1 đạt `0.7163`) và ở `recall` lên tới `±0.049`. Mức chênh `-0.027` gần như nằm trọn trong biên độ dao động ngẫu nhiên giữa các seed, trong khi `precision` của `clean` cao hơn (`0.796` so với `0.764`) còn `recall` thấp hơn (`0.596` so với `0.649`). Với số lượng ảnh nhỏ và mới chạy 2 seed trên bản nháp Hold, không thể kết luận `clean` tốt hơn `raw` trên tập test này (thậm chí `mAP50` trung bình đang thấp hơn nhẹ).

**Hai nguyên nhân có thể giải thích và cách kiểm:**
1. *Rò rỉ tập test (`LEAK_TEST`) và trùng lặp làm điểm `raw` bị thổi phồng giả:* Pipeline đã phát hiện 5 ảnh trong `raw` trùng với `data/test/` (`test_009`, `test_033`, `test_039`, `test_044`, `test_085`, chiếm 5% tập test) cùng 11 ảnh trùng nội bộ. Khi train trên `raw`, model được nhìn trước 5 ảnh này nên điểm trên `test` được cộng lợi thế giả, còn `clean` đã loại bỏ chúng. *Cách kiểm:* tách riêng 5 ảnh rò rỉ khỏi `data/test/` và đo lại `mAP50` trên 95 ảnh test không bị rò rỉ.
2. *`clean` mới dừng ở bản nháp Hold và lớp `truck` quá ít mẫu:* Theo `class_stats.csv`, lớp `truck` chỉ có 41 box, nên việc giảm từ 224 xuống 203 ảnh kèm 45 ảnh cuối hàng đợi chưa duyệt (còn 13 `POSSIBLE_MISSING` và 20 `POSSIBLE_LOOSE_BOX` trong `07_error_list_v3.txt`) làm `AP50` của riêng `truck` ở `clean` tụt mạnh và dao động lớn (`0.4841` ở seed 0, `0.5876` ở seed 1 so với `0.6574` / `0.6395` của `raw`), trong khi `car` (`0.741` vs `0.738`), `person` (`0.824` vs `0.822`) và `bus` (`0.695` vs `0.695`) gần như ngang nhau. *Cách kiểm:* duyệt nốt 45 ảnh còn lại trong hàng đợi (đưa về Release) và chạy thêm biến thể `oracle` (nhãn người chuẩn) cùng ≥ 3 seed để tách riêng tác động của lỗi nhãn còn sót và nhiễu ngẫu nhiên.

**Việc tiếp theo:** Hoàn tất duyệt 45 ảnh còn lại để đạt Release trước khi dùng `clean_v1` cho các vòng train chính thức, đồng thời đánh giá trên tập test hoàn toàn không trùng lặp với `raw`.

## 6. Quyết định và ghi chú AI
- **(a) Quyết định cuối:** **Hold**, chỉ giữ `clean_v1` làm bản nháp để chạy thí nghiệm so sánh, chưa phát hành chính thức. Để chuyển sang **Release**, cần tăng ngân sách duyệt thêm ít nhất 18 ảnh trong số 45 ảnh còn lại ở cuối `05_review_queue.csv` (kéo tỉ lệ ảnh còn cảnh báo chưa duyệt từ 38/203 = 18.72% xuống dưới 10%, tức dưới 21 ảnh) hoặc xử lý thủ công các ca còn lại trong `07_error_list_v3.txt`.
- **(b) Ghi chú sử dụng AI:**
  - *Prompt chính:* Mô tả rõ 4 phần cho từng module gồm dữ liệu vào/ra qua `Dataset`, các mã lỗi và ngưỡng lấy từ `cfg`, thứ tự xử lý ca biên và yêu cầu giữ nguyên số dòng gốc cũng như không chạm tới `data/raw`.
  - *Cách kiểm:* Đọc lại code từng module, chạy đủ 79 unit/e2e test (`python -m pytest -q`), chạy `calibrate` kiểm tra precision/recall trên `calib`, mở xem ảnh ở `work/viz/` và `work/viz_dup/`, và chạy thử `--budget 0` để xác nhận gate chặn Rework.
  - *Điểm cần tự rà soát chỉnh lại:* Ở `autofix.py`, khi 1 dòng vừa tràn ảnh (`BOX_OUT_OF_IMAGE`) vừa trùng box (`DUP_BOX`), phải gom thao tác theo dòng gốc để `drop_line` thắng `clip_box` và ghi file lại một lần duy nhất, tránh sửa tuần tự làm lệch số dòng phía sau.
