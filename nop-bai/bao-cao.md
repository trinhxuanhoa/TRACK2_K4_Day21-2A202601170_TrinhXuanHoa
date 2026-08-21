# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Trịnh Xuân Hòa |
| MSSV | 2A202601170 |
| Lớp / Khóa | E403 |
| Repo GitHub | https://github.com/trinhxuanhoa/TRACK2_K4_Day21-2A202601170_TrinhXuanHoa |
| Ngày nộp | 21/08/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 200 | 0.10 | 5 | 0.7149 | 0.8740 |
| 2 | 100 | 0.10 | 3 | 0.7109 | 0.8780 |
| 3 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 4 | 10 | 0.01 | 1 | 0.0000 | 0.7520 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Bộ tham số này đạt `f1_score` cao nhất (0.7149 ở Bước 2 và 0.7354 ở Bước 3), vượt qua quality gate (ngưỡng 0.65). Độ sâu cây `max_depth=5` kết hợp với 200 estimators cho phép mô hình học được các tương tác phi tuyến tính phức tạp giữa các thuộc tính nhân khẩu học. Ta nhận thấy sự đánh đổi: nếu `learning_rate` quá nhỏ (0.01) và số estimators thấp (10), mô hình bị underfitting nặng nề với F1=0 dù accuracy đạt 75.2% (do thiên lệch đoán toàn bộ là lớp đa số).

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Adult Census có sự mất cân bằng lớp rõ rệt với tỷ lệ xấp xỉ 75% nhãn thu nhập thấp (<=50K) và chỉ 25% nhãn thu nhập cao (>50K). Một mô hình ngây thơ (naive) chỉ cần dự đoán tất cả các mẫu là thu nhập thấp đã dễ dàng đạt độ chính xác (accuracy) 75.2% mà thực chất không học được bất kỳ thông tin nào và hoàn toàn vô dụng trong nghiệp vụ. 

Do đó, ngưỡng chất lượng bắt buộc phải đặt trên F1-score của lớp dương (target = 1) — trung bình điều hòa giữa Precision và Recall của nhóm thu nhập cao. F1 đo lường chính xác năng lực phát hiện lớp thiểu số quan trọng này mà không bị phóng đại bởi lớp đa số. Ta không sử dụng `average="weighted"` hay `average="macro"` vì các phương pháp này gộp cả lớp âm (chiếm 75%), làm loãng và che lấp sự yếu kém khi nhận diện lớp dương.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Xác thực DVC và Cloud Storage tự động trên CI runner | CI runner môi trường sạch chưa có sẵn credentials truy cập S3/GCS bucket | Sử dụng GitHub Secret `STORAGE_CREDENTIALS` và viết script parse JSON động trong workflow để export biến môi trường AWS/GCP |
| Thiết lập Quality Gate chặn deploy khi model chưa đạt chuẩn | Cần truyền metric F1 từ job Train sang job Quality Gate để kiểm tra điều kiện | Sử dụng GITHUB_OUTPUT trong step train để xuất biến `f1` và validate điều kiện `f1 >= 0.65` trước khi kích hoạt job Release |
| Đảm bảo server FastAPI nhận diện đúng schema dữ liệu inference | Model scikit-learn yêu cầu mảng đặc trưng số đầu vào đúng 10 chiều | Xây dựng model Pydantic `ScoreRequest` và kiểm tra độ dài vector đặc trưng trước khi gọi `model.predict()` |

---

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1` - 22.361 mẫu) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2` - 44.722 mẫu) | 0.7354 | 0.8820 |

**Nhận xét:** Khi bổ sung thêm 22.361 mẫu dữ liệu mới, `f1_score` tăng nhẹ từ 0.7149 lên 0.7354 (+0.0205) và `accuracy` tăng từ 0.8740 lên 0.8820. Điều này cho thấy tập dữ liệu lớn hơn giúp mô hình học thêm các biến thể biên và tổng quát hóa tốt hơn. Quan trọng nhất, quy trình tự động hóa ở Bước 3 đã hoàn toàn thành công: chỉ với một commit DVC dữ liệu mới, toàn bộ pipeline CI/CD đã tự động kích hoạt, kiểm tra chất lượng và tái triển khai mô hình lên VM mà không cần can thiệp thủ công.

---

## 5. Báo Cáo Thách Thức Nâng Cao (Bonus 1 - 5)

- **Bonus 1 (MLflow Remote Tracking)**: Cập nhật `src/train.py` và `.github/workflows/cicd.yml` tự động nhận diện `MLFLOW_TRACKING_URI` và credentials qua GitHub Secrets để ghi nhận thí nghiệm lên Remote Server / DagsHub.
- **Bonus 2 (Decision Threshold Tuning)**: Quét ngưỡng xác suất từ 0.1 đến 0.9 (bước 0.05). Tại ngưỡng mặc định 0.50 đạt F1 = `0.7354`, trong khi ngưỡng tối ưu **0.30** nâng F1 lên **0.7537** (tăng mạnh Recall cho lớp thiểu số).
- **Bonus 3 (Báo Cáo Chi Tiết & Phân Tích Sai Lầm)**: Tự động xuất ma trận nhầm lẫn và Precision/Recall vào `outputs/detail.txt`, lưu thành GitHub Actions artifact. *Phân tích:* Bỏ sót người thu nhập cao (**FN - Recall thấp**) tốn kém hơn nhiều so với gán nhầm (**FP - Precision thấp**) do mất đi cơ hội kinh doanh/doanh thu từ nhóm khách hàng giá trị cao.
- **Bonus 4 (Model Rollback An Toàn)**: Thêm cơ chế trong CI/CD tự động kéo `report.json` cũ từ S3/GCS để so sánh. Nếu model mới có `F1_new < F1_old`, pipeline tự động hủy upload và dừng triển khai.
- **Bonus 5 (Cảnh Báo Lệch Dữ Liệu - Data Drift)**: Đo tỷ lệ lớp dương tập train (thực tế `24.78%` so với tham chiếu `24.8%`), tự động phát cảnh báo khi độ lệch vượt 5% và lưu vào `outputs/report.json`.


