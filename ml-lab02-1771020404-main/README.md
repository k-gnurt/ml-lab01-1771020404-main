# LAB 02 — K-Nearest Neighbors: từ khoảng cách đến quyết định

**Học phần:** Nhập môn Học máy  
**Case study xuyên suốt:** DNU Learning Analytics Lab  
**Dataset:** Student Performance Factors  
**Thuật toán:** K-Nearest Neighbors (K-NN)

## Câu hỏi trung tâm

> **KNN đưa ra dự đoán dựa trên “hàng xóm” như thế nào, và điều gì làm thay đổi những hàng xóm đó?**

Bài lab được thiết kế theo ba tầng:

1. **KNN tổng quát** trên các điểm 2D và nhãn A/B;
2. **KNN from scratch** để hiểu thật sự thuật toán làm gì;
3. **Quay lại case study Student Performance** để xây dựng mô hình `Needs_Support` và đánh giá tác động của scaling, K và distance metric.

Mục tiêu là hiểu KNN như một thuật toán tổng quát, không đồng nhất KNN với riêng bài toán Student Performance.

---

## Mục tiêu

Sau bài lab, sinh viên có thể:

1. Tính và giải thích Euclidean distance và Manhattan distance.
2. Tự cài đặt một phiên bản KNN classification cơ bản mà **không dùng sklearn**.
3. Giải thích các bước: tính khoảng cách → chọn K hàng xóm → bỏ phiếu → dự đoán.
4. Xây dựng mô hình KNN bằng `scikit-learn` cho một bài toán phân loại.
5. Giải thích vì sao chuẩn hóa đặc trưng có ảnh hưởng mạnh đến KNN.
6. So sánh một số giá trị K và hai distance metrics bằng vòng lặp thủ công.
7. Đọc confusion matrix, accuracy, precision và recall trong bối cảnh `Needs_Support`.
8. Chuyển quy trình KNN sang một dataset khác.

---

## Thời lượng gợi ý

**Trên lớp (2 tiết):** Mission 1–6.  
**Sau lớp / mở rộng:** Transfer Challenge + Final Reflection.

---

## Cấu trúc repo

```text
ml-lab02-knn-starter/
│
├── README.md
├── Lab02_KNN.ipynb
├── knn_manual.py
├── requirements.txt
│
├── data/
│   └── .gitkeep
│
├── scripts/
│   └── download_data.py
│
├── tests/
│   ├── check_lab02.py
│   └── test_knn_manual.py
│
└── .github/
    └── workflows/
        └── lab-check.yml
```

---

## Chuẩn bị môi trường

Khuyến nghị Python 3.12.

```bash
conda create --name machine_learning python=3.12
conda activate machine_learning
pip install -r requirements.txt
```

Tải dataset:

```bash
python scripts/download_data.py
```

Khởi động notebook:

```bash
jupyter notebook
```

Mở:

```text
Lab02_KNN.ipynb
```

---

## Quy tắc quan trọng

### 1. Mission 1–2: không dùng sklearn để làm KNN

Bạn phải hoàn thiện các hàm trong `knn_manual.py`:

```python
euclidean_distance(...)
manhattan_distance(...)
get_k_neighbors(...)
knn_predict_one(...)
```

Sau đó kiểm tra bằng:

```bash
python -m pytest -q tests/test_knn_manual.py
```

> Starter repo có thể **chưa pass unit tests**. Mục tiêu của bạn là làm các test chuyển sang màu xanh.

### 2. Không dùng `GridSearchCV`

Lab này yêu cầu thử K và distance metric bằng vòng lặp thủ công để quan sát tác động của từng lựa chọn.

### 3. Không dùng `Exam_Score` làm feature

Trong lab, chúng ta tạo một nhãn giảng dạy:

```python
Needs_Support = 1 nếu Exam_Score < 65
```

Do nhãn này được tạo trực tiếp từ `Exam_Score`, đưa `Exam_Score` vào feature sẽ gây **target leakage**.

> `Needs_Support` trong lab chỉ là nhãn giả lập phục vụ học tập, **không phải quy định chính thức của DNU** và không nên được dùng để ra quyết định thật về sinh viên.

### 4. Scaling phải fit trên train set

Đúng:

```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Không fit scaler trên toàn bộ dataset trước khi chia train/test.

### 5. Mỗi Mission cần có code + nhận xét

Không chỉ đưa ra con số hoặc biểu đồ. Hãy giải thích kết quả bằng Markdown.

---

## Các Mission

- **Mission 1 — Distance first:** Euclidean vs Manhattan.
- **Mission 2 — Build KNN from scratch:** tìm hàng xóm và majority vote trên bài toán A/B tổng quát.
- **Mission 3 — Back to DNU case study:** tạo `Needs_Support`, chọn features, train/test split.
- **Mission 4 — Baseline KNN:** mô hình sklearn chưa scaling.
- **Mission 5 — Scaling matters:** so sánh trước/sau `StandardScaler`.
- **Mission 6 — K and metric matter:** thử nhiều K và Euclidean/Manhattan bằng loop.
- **Transfer Challenge — Iris:** áp dụng quy trình sang một bài toán khác.
- **Final — Model Reflection Card:** chốt những điều làm KNN thay đổi quyết định.

---

## Nộp bài

Bài được xem là hoàn thành khi:

- `Lab02_KNN.ipynb` chạy được từ đầu đến cuối;
- `knn_manual.py` hoàn thiện;
- public unit tests pass;
- Mission 1–6 có câu trả lời;
- có Final Reflection;
- bài được push lên branch `main`.

```bash
git add .
git commit -m "Complete Lab 02 KNN"
git push
```

---

## GitHub Actions

Mỗi lần push lên `main`, Actions sẽ:

1. cài Python và dependencies;
2. tải Student Performance Factors dataset;
3. kiểm tra cấu trúc repo/notebook;
4. chạy notebook trong môi trường sạch;
5. chạy public unit tests cho phần KNN from scratch.

Actions giúp phát hiện lỗi kỹ thuật; phần giải thích, lựa chọn mô hình và lập luận vẫn cần giảng viên đánh giá.
