# LAB 03 — Naive Bayes: từ bằng chứng đến xác suất

**Học phần:** Nhập môn Học máy  
**Case study xuyên suốt:** DNU Learning Analytics Lab  
**Dataset:** Student Performance Factors  
**Thuật toán chính:** Gaussian Naive Bayes

## Câu hỏi trung tâm

> **Nếu K-NN dự đoán bằng những sinh viên “gần” nhất, Naive Bayes sẽ dự đoán cùng một sinh viên như thế nào bằng xác suất?**

Lab 03 tiếp tục đúng bài toán của Lab 02 để so sánh **hai cách suy luận trên cùng dữ liệu**:

- K-NN: khoảng cách → hàng xóm → bỏ phiếu;
- Naive Bayes: prior → likelihood → posterior.

Nhãn giảng dạy vẫn là:

```python
Needs_Support = 1 nếu Exam_Score < 65
```

> `Needs_Support` chỉ là nhãn giả lập phục vụ học tập, **không phải quy định chính thức của DNU** và không được dùng để ra quyết định thật về sinh viên.

---

## Mục tiêu

Sau bài lab, sinh viên có thể:

1. Giải thích prior, likelihood và posterior trong một bài toán phân loại.
2. Tính prior của lớp từ dữ liệu huấn luyện.
3. Giải thích GaussianNB học mean và variance của từng feature theo từng lớp.
4. Xây dựng mô hình `GaussianNB` bằng `scikit-learn`.
5. Dùng `predict_proba()` để đọc xác suất dự đoán của một mẫu.
6. Đánh giá mô hình bằng accuracy, precision, recall, F1 và confusion matrix.
7. Phân tích trường hợp **bỏ sót sinh viên cần hỗ trợ**.
8. So sánh GaussianNB và K-NN trên **cùng train/test split**.
9. Giải thích vì sao dữ liệu đếm văn bản phù hợp với `MultinomialNB` hơn `GaussianNB`.

---

## Thời lượng gợi ý

**Trên lớp (2 tiết):** Mission 1–6.  
**Sau lớp / mở rộng:** Transfer Challenge + Final Reflection.

---

## Cấu trúc repo

```text
ml-lab03-naive-bayes-starter/
│
├── README.md
├── Lab03_NaiveBayes.ipynb
├── nb_manual.py
├── requirements.txt
│
├── data/
│   └── .gitkeep
│
├── scripts/
│   └── download_data.py
│
├── tests/
│   ├── check_lab03.py
│   └── test_nb_manual.py
│
└── .github/
    └── workflows/
        └── lab-check.yml
```

---

## Chuẩn bị môi trường

```bash
conda create --name machine_learning python=3.12
conda activate machine_learning
pip install -r requirements.txt
python scripts/download_data.py
jupyter notebook
```

Mở `Lab03_NaiveBayes.ipynb`.

---

## Quy tắc quan trọng

### 1. Giữ nguyên bài toán của Lab 02

```python
Needs_Support = 1 nếu Exam_Score < 65
```

Feature chính:

```python
feature_cols = [
    "Hours_Studied",
    "Attendance",
    "Previous_Scores",
    "Sleep_Hours",
]
```

Dùng `test_size=0.25`, `random_state=42`, `stratify=y` để so sánh với Lab 02.

### 2. Không dùng `Exam_Score` làm feature

Nhãn được tạo trực tiếp từ `Exam_Score`; đưa nó vào `X` sẽ gây **target leakage**.

### 3. Không dùng `GridSearchCV`

Mục tiêu là hiểu Naive Bayes, không phải tối ưu tự động.

### 4. Không dùng TF-IDF trong transfer task

Dùng đúng pipeline:

```text
CountVectorizer → MultinomialNB
```

### 5. Không tự cài toàn bộ GaussianNB from scratch

Chỉ hoàn thiện hai hàm nhỏ trong `nb_manual.py`:

- `compute_class_priors(y)`;
- `gaussian_log_likelihood(x, mean, var)`.

Sau đó dùng `sklearn.naive_bayes.GaussianNB` cho mô hình đầy đủ.

### 6. Mỗi Mission phải có code + nhận xét

Không chỉ chạy code và chép con số; cần giải thích kết quả bằng Markdown.

---

## Các Mission

- **Mission 1 — Back to DNU case study:** tạo `Needs_Support`, chọn feature, chia train/test.
- **Mission 2 — Prior first:** tự tính prior của lớp.
- **Mission 3 — Gaussian evidence:** hiểu `class_prior_`, `theta_`, `var_` và Gaussian likelihood.
- **Mission 4 — Evaluate beyond Accuracy:** accuracy, precision, recall, F1, confusion matrix.
- **Mission 5 — Explain one student:** dùng `predict_proba()` và log-score để giải thích một dự đoán.
- **Mission 6 — K-NN vs Naive Bayes:** so sánh hai model trên cùng test set và tìm các mẫu bất đồng.
- **Transfer Challenge — 20 Newsgroups:** `CountVectorizer → MultinomialNB`.
- **Final — Model Reflection Card:** tổng kết cách Naive Bayes ra quyết định.

---

## Kiểm tra trước khi nộp

```bash
python scripts/download_data.py
python tests/check_lab03.py
python -m pytest -q tests/test_nb_manual.py
```

Notebook cần chạy được từ đầu đến cuối.

---

## Nộp bài

```bash
git add .
git commit -m "Complete Lab 03 Naive Bayes"
git push
```

Bài được xem là hoàn thành khi:

- `Lab03_NaiveBayes.ipynb` chạy được;
- `nb_manual.py` hoàn thiện;
- public unit tests pass;
- Mission 1–6 có câu trả lời;
- có Final Reflection;
- bài được push lên branch `main`.

---

## GitHub Actions

Mỗi lần push lên `main`, Actions sẽ:

1. cài Python và dependencies;
2. tải Student Performance Factors dataset;
3. kiểm tra cấu trúc repo/notebook;
4. chạy notebook trong môi trường sạch;
5. chạy public unit tests cho prior và Gaussian likelihood.

Actions giúp phát hiện lỗi kỹ thuật; phần giải thích và lập luận vẫn cần giảng viên đánh giá.
