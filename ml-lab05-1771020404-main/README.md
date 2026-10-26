# LAB 05 — K-Means: từ khoảng cách đến khám phá cấu trúc ẩn

**Học phần:** Nhập môn Học máy  
**Case study xuyên suốt:** DNU Learning Analytics Lab  
**Dataset:** Student Performance Factors  
**Thuật toán:** K-Means Clustering

## Câu hỏi trung tâm

> **Nếu không dùng nhãn `Needs_Support`, dữ liệu học tập có tự cho thấy những nhóm sinh viên có đặc điểm tương đồng hay không?**

Lab 05 chuyển từ **supervised learning** sang **unsupervised learning**.  
Ở các lab trước, mô hình học từ một nhãn để dự đoán. Trong lab này, K-Means **không nhìn thấy nhãn** mà chỉ dựa vào các đặc trưng số và khoảng cách để khám phá cấu trúc dữ liệu.

---

## Mục tiêu

Sau bài lab, sinh viên có thể:

1. Giải thích quy trình **khởi tạo centroid → gán cụm → cập nhật centroid → hội tụ**.
2. Hoàn thiện một phiên bản K-Means cơ bản **không dùng scikit-learn**.
3. Giải thích vì sao **scaling** quan trọng với thuật toán dựa trên khoảng cách.
4. Dùng **Elbow** và **Silhouette Score** để hỗ trợ lựa chọn số cụm K.
5. Xây dựng mô hình K-Means bằng `scikit-learn`.
6. Thực hiện **cluster profiling** để mô tả đặc điểm từng nhóm.
7. Phân biệt rõ **clustering** với **classification** khi đối chiếu cluster với `Needs_Support` sau huấn luyện.
8. Đưa ra diễn giải và khuyến nghị có căn cứ từ kết quả phân cụm.

---

## Cấu trúc repo

```text
ml-lab05-kmeans-starter/
│
├── README.md
├── Lab05_KMeans.ipynb
├── kmeans_manual.py
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md
│
├── scripts/
│   └── download_data.py
│
├── tests/
│   ├── check_lab05.py
│   └── test_kmeans_manual.py
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

Chạy notebook:

```bash
jupyter notebook
```

Mở:

```text
Lab05_KMeans.ipynb
```

---

## Quy tắc quan trọng

### 1. Mission 1: không dùng sklearn để cài K-Means

Hoàn thiện các hàm trong `kmeans_manual.py`:

```python
init_centroids(...)
assign_clusters(...)
update_centroids(...)
kmeans_manual(...)
```

Kiểm tra bằng:

```bash
python -m pytest -q tests/test_kmeans_manual.py
```

> Starter repo có thể **chưa pass unit tests**. Nhiệm vụ của bạn là làm các test chuyển sang màu xanh.

### 2. Không dùng `Needs_Support` hoặc `Exam_Score` làm feature phân cụm

Trong chuỗi case study, ta tạo nhãn giảng dạy:

```python
Needs_Support = 1 nếu Exam_Score < 65
```

Trong Lab 05:

- `Needs_Support` **không được dùng để huấn luyện K-Means**;
- `Exam_Score` cũng **không nằm trong feature set chính**;
- hai biến này chỉ được dùng **sau khi phân cụm** để hỗ trợ diễn giải.

> `Needs_Support` là nhãn giả lập phục vụ học tập, **không phải quy định chính thức của DNU** và không được dùng để ra quyết định thật về sinh viên.

### 3. Feature set chính

Lab sử dụng 5 đặc trưng số:

```python
feature_cols = [
    "Hours_Studied",
    "Attendance",
    "Previous_Scores",
    "Sleep_Hours",
    "Tutoring_Sessions",
]
```

Mục tiêu là tập trung vào K-Means và tác động của khoảng cách, thay vì trộn thêm bài toán mã hóa categorical variables.

### 4. Không mặc định K = 3

Sinh viên phải thử nhiều giá trị K, tối thiểu:

```text
K = 2, 3, 4, 5, 6, 7, 8
```

và dùng:

- **Inertia / Elbow**
- **Silhouette Score**
- **khả năng diễn giải trong bối cảnh giáo dục**

để giải thích lựa chọn cuối cùng.

### 4.1. Khởi tạo K-Means

Trong các phần dùng scikit-learn, lab sử dụng:

```python
KMeans(..., random_state=42, n_init=10)
```

`n_init=10` giúp giảm phụ thuộc vào một lần khởi tạo centroid duy nhất và làm kết quả ổn định hơn cho mục đích thực hành.

### 5. K-Means luôn trả về K cụm

K-Means sẽ tạo ra đúng K cluster ngay cả khi dữ liệu **không có cấu trúc cụm tự nhiên rõ ràng**.

Vì vậy, sau Elbow và Silhouette, bạn phải đánh giá thêm:

- các Silhouette Score có đủ cao để cho thấy sự tách biệt hay không;
- các kết quả có ổn định và có thể diễn giải hay không;
- có nên coi cluster là phân khúc thật hay chỉ là **exploratory partition**.

Nếu cấu trúc cụm yếu, không được dùng cluster để đưa ra quyết định hỗ trợ sinh viên.

### 6. Cluster ID không phải ý nghĩa

K-Means trả về:

```text
Cluster 0
Cluster 1
Cluster 2
...
```

Không được tự động hiểu các số này là “tốt”, “yếu”, “nguy cơ”.  
Tên mô tả chỉ được đặt **sau cluster profiling**.

### 7. Mỗi Mission phải có code + nhận xét

Không chỉ chạy code hoặc vẽ biểu đồ. Các câu hỏi Markdown là một phần của bài nộp.

---

## Các Mission

- **Mission 1 — K-Means from scratch:** hoàn thiện thuật toán trên dữ liệu 2D nhỏ.
- **Mission 2 — Back to DNU case study:** đọc Student Performance Factors và xác định feature set.
- **Mission 3 — Scaling matters:** so sánh phân cụm trước/sau StandardScaler.
- **Mission 4 — Choose K:** Elbow + Silhouette.
- **Mission 5 — Final K-Means model:** huấn luyện mô hình với K đã chọn.
- **Mission 6 — Cluster profiling:** mô tả và đặt tên trung tính cho các nhóm.
- **Mission 7 — Post-hoc analysis:** đối chiếu cluster với `Needs_Support` và `Exam_Score`.
- **Optional Challenge — PCA visualization:** chiếu dữ liệu xuống 2D để trực quan hóa.
- **Final — Model Reflection Card:** chốt điều đã học và giới hạn của kết quả.

---

## Nộp bài

Bài được xem là hoàn thành khi:

- `Lab05_KMeans.ipynb` chạy được từ đầu đến cuối;
- `kmeans_manual.py` hoàn thiện;
- public unit tests pass;
- Mission 1–7 có câu trả lời;
- có Final Reflection Card;
- bài được push lên branch `main`.

```bash
git add .
git commit -m "Complete Lab 05 K-Means"
git push
```

---

## GitHub Actions

Mỗi lần push lên `main`, Actions sẽ:

1. cài Python và dependencies;
2. tải Student Performance Factors;
3. kiểm tra cấu trúc repo/notebook;
4. chạy public unit tests cho K-Means from scratch;
5. chạy notebook trong môi trường sạch.

Actions kiểm tra lỗi kỹ thuật và cấu trúc.  
**Phần diễn giải cluster, lựa chọn K và lập luận của sinh viên vẫn cần giảng viên đánh giá.**
