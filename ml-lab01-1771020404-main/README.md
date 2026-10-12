# LAB 01 — Khám phá và chuẩn bị dữ liệu cho học máy

**Học phần:** Nhập môn Học máy  
**Case study xuyên suốt:** DNU Learning Analytics Lab  
**Dataset:** Student Performance Factors

## Bối cảnh

DNU Learning Analytics Lab nhận được một bộ dữ liệu mô tả các yếu tố liên quan đến kết quả học tập của sinh viên.

Trước khi xây dựng bất kỳ mô hình học máy nào, nhóm cần trả lời câu hỏi:

> **Dataset này có đủ hiểu, đủ sạch và đủ phù hợp để bắt đầu modeling chưa?**

Trong Lab 01, bạn sẽ làm việc như một thành viên của nhóm phân tích dữ liệu: hiểu cấu trúc dữ liệu, xác định feature/target, kiểm tra chất lượng dữ liệu, xử lý vấn đề phù hợp và thực hiện EDA để đưa ra kết luận có căn cứ.

---

## Mục tiêu

Sau bài lab, sinh viên có thể:

1. Đọc và khám phá một dataset dạng bảng bằng Pandas.
2. Giải thích ý nghĩa của sample, feature và target trong bài toán học máy.
3. Phân biệt dữ liệu numerical, categorical và ordinal categorical.
4. Kiểm tra missing values, duplicates và giá trị bất thường.
5. Đề xuất cách xử lý dữ liệu dựa trên ngữ cảnh thay vì áp dụng máy móc.
6. Chọn biểu đồ phù hợp để trả lời một câu hỏi EDA.
7. Viết một **Data Readiness Report** ngắn trước khi modeling.

---

## Cấu trúc bài lab

Notebook chính:

```text
Lab01_StudentPerformance.ipynb
```

Dataset sau khi tải:

```text
data/StudentPerformanceFactors.csv
```

Các nhiệm vụ:

- Mission 1 — Understand the dataset
- Mission 2 — Build a data dictionary
- Mission 3 — Data quality audit
- Mission 4 — Investigate suspicious data
- Mission 5 — Handle missing values
- Mission 6 — EDA driven by questions
- Final — Data Readiness Report

---

## Chuẩn bị môi trường

Khuyến nghị Python 3.12.

```bash
conda create --name machine_learning python=3.12
conda activate machine_learning
pip install -r requirements.txt
```

### Tải dataset

Sau khi clone repo, chạy một lần:

```bash
python scripts/download_data.py
```

Script sẽ tạo:

```text
data/StudentPerformanceFactors.csv
```

GitHub Actions cũng tự động thực hiện bước tải dataset khi kiểm tra bài.

Khởi động Jupyter:

```bash
jupyter notebook
```

---

## Nguyên tắc thực hiện

- Không đổi tên file dataset.
- Không dùng đường dẫn tuyệt đối như `C:/Users/...` hoặc `/content/drive/...`.
- Đọc dữ liệu bằng đường dẫn tương đối:

```python
pd.read_csv("data/StudentPerformanceFactors.csv")
```

- Mỗi Mission cần có cả **code** và **nhận xét bằng Markdown**.
- Không chỉ đưa ra biểu đồ; phải giải thích biểu đồ cho thấy điều gì.
- Không tự động xóa outlier hoặc điền missing value nếu chưa giải thích lý do.
- Không chỉnh sửa file CSV gốc.

---

## Yêu cầu nộp bài

Điền đầy đủ thông tin sinh viên ở đầu notebook.

Bài được xem là hoàn thành khi:

- notebook chạy được từ đầu đến cuối;
- các Mission 1–6 có câu trả lời;
- có ít nhất các trực quan hóa được yêu cầu;
- có Data Readiness Report cuối bài;
- bài đã được `git push` lên branch `main`.

Commit gợi ý:

```bash
git add .
git commit -m "Complete Lab 01"
git push
```

---

## GitHub Actions

Mỗi lần push lên `main`, GitHub Actions sẽ:

1. cài môi trường;
2. tải dataset;
3. kiểm tra các file bắt buộc và cấu trúc notebook;
4. chạy notebook trong môi trường sạch;
5. cảnh báo nếu vẫn còn các placeholder chưa hoàn thiện.

Mục tiêu của kiểm tra tự động là giúp phát hiện lỗi kỹ thuật trước khi nộp bài, không thay thế phần đánh giá nhận xét EDA của giảng viên.
