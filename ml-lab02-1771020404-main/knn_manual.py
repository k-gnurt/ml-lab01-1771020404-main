"""LAB 02 - KNN from scratch.


Hoàn thiện các hàm trong file này mà KHÔNG dùng sklearn.neighbors.
Bạn có thể dùng Python chuẩn và NumPy.
"""


from collections import Counter
from math import sqrt




def euclidean_distance(x, y):
    """Trả về Euclidean distance giữa hai vector cùng số chiều.


    Parameters
    ----------
    x, y : iterable of numbers


    Returns
    -------
    float
    """
    # Implementation
    x, y = list(x), list(y)
    if len(x) != len(y):
        raise ValueError("Vectors must have the same dimension")
    return float(sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(x, y))))




def manhattan_distance(x, y):
    """Trả về Manhattan distance giữa hai vector cùng số chiều."""
    # Implementation
    x, y = list(x), list(y)
    if len(x) != len(y):
        raise ValueError("Vectors must have the same dimension")
    return float(sum(abs(float(a) - float(b)) for a, b in zip(x, y)))




def get_k_neighbors(X_train, y_train, sample, k=3, metric="euclidean"):
    """Tìm K hàng xóm gần sample nhất.


    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
    y_train : array-like, shape (n_samples,)
    sample : array-like, shape (n_features,)
    k : int
    metric : {'euclidean', 'manhattan'}


    Returns
    -------
    list[tuple]
        Danh sách đã sắp xếp theo khoảng cách tăng dần.
        Mỗi phần tử có dạng: (index, distance, label)


    Gợi ý
    -----
    1. Chọn hàm distance theo metric.
    2. Tính distance từ sample tới từng row trong X_train.
    3. Lưu (index, distance, label).
    4. Sort theo (distance, index) để tie có kết quả xác định.
    5. Trả về K phần tử đầu tiên.
    """
    # Implementation
    from numbers import Integral
    if not isinstance(k, Integral) or isinstance(k, bool) or not 1 <= k <= len(X_train):
        raise ValueError("k must be an integer from 1 to len(X_train)")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length")
    metrics = {"euclidean": euclidean_distance, "manhattan": manhattan_distance}
    if metric not in metrics:
        raise ValueError("Unknown metric: " + str(metric))
    distance = metrics[metric]
    neighbors = [(i, distance(row, sample), label)
                 for i, (row, label) in enumerate(zip(X_train, y_train))]
    return sorted(neighbors, key=lambda item: (item[1], item[0]))[:k]




def knn_predict_one(X_train, y_train, sample, k=3, metric="euclidean"):
    """Dự đoán nhãn cho một sample bằng majority vote.


    Trong lab chính, ưu tiên K lẻ để hạn chế hòa phiếu.
    """
    # Implementation
    neighbors = get_k_neighbors(X_train, y_train, sample, k, metric)
    counts = Counter(label for _, _, label in neighbors)
    # Khi hòa phiếu, ưu tiên nhãn có hàng xóm gần nhất.
    return max(counts, key=counts.get)

