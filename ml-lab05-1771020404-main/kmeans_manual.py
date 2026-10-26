"""K-Means from scratch — Lab 05.

Hoàn thiện các TODO. Không dùng sklearn trong file này.
Chỉ dùng NumPy.
"""

import numpy as np


def init_centroids(X, k, random_state=42):
    """Chọn ngẫu nhiên k điểm khác nhau trong X làm centroid ban đầu.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    k : int
    random_state : int

    Returns
    -------
    centroids : ndarray, shape (k, n_features)
    """
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]
    if not 1 <= k <= n_samples:
        raise ValueError("k phải nằm trong khoảng [1, n_samples]")

    rng = np.random.default_rng(random_state)
    idx = rng.choice(n_samples, size=k, replace=False)  # k chỉ số khác nhau
    return X[idx].copy()


def assign_clusters(X, centroids):
    """Gán mỗi điểm vào centroid gần nhất theo Euclidean distance.

    Returns
    -------
    labels : ndarray, shape (n_samples,)
    """
    X = np.asarray(X, dtype=float)
    centroids = np.asarray(centroids, dtype=float)

    # Broadcasting: (n, 1, d) - (1, k, d) -> (n, k, d)
    diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=2))  # shape (n, k)
    return np.argmin(distances, axis=1)


def update_centroids(X, labels, old_centroids):
    """Tính centroid mới bằng trung bình các điểm trong từng cụm.

    Nếu một cụm rỗng, giữ nguyên centroid cũ của cụm đó.
    """
    X = np.asarray(X, dtype=float)
    old_centroids = np.asarray(old_centroids, dtype=float)
    new_centroids = old_centroids.copy()

    for j in range(old_centroids.shape[0]):
        members = X[labels == j]
        if len(members) > 0:  # cụm rỗng -> giữ centroid cũ
            new_centroids[j] = members.mean(axis=0)
    return new_centroids


def kmeans_manual(X, k, max_iters=100, tol=1e-4, random_state=42):
    """Chạy K-Means cho tới khi centroid gần như không đổi.

    Dùng np.allclose(..., atol=tol) để kiểm tra hội tụ.

    Returns
    -------
    centroids : ndarray
    labels : ndarray
    n_iters : int
    """
    X = np.asarray(X, dtype=float)
    centroids = init_centroids(X, k, random_state)
    labels = assign_clusters(X, centroids)
    n_iters = 0

    for i in range(1, max_iters + 1):
        n_iters = i
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, centroids)

        converged = np.allclose(new_centroids, centroids, atol=tol)
        centroids = new_centroids
        if converged:
            break

    # Gán nhãn lần cuối theo centroid cuối cùng để labels khớp với centroids
    labels = assign_clusters(X, centroids)
    return centroids, labels, n_iters