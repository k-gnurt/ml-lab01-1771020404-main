import math
import numpy as np
import pytest

from knn_manual import (
    euclidean_distance,
    manhattan_distance,
    get_k_neighbors,
    knn_predict_one,
)


def test_euclidean_basic():
    assert euclidean_distance([3, 3], [5, 5]) == pytest.approx(math.sqrt(8))


def test_manhattan_basic():
    assert manhattan_distance([3, 3], [5, 5]) == pytest.approx(4.0)


def test_neighbors_are_sorted_and_have_expected_length():
    X = np.array([[0, 0], [2, 0], [1, 0], [4, 0]], dtype=float)
    y = np.array(["A", "B", "A", "B"])
    neighbors = get_k_neighbors(X, y, np.array([0.5, 0.0]), k=3, metric="euclidean")

    assert len(neighbors) == 3
    distances = [item[1] for item in neighbors]
    assert distances == sorted(distances)
    assert [item[0] for item in neighbors] == [0, 2, 1]


def test_metric_can_change_prediction_without_vote_tie():
    # sample=(5,5). Chỉ đổi metric, K=3 giữ nguyên.
    # Euclidean -> ba hàng xóm A; Manhattan -> 1 A + 2 B.
    X = np.array([
        [6, 6],   # A
        [8, 8],   # A
        [2, 8],   # A
        [5, 10],  # B
        [10, 5],  # B
    ], dtype=float)
    y = np.array(["A", "A", "A", "B", "B"])
    sample = np.array([5, 5], dtype=float)

    assert knn_predict_one(X, y, sample, k=3, metric="euclidean") == "A"
    assert knn_predict_one(X, y, sample, k=3, metric="manhattan") == "B"


def test_unknown_metric_rejected():
    X = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
    y = np.array([0, 1, 1])
    with pytest.raises((ValueError, KeyError)):
        get_k_neighbors(X, y, [1, 0], k=1, metric="cosine")
