import numpy as np

from kmeans_manual import (
    init_centroids,
    assign_clusters,
    update_centroids,
    kmeans_manual,
)


def test_init_centroids_shape_and_rows():
    X = np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]])
    C = init_centroids(X, 2, random_state=42)

    assert C.shape == (2, 2)
    assert any(np.allclose(C[0], row) for row in X)
    assert any(np.allclose(C[1], row) for row in X)
    assert not np.allclose(C[0], C[1])


def test_assign_clusters():
    X = np.array([[0., 0.], [1., 1.], [9., 9.], [10., 10.]])
    C = np.array([[0., 0.], [10., 10.]])
    labels = assign_clusters(X, C)

    assert labels.tolist() == [0, 0, 1, 1]


def test_update_centroids():
    X = np.array([[0., 0.], [2., 2.], [8., 8.], [10., 10.]])
    labels = np.array([0, 0, 1, 1])
    old = np.array([[0., 0.], [10., 10.]])

    C = update_centroids(X, labels, old)

    assert np.allclose(C[0], [1., 1.])
    assert np.allclose(C[1], [9., 9.])


def test_update_centroids_keeps_empty_cluster():
    X = np.array([[0., 0.], [2., 2.]])
    labels = np.array([0, 0])
    old = np.array([[0., 0.], [9., 9.]])

    C = update_centroids(X, labels, old)

    assert np.allclose(C[0], [1., 1.])
    assert np.allclose(C[1], [9., 9.])


def test_kmeans_manual_separates_two_groups():
    X = np.array([
        [0.0, 0.0],
        [0.2, 0.1],
        [0.1, 0.3],
        [9.8, 10.0],
        [10.2, 9.9],
        [10.0, 10.3],
    ])

    C, labels, n_iters = kmeans_manual(X, 2, random_state=7)

    assert C.shape == (2, 2)
    assert labels.shape == (6,)
    assert 1 <= n_iters <= 100

    order = np.argsort(C[:, 0])
    C_sorted = C[order]

    assert np.linalg.norm(C_sorted[0] - np.array([0.1, 0.1333333333])) < 0.5
    assert np.linalg.norm(C_sorted[1] - np.array([10.0, 10.0666666667])) < 0.5
