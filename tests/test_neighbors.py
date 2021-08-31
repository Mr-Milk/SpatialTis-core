import numpy as np
from spatialtis_core import points_neighbors, bbox_neighbors, neighbor_components, spatial_weight

points = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
labels = [1, 2, 3, 4]
bboxs = [(0.0, 0.0, 1.0, 1.0), (2.0, 2.0, 4.0, 5.0)]


def test_kd_tree_search_r():
    neighbors = points_neighbors(points, r=1.0, method="kdtree")
    results = [[0, 1, 3], [0, 1, 2], [1, 2, 3], [0, 2, 3]]
    for n, r in zip(neighbors, results):
        assert set(n) == set(r)


def test_kd_tree_search_k():
    neighbors = points_neighbors(points, k=3, method="kdtree")
    results = [[0, 1, 3], [0, 1, 2], [1, 2, 3], [0, 2, 3]]
    for n, r in zip(neighbors, results):
        assert set(n) == set(r)


def test_kd_tree_search_k_and_r():
    neighbors = points_neighbors(points, r=1.1, k=3, method="kdtree")
    results = [[0, 1, 3], [0, 1, 2], [1, 2, 3], [0, 2, 3]]
    for n, r in zip(neighbors, results):
        assert set(n) == set(r)


def test_delaunay():
    neighbors = points_neighbors(points, method="delaunay")
    results = [[0, 1, 2, 3], [1, 0, 2], [2, 0, 1, 3], [3, 0, 2]]
    for n, r in zip(neighbors, results):
        assert set(n) == set(r)


def test_bbox_neighbors():
    neighbors = bbox_neighbors(bboxs, expand=3)
    results = [[0, 1], [0, 1]]
    for n, r in zip(neighbors, results):
        assert set(n) == set(r)


def test_neighbors_components():
    neighbors = points_neighbors(points, k=3, method="kdtree")
    col, data = neighbor_components(neighbors, [i for i in range(len(points))], ["1", "1", "2", "2"])
    if col == ["1", "2"]:
        assert data == [[2, 1], [2, 1], [1, 2], [1, 2]]
    else:
        assert data == [[1, 2], [1, 2], [2, 1], [2, 1]]


def test_spatial_weight():
    neighbors = points_neighbors(points, k=3, method="kdtree")
    matrix = spatial_weight(neighbors, [0, 1, 2, 3])
    A = matrix.toarray()
    result = np.array([[1 / 3, 1 / 3, 0, 1 / 3],
                       [1 / 3, 1 / 3, 1 / 3, 0],
                       [0, 1 / 3, 1 / 3, 1 / 3],
                       [1 / 3, 0, 1 / 3, 1 / 3]])
    assert np.array_equal(A, result)  # If it match, all elements should be 0.0
