from .spatialtis_core import (
    multi_points_bbox,
    points2bbox,
    points2shapes,
    points_neighbors,
    bbox_neighbors,
    neighbor_components,
    spatial_autocorr,
    spatial_entropy,
    spatial_distribution_pattern,
    build_neighbors_matrix,
    getis_ord,
    comb_bootstrap,
    CellCombs,
)

from scipy.sparse import csr_matrix
from typing import List


def spatial_weight(neighbors: List[List[int]], labels: List[int]) -> csr_matrix:
    """Build a neighbors sparse matrix from neighbors data"""
    shape_n, indptr, col_index, row_index, data = build_neighbors_matrix(neighbors, labels)
    return csr_matrix((data, col_index, indptr))
