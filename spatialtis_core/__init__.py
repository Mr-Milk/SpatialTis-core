from typing import List

from scipy.sparse import csr_matrix

from .spatial_de import somde
from .spatialtis_core import (
    points_bbox,
    multipoints_bbox,
    polygons_area,
    multipolygons_area,
    points_shapes,
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
    fast_corr,
)


def spatial_weight(neighbors: List[List[int]], labels: List[int]) -> csr_matrix:
    """Build a neighbors sparse matrix from neighbors data"""
    shape_n, indptr, col_index, row_index, data = build_neighbors_matrix(neighbors, labels)
    return csr_matrix((data, col_index, indptr))
