from typing import List

from scipy.sparse import csr_matrix

from .spatial_de import somde
from .spatialtis_core import (CellCombs, bbox_neighbors,
                              build_neighbors_matrix, comb_bootstrap,
                              dumps_wkt_points, dumps_wkt_polygons, fast_corr,
                              getis_ord, multipoints_bbox, multipolygons_area,
                              neighbor_components, points_bbox,
                              points_neighbors, points_shapes, polygons_area,
                              reads_wkt_points, reads_wkt_polygons,
                              spatial_autocorr, spatial_distribution_pattern,
                              spatial_entropy)
from .spatialtis_core import pdist


def spatial_weight(neighbors: List[List[int]], labels: List[int]) -> csr_matrix:
    """Build a neighbors sparse matrix from neighbors data"""
    shape_n, indptr, col_index, row_index, data = build_neighbors_matrix(neighbors, labels)
    return csr_matrix((data, col_index, indptr))
