from scipy.sparse import csr_matrix

from .types import Neighbors, Labels
from .spatial_de import somde
from .spatialtis_core import build_neighbors_matrix
# import rust mod as naive function, document in .pyi
from .cell_interaction import (neighbor_components, CellCombs, comb_bootstrap)

# import python side function
from .io import dumps_points_wkt, dumps_polygons_wkt, reads_wkt_points, reads_wkt_polygons
from .neighbors import points_neighbors, bbox_neighbors
from .shape_ops import points_bbox, multipoints_bbox, points_shapes, polygons_area, multipolygons_area
from .geo_analysis import spatial_autocorr, spatial_distribution_pattern, spatial_entropy, getis_ord
from .spatial_analysis import fast_corr


def spatial_weight(neighbors: Neighbors, labels: Labels) -> csr_matrix:
    """Build a neighbors sparse matrix from neighbors data

    Args:
        neighbors: List of neighbors
        labels: List of neighbors

    Return:
        A scipy sparse matrix in csr format

    """
    shape_n, indptr, col_index, row_index, data = build_neighbors_matrix(neighbors, labels)
    return csr_matrix((data, col_index, indptr))

