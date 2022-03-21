from typing import Tuple, List, Optional

import numpy as np

from .spatialtis_core import (
    moran_i_parallel,
    geary_c_parallel,
    ix_dispersion_parallel,
    ix_dispersion_3d_parallel,
    morisita_parallel,
    clark_evans_parallel,
    leibovici_parallel,
    altieri_parallel,
    leibovici_3d_parallel,
    altieri_3d_parallel,
    hotspot,
    _points_bbox,
    _points3d_bbox
)
from .types import Neighbors, Labels, Points, BoundingBox
from .utils import show_options, default_radius, default_radius_3d


def spatial_autocorr(x: np.ndarray,
                     neighbors: Neighbors,
                     labels: Labels,
                     two_tailed: bool = True,
                     pval: float = 0.05,
                     method: str = "moran_i",
                     ) -> List[Tuple[float, float]]:
    """Compute spatial auto-correlation value for a 2D array in parallel

    The p-value is under the assumption of normal distribution
    Return is tuples of (spatial_autocorr value, p value)

    Args:
        x: Gene expression matrix, each row is the expression of a gene
        neighbors: A list of neighbors
        labels: A list of labels
        two_tailed: Determine the p value
        pval: The p-value threshold
        method: "moran_i" or "geary_c"

    Return:
        A list of (value, p_value)

    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if method == "moran_i":
        return moran_i_parallel(x, neighbors, labels, two_tailed, pval)
    elif method == "geary_c":
        return geary_c_parallel(x, neighbors, labels, pval)
    else:
        msg = show_options(method, ["moran_i", "geary_c"])
        raise ValueError(msg)


def spatial_distribution_pattern(points_collections: List[Points],
                                 bbox: BoundingBox,
                                 method: str = "id",
                                 r: Optional[float] = None,
                                 resample: int = 1000,
                                 quad: Optional[Tuple[int, int]] = None,
                                 rect_side: Optional[Tuple[float, float]] = None,
                                 pval: float = 0.05,
                                 min_cells: int = 10,
                                 dims: int = 2,
                                 ) -> List[Tuple[float, float, int]]:
    """Compute the distribution index and determine the pattern for different cells in a ROI in parallel

    If data is 3D, only method="id" is supported

    Args:
        points_collections: A list of list of points
        bbox: The bounding box
        method: "id" for index of dispersion, "morisita" for morisita index, "clark_evans" for clark evans' index
        r: If method == "id"; The sample windows' radius
        resample: If method == "id"; The number of sampling times
        quad: If method == "morisita"; eg.(X, Y) Use X * Y grid to perform analysis
        rect_side: If method == "morisita"; eg.(X, Y) Use X * Y rectangle to perform analysis
        pval: The threshold for p-value
        min_cells: The minimum number of cells to perform analysis
        dims: The dimension of data

    Return:
        A list of (index_value, p_value, pattern)

    """
    if dims == 3:
        return ix_dispersion_3d_parallel(points_collections, bbox, r, resample, pval, min_cells)
    if method == "id":
        if r is None:
            r = default_radius(bbox)
        return ix_dispersion_parallel(points_collections, bbox, r, resample, pval, min_cells)
    elif method == "morisita":
        return morisita_parallel(points_collections, bbox, quad, rect_side, pval, min_cells)
    elif method == "clark_evans":
        return clark_evans_parallel(points_collections, bbox, pval, min_cells)
    else:
        msg = show_options(method, ["id", "morisita", "clark_evans"])
        raise ValueError(msg)


def spatial_entropy(points_collections: List[Points],
                    types_collections: List[List[str]],
                    method: str = "leibovici",
                    d: Optional[float] = None,
                    cut: int = 3,
                    dims: int = 2
                    ) -> List[float]:
    """Compute spatial entropy value of multiple ROIs in parallel

    Args:
        points_collections: A list of list of points
        types_collections: A list of list of types
        bbox: The bounding box
        method: "leibovici" or "altieri"
        d: If method == "leibovici"; The distance threshold to determine co-occurrence
        cut: If method == "altieri"; The distance interval to determine co-occurrence
        dims: The dimension of data

    Return:
        A list of spatial entropy

    """
    if method == "leibovici":
        if dims == 2:
            if d is None:
                bbox = _points_bbox(points_collections[0])
                d = default_radius(bbox)
            return leibovici_parallel(points_collections, types_collections, d)
        else:
            if d is None:
                bbox = _points3d_bbox(points_collections[0])
                d = default_radius_3d(bbox)
            return leibovici_3d_parallel(points_collections, types_collections, d)
    elif method == "altieri":
        if dims == 2:
            return altieri_parallel(points_collections, types_collections, cut)
        else:
            return altieri_3d_parallel(points_collections, types_collections, cut)
    else:
        msg = show_options(method, ["leibovici", "altieri"])
        raise ValueError(msg)


def getis_ord(points: Points,
              bbox: BoundingBox,
              search_level: int = 3,
              quad: Optional[Tuple[int, int]] = None,
              rect_side: Optional[Tuple[float, float]] = None,
              pval: float = 0.05,
              min_cells: int = 10,
              ) -> List[bool]:
    """Getis-ord analysis to find hot cells

    Args:
        points: A list of points
        bbox: The bounding box
        search_level: The level of outer-ring to search for
        quad: eg.(X, Y) Use X * Y grid to perform analysis
        rect_side: eg.(X, Y) Use X * Y rectangle to perform analysis
        pval: The threshold for p-value
        min_cells: The minimum number of cells to perform analysis

    Return:
        A list of bool

    """
    return hotspot(points, bbox, search_level, quad, rect_side, pval, min_cells)

