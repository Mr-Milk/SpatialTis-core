from typing import List, Sequence

import numpy as np
import pandas as pd

from .spatialtis_core import (_points_bbox,
                              _points3d_bbox,
                              _multipoints_bbox,
                              _multipoints3d_bbox,
                              _polygon_area,
                              _multipolygons_area,
                              multipolygons_concave,
                              multipolygons_convex,
                              concave,
                              convex)
from .types import Points, BoundingBox
from .utils import show_options


def points_bbox(points: Points) -> BoundingBox:
    """Return minimum bounding box of points

    Args:
        points: A list of points

    Return:
        The bounding box (minx, miny, maxx, maxy) or (minx, miny, minz, maxx, maxy, maxz)

    """
    if isinstance(points, np.ndarray):
        points = points.tolist()
    if len(points[0]) == 2:
        return _points_bbox(points)
    else:
        return _points3d_bbox(points)


def multipoints_bbox(points_collections: Sequence[Points]) -> Sequence[BoundingBox]:
    """A utility function to return minimum bounding box list of polygons

    Args:
        points_collections: List of 2d points collections

    Return:
        A list of bounding box (minx, miny, maxx, maxy)

    """
    if isinstance(points_collections, np.ndarray):
        points_collections = points_collections.tolist()
    if len(points_collections[0][0]) == 2:
        return _multipoints_bbox(points_collections)
    else:
        return _multipoints3d_bbox(points_collections)


def polygons_area(points: Points) -> float:
    """Calculate the area of polygons using shoelace formula

    Args:
        points: A list of points represents a polygon

    Return:
        The area of the polygon

    """
    if isinstance(points, np.ndarray):
        points = points.tolist()
    return _polygon_area(points)


def multipolygons_area(points_collections: Sequence[Points]) -> Sequence[float]:
    """Calculate the area of polygons using shoelace formula

        Args:
            points_collections: List of 2d points collections, represents a list of polygons

        Return:
            The area of the polygons in list

        """
    if isinstance(points_collections, np.ndarray):
        points_collections = points_collections.tolist()
    return _multipolygons_area(points_collections)


def points_shapes(polygons: List[Points], method: str = "convex", concavity: float = 1.5) -> Points:
    """Acquire multipoints (shapes) that describe the points

    Args:
        polygons: A list of polygons
        method: "convex" or "concave"
        concavity: Determine the concavity in concave hull

    Return:
        A list of points

    """
    if isinstance(polygons, (np.ndarray, pd.Series)):
        polygons = polygons.tolist()

    if method == "concave":
        return multipolygons_concave(polygons, concavity)
    elif method == "convex":
        return multipolygons_convex(polygons)
    else:
        msg = show_options(method, ["concave", "convex"])
        raise ValueError(msg)

