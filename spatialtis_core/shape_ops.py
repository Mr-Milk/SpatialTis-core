from typing import List, Sequence

import numpy as np

from .spatialtis_core import _points_bbox, _multipoints_bbox, _polygon_area, _multipolygons_area, concave, convex
from .types import Points, BoundingBox
from .utils import show_options


def points_bbox(points: Points) -> BoundingBox:
    """Return minimum bounding box of points

    Args:
        points: A list of points

    Return:
        The bounding box (minx, miny, maxx, maxy)

    """
    if isinstance(points, np.ndarray):
        points = points.tolist()
    return _points_bbox(points)


def multipoints_bbox(points_collections: Sequence[Points]) -> Sequence[BoundingBox]:
    """A utility function to return minimum bounding box list of polygons

    Args:
        points_collections: List of 2d points collections

    Return:
        A list of bounding box (minx, miny, maxx, maxy)

    """
    if isinstance(points_collections, np.ndarray):
        points_collections = points_collections.tolist()
    return _multipoints_bbox(points_collections)


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


def points_shapes(points: Points, method: str = "convex", concavity: float = 1.5) -> Points:
    """Acquire multipoints (shapes) that describe the points

    Args:
        points: A list of points
        method: "convex" or "concave"
        concavity: Determine the concavity in concave hull

    Return:
        A list of points

    """
    if isinstance(points, np.ndarray):
        points = points.tolist()

    if method == "concave":
        return concave(points, concavity)
    elif method == "convex":
        return convex(points)
    else:
        msg = show_options(method, ["concave", "convex"])
        raise ValueError(msg)

