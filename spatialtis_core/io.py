from typing import List

import numpy as np

from .spatialtis_core import points_wkt, wkt_points, polygons_wkt, wkt_polygons
from .types import Points


def dumps_points_wkt(points: Points) -> List[str]:
    """Dumps points into wkt format

    Args:
        points: A list of 2D points

    Returns:
        A list of wkt string

    """
    if isinstance(points, np.ndarray):
        points = points.tolist()
    return points_wkt(points)


def reads_wkt_points(wkt_strings: List[str]) -> Points:
    """Reads wkt points into python object

    Args:
        wkt_strings: A list of wkt string represents points

    Returns:
        A list of 2D points

    """
    return wkt_points(wkt_strings)


def dumps_polygons_wkt(polygons: List[Points]) -> List[str]:
    """Dumps points into wkt format

    Args:
        polygons: A list of polygons

    Returns:
        A list of wkt string

    """
    return polygons_wkt(polygons)


def reads_wkt_polygons(wkt_strings: List[str]) -> List[Points]:
    """Reads wkt points into python object

    Args:
        wkt_strings: A list of wkt string represents polygons

    Returns:
        A list of 2D polygons

    """
    return wkt_polygons(wkt_strings)
