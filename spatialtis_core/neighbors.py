from typing import Optional, List

from .spatialtis_core import (points_neighbors_kdtree,
                              points_neighbors_kdtree_3d,
                              points_neighbors_triangulation,
                              bbox_neighbors_rtree,
                              points_neighbors_kdtree_parallel,
                              points_neighbors_kdtree_3d_parallel,
                              points_neighbors_triangulation_parallel,
                              bbox_neighbors_rtree_parallel
                              )
from .types import Points, Labels, BoundingBox


def points_neighbors(points: Points,
                     labels: Labels,
                     r: Optional[float] = None,
                     k: Optional[int] = None,
                     method: str = "kdtree",
                     ) -> List[List[int]]:
    """Get neighbors for each points

    When search with KD-tree, you can use either `r` or `k`.
    If r = 5, it will search for all points within 5; If k = 5, it will
    search for the 5 nearest neighbors; If combined, r = 5 and k = 5, it will
    search for at most 5 neighbors within 5;

    When search with Delaunay triangulation, there is no parameter.

    The return list follow the order of labels. For example, if serach for points of
    [100, 101, 102], the result may look like [[100, 102], [101, 102], [102, 100, 101]]

    Args:
        points: A list of points
        labels: Integer to labels your points
        r: Radius range to search for neighbors
        k: Number of nearest neighbors
        method: "kdtree" or "delaunay"

    Return:
         A list of neighbors

    """
    # Determine the dimension of points
    dims = len(points[0])

    if (r is None) & (k is None):
        k = 5
    elif r is None:
        r = -1.0
    elif k is None:
        k = 0

    if dims == 2:
        if method == "kdtree":
            return points_neighbors_kdtree(points, labels, r, k)
        else:
            return points_neighbors_triangulation(points, labels)
    elif dims == 3:
        return points_neighbors_kdtree_3d(points, labels, r, k)
    else:
        raise NotImplementedError("Only support 2D and 3D data")


def points_neighbors_parallel(points_collections: List[Points],
                              labels_collections: List[Labels],
                              r: Optional[float] = None,
                              k: Optional[int] = None,
                              method: str = "kdtree",
                              ) -> List[List[int]]:
    """Get neighbors for each points

    When search with KD-tree, you can use either `r` or `k`.
    If r = 5, it will search for all points within 5; If k = 5, it will
    search for the 5 nearest neighbors; If combined, r = 5 and k = 5, it will
    search for at most 5 neighbors within 5;

    When search with Delaunay triangulation, there is no parameter.

    The return list follow the order of labels. For example, if serach for points of
    [100, 101, 102], the result may look like [[100, 102], [101, 102], [102, 100, 101]]

    Args:
        points_collections: A list of points in ROIs
        labels_collections: Integer to labels your points
        r: Radius range to search for neighbors
        k: Number of nearest neighbors
        method: "kdtree" or "delaunay"

    Return:
         A list of neighbors

    """
    # Determine the dimension of points
    dims = len(points_collections[0][0])

    if (r is None) & (k is None):
        k = 5
    elif r is None:
        r = -1.0
    elif k is None:
        k = 0

    if dims == 2:
        if method == "kdtree":
            return points_neighbors_kdtree_parallel(points_collections, labels_collections, r, k)
        else:
            return points_neighbors_triangulation_parallel(points_collections, labels_collections)
    elif dims == 3:
        return points_neighbors_kdtree_3d_parallel(points_collections, labels_collections, r, k)
    else:
        raise NotImplementedError("Only support 2D and 3D data")


def bbox_neighbors(bbox: List[BoundingBox],
                   labels: Labels,
                   expand: float = -1.0,
                   scale: float = 1.3,
                   ) -> List[List[int]]:
    """Get neighbors for each bounding box

    Args:
        bbox: A list of bounding box
        labels: Integer to label your bounding box
        expand: Expand the bounding box to search for neighbors
        scale: Scale the bounding box to search for neighbors

    Return:
        A list of neighbors

    """
    return bbox_neighbors_rtree(bbox, labels, expand, scale)


def bbox_neighbors_parallel(bbox_collections: List[List[BoundingBox]],
                            labels: List[Labels],
                            expand: Optional[float] = None,
                            scale: Optional[float] = None,
                            ) -> List[List[int]]:
    """Get neighbors for each bounding box

    Args:
        bbox_collections: A list of bounding box
        labels: Integer to label your bounding box
        expand: Expand the bounding box to search for neighbors
        scale: Scale the bounding box to search for neighbors

    Return:
        A list of neighbors

    """
    if (expand is None) & (scale is None):
        scale = 1.3
        expand = -1.0

    if expand is None:
        expand = -1.0

    return bbox_neighbors_rtree_parallel(bbox_collections, labels, expand, scale)
