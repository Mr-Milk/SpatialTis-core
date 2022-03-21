from typing import List, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from .types import Neighbors, Labels


def spatial_weight(neighbors: Neighbors, labels: Labels) -> csr_matrix:
    """Build a neighbors sparse matrix from neighbors data

    Args:
        neighbors: List of neighbors
        labels: List of neighbors

    Return:
        A scipy sparse matrix in csr format

    """


def neighbor_components(neighbors: Neighbors,
                        labels: Labels,
                        types: List[str],
                        ) -> (List[str], List[List[int]]):
    """Compute the number of different cells at neighbors

    Args:
        neighbors: The neighbors dict
        labels: Integer to label points
        types: A list of types match to points

    Return:
        (header, data): can be used to construct dataframe

    """
    ...


def comb_bootstrap(exp_matrix: np.ndarray,
                   markers: List[str],
                   neighbors: Neighbors,
                   labels: Labels,
                   pval: float = 0.05,
                   order: bool = False,
                   times: int = 1000,
                   ignore_self: bool = False,
                   ) -> List[Tuple[str, str, float]]:
    """
    Bootstrap between two types

    If you want to test co-localization between protein X and Y, first determine if the cell is X-positive
    and/or Y-positive. True is considered as positive and will be counted.

    Args:
        exp_matrix: The expression matrix, each row should be a marker
        markers: Match to the row of exp_matrix
        neighbors: List of neighbors
        labels: List of labels
        pval: The threshold of p-value
        order: If order, (A, B) and (B, A) is different
        times: How many times to perform bootstrap
        ignore_self: Whether to consider self as a neighbor

    Return:
        The significance between markers List of (marker1, marker2, p-value)

    """
    ...


class CellCombs:
    """Profile cell-cell interaction using permutation test

    Args:
        types: All the type of cells in your research
        order: bool (False); If False, A->B and A<-B is the same

    """

    def __init__(self, types: List[str], order: bool = False): ...

    def bootstrap(self,
                  types: List[str],
                  neighbors: Neighbors,
                  labels: Labels,
                  times: int = 1000,
                  pval: float = 0.05,
                  method: str = 'pval',
                  ignore_self: bool = False,
                  ) -> List[Tuple[str, str, float]]:
        """
        Bootstrap functions

        1.0 means association, -1.0 means avoidance, 0.0 means insignificance.

        Args:
            types: The type of all the cells
            neighbors: List of neighbors
            labels: List of labels
            times: How many times to perform bootstrap
            pval: The threshold of p-value
            method: 'pval' or 'zscore'
            ignore_self: Whether to consider self as a neighbor

        Return:
            List of tuples, eg.('a', 'b', 1.0), the type a and type b has a relationship as association

        """
        ...
