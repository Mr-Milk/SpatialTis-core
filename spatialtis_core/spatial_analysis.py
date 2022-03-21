from typing import Optional

import numpy as np

from .spatialtis_core import spearman_corr, pearson_corr


def fast_corr(data1: np.ndarray,
              data2: np.ndarray,
              method: Optional[str] = "pearson",
              ) -> np.ndarray:
    """Parallel pairwise correlation

    Compute pairwise (combination with replacement) correlation between
    two 2D array with same shape. Faster than scipy's implementation.

    Args:
        data1: 2D Array
        data2: 2D Array
        method: "pearson" or "spearman"

    """
    if not isinstance(data1, np.ndarray):
        data1 = np.array(data1)
    if not isinstance(data2, np.ndarray):
        data2 = np.array(data2)

    if method == "pearson":
        return pearson_corr(data1, data2)
    elif method == "spearman":
        return spearman_corr(data1, data2)
