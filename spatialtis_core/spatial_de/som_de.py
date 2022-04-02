from typing import List, Union

import numpy as np
import pandas as pd

from ..external import SomNode


def somde(exp: pd.DataFrame,
          coord: Union[List, np.ndarray],
          k: int = 20,
          alpha: float = 0.5,
          epoch: int = 100,
          pval: float = 0.05,
          qval: float = 0.05) -> np.ndarray:
    """A wrapper for somde, a method to identify spatial variable genes

    `Publications <https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btab471/6308937>`_

    Github: `SOMDE <https://github.com/WhirlFirst/somde>`_

    Args:
        exp: A dataframe, gene name as index, spatial points as columns
        coord: N*2 array of coordination
        k: Number of SOM nodes
        alpha: Parameters for generate pseudo gene expression
        epoch: Number of epoch
        qval: Threshold for pval
        pval: Threshold for pval

    """
    if not isinstance(coord, np.ndarray):
        coord = np.array(coord)
    som = SomNode(coord, k, epoch)
    som.mtx(exp, alpha=alpha)
    sv_genes = np.array([])
    if som.norm():
        result, _ = som.run()
        if result is not None:
            sv_genes = result[(result['pval'] < pval) & (result['qval'] < qval)]['g'].to_numpy()
    return sv_genes
