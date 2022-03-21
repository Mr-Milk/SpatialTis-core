from collections import Counter
from typing import List

import numpy as np
from scipy.stats import entropy

from spatialtis_core import spatial_autocorr, spatial_distribution_pattern, spatial_entropy, points_neighbors, getis_ord

N = 100
points: List[List[float]] = np.random.randn(N, 2).tolist()
labels = [i for i in range(len(points))]
types = [str(i) for i in np.random.choice([1, 2, 3, 4, 5, 6], N)]
neighbors = points_neighbors(points, labels, k=3)
exp = np.random.randn(2, N)


def test_spatial_autocorr():
    i = spatial_autocorr(exp, neighbors, labels, method="moran_i")
    c = spatial_autocorr(exp, neighbors, labels, method="geary_c")


def test_spatial_distribution():
    bbox = (-10.0, -10.0, 10.0, 10.0)
    i = spatial_distribution_pattern([points], bbox, method='id')
    assert i[0][-1] == 3
    i = spatial_distribution_pattern([points], bbox, r=0.1, method='id')
    assert i[0][-1] == 3
    i = spatial_distribution_pattern([points], bbox, method='morisita')
    assert i[0][-1] == 3
    i = spatial_distribution_pattern([points], bbox, quad=(10, 10), method='morisita')
    assert i[0][-1] == 3
    i = spatial_distribution_pattern([points], bbox, rect_side=(0.5, 0.5), method='morisita')
    assert i[0][-1] == 3
    i = spatial_distribution_pattern([points], bbox, method="clark_evans")
    assert i[0][-1] == 3


def test_spatial_entropy():
    e = entropy(np.array(list(Counter(types).values())))
    e1 = spatial_entropy([points], [types], method="leibovici")
    e2 = spatial_entropy([points], [types], method="altieri")
    assert e1[0] > e
    assert e2[0] > e


def test_hotspot():
    getis_ord(points, (-3.0, -3.0, 3.0, 3.0))
    getis_ord([], (-3.0, -3.0, 3.0, 3.0))
