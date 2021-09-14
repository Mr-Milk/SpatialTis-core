from collections import Counter

import numpy as np
from scipy.stats import entropy

from spatialtis_core import spatial_autocorr, spatial_distribution_pattern, spatial_entropy, points_neighbors, getis_ord

points = [(x, y) for x, y in np.random.randn(100, 2)]
labels = [i for i in range(len(points))]
types = [i for i in np.random.choice([1, 2, 3, 4, 5, 6], 100)]
neighbors = points_neighbors(points, k=3)
exp = np.random.randn(2, 100)


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


def test_hotspot():
    getis_ord(points, (-3.0, -3.0, 3.0, 3.0))
    getis_ord([], (-3.0, -3.0, 3.0, 3.0))
