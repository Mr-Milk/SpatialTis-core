from spatialtis_core import (
    multi_points_bbox,
    points2bbox,
    points2shapes,
    points_neighbors,
    bbox_neighbors,
    neighbor_components,
    spatial_autocorr,
    spatial_distribution_pattern,
    spatial_entropy,
    getis_ord,
    comb_bootstrap
)

# from scipy.stats import spearmanr
# from scipy.sparse import csr_matrix
# from libpysal.weights import W
# from esda import Moran, Geary

from time import time
import numpy as np

data1 = np.random.rand(1000, 3000)
data2 = np.random.rand(1000, 3000)
labels = [str(i) for i in range(1)]

# t1 = time()
# print(fast_corr(data1, data1, "spearman"))
# t2 = time()
# print(f"fast corr used {t2-t1:5f}")
#
# t1 = time()
# for a1 in data1:
#     for a2 in data2:
#         spearmanr(a1, a2)
# t2 = time()
# print(f"scipy used {t2-t1:5f}")

# get random points
N = 100  # number of points
points = [(x, y) for x, y in np.random.randn(N, 2)]
types = np.random.randint(0, 30, N)
#
# id = spatial_distribution_pattern([points], (0, 0, 10, 10), method="id")
# morisita = spatial_distribution_pattern([points], (0, 0, 10, 10), method="morisita")
# ce = spatial_distribution_pattern([points], (0, 0, 10, 10), method="clark_evans")
#
# print(f"spatial dist {id} {morisita} {ce}")
#
# print(f"hotspot {getis_ord(points, (0, 0, 10, 10))}")


t1 = time()
e = spatial_entropy([points], [types.tolist()], method="leibovici")
t2 = time()
print(f"Get leibovici entropy {e} used {t2 - t1:5f}")

t1 = time()
e = spatial_entropy([points, points], [types.tolist(), types.tolist()], method="altieri")
t2 = time()
print(f"Get altieri entropy {e} used {t2 - t1:5f}")


# neighbors = points_neighbors(points, r=5)
# labels = [_ for _ in range(len(neighbors))]
# t1 = time()
# matrix = neighbors_matrix(neighbors, labels)
# t2 = time()
# print(f"Get matrix used {t2 - t1:5f}")

# neighbors_obj = dict(zip(labels, neighbors))
#
# exp = np.random.randn(1, N)
# t1 = time()
# i = spatial_autocorr(exp, neighbors, labels)
# t2 = time()
# print(f"Used {t2 - t1:5f} Moran'I is {i}")
#
#
# t1 = time()
# w = W(neighbors_obj)
# m = Moran(exp[0], w)
# i = m.I
# t2 = time()
# print(f"Used {t2 - t1:5f} esda Moran'I is {i} p {m.p_norm}")
#
# t1 = time()
# i = spatial_autocorr(exp, neighbors, labels, method="geary_c")
# t2 = time()
# print(f"Used {t2-t1:5f} Geary'C is {i}")
# m = Geary(exp, w)
# print(f"esda result C:{m.C}, p:{m.p_norm}")
