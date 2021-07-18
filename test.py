from spatialtis_core import points_neighbors,\
    fast_corr, MoranI, neighbors_matrix

from scipy.stats import spearmanr

from time import time
import numpy as np
data1 = np.random.rand(1, 1000).tolist()
data2 = np.random.rand(1, 1000).tolist()
labels = [str(i) for i in range(1)]

# t1 = time()
# print(fast_corr(labels, data1, data1, "spearman"))
# t2 = time()
# print(f"fast corr used {t2-t1:5f}")
#
# t1 = time()
# print(spearmanr(data1[0], data1[0]))
# t2 = time()
# print(f"scipy used {t2-t1:5f}")

# get random points
N = 10000 # number of points
points = [(x, y) for x, y in np.random.randn(N, 2)]
neighbors = points_neighbors(points, r=5)
labels = [_ for _ in range(len(neighbors))]
matrix = neighbors_matrix(neighbors, labels)

exp = np.random.randn(N)
i = MoranI(exp, matrix)
print("Moran'I is", i)
