import numpy as np
from numba import njit, prange


def moran_i(x, w):
    n = len(x)
    s0 = w.sum()
    z = x - x.mean()
    z2ss = (z * z).sum()

    wx = (w * z * z).sum()
    I = n / s0 * wx / z2ss
    EI = -1.0 / (n - 1.0)
    n2 = n * n
    w1 = w + w.transpose()
    s1 = w1.multiply(w1).sum() / 2.0
    s2 = (np.array(w.sum(1) + w.sum(0).transpose()) ** 2).sum()
    s02 = s0 * s0
    v_num = n2 * s1 - n * s2 + 3.0 * s02
    v_den = (n - 1.0) * (n + 1.0) * s02
    VI_norm = v_num / v_den - (1.0 / (n - 1.0)) ** 2
    seI_norm = VI_norm ** (1.0 / 2.0)
    z_norm = I - EI / seI_norm
    return I, z_norm


# @njit
# def euclidean_distance(p1, p2):
#     (p1[0] - p2[0]) ** 2 + (p1[1])
#
#
# @njit
# def pdist(points):
#
#     for i in prange(points):
#         for j in prange(points):
#             euclidean_distance(i, j)
