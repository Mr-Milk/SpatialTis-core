from spatialtis_core import fast_corr

from scipy.stats import spearmanr

from time import time
import numpy as np
data1 = np.random.rand(1, 1000).tolist()
data2 = np.random.rand(1, 1000).tolist()
labels = [str(i) for i in range(1)]

t1 = time()
print(fast_corr(labels, data1, data1, "spearman"))
t2 = time()
print(f"fast corr used {t2-t1:5f}")

t1 = time()
print(spearmanr(data1[0], data1[0]))
t2 = time()
print(f"scipy used {t2-t1:5f}")
