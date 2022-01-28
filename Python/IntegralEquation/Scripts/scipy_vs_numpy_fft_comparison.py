import numpy as np
import scipy as sc
import time

n = 2000
precision = np.complex64
ntimes = 20

a = np.zeros(shape=(n, n), dtype=np.complex64)

t1 = time.time()
for _ in range(ntimes):
    b = np.fft.fftn(a, axes=(0, 1))
t2= time.time()
print("Average time taken (numpy) = ", (t2-t1) / ntimes, " s")

t1 = time.time()
for _ in range(ntimes):
    b = sc.fft.fftn(a, axes=(0, 1))
t2= time.time()
print("Average time taken (scipy) = ", (t2-t1) / ntimes, " s")


