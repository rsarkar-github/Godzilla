import numpy as np
import time
from numba import jit, double


s = 15
array_a = np.random.rand(s ** 3).reshape(s, s, s)
array_b = np.random.rand(s ** 3).reshape(s, s, s)


# Original code
def custom_convolution(A, B):

    dimA = A.shape[0]
    dimB = B.shape[0]
    dimC = dimA + dimB

    C = np.zeros((dimC, dimC, dimC))
    for x1 in range(dimA):
        for x2 in range(dimB):
            for y1 in range(dimA):
                for y2 in range(dimB):
                    for z1 in range(dimA):
                        for z2 in range(dimB):
                            x = x1 + x2
                            y = y1 + y2
                            z = z1 + z2
                            C[x, y, z] += A[x1, y1, z1] * B[x2, y2, z2]
    return C


# Numba'ing the function with the JIT compiler
fast_convolution = jit(double[:, :, :](double[:, :, :], double[:, :, :]))(custom_convolution)

start = time.time()
slow_result = custom_convolution(array_a, array_b)
end = time.time()
print("Elapsed = %s" % (end - start))

start = time.time()
fast_result = fast_convolution(array_a, array_b)
end = time.time()
print("Elapsed = %s" % (end - start))

print(np.max(np.abs(slow_result - fast_result)))
