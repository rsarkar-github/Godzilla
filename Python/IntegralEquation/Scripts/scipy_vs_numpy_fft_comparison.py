import numpy as np
import scipy as sc
import time

n = 4000
ntimes = 50

print("-------------------------------------------------")
print("Datatype: complex 64-bit tests")
print("-------------------------------------------------")
print("\n")

precision = np.complex64

a = np.zeros(shape=(n, n), dtype=precision)

t1 = time.time()
for _ in range(ntimes):
    np.fft.fftn(a, axes=(0, 1))
t2= time.time()
print("Average time taken (numpy) = ", (t2-t1) / ntimes, " s")

t1 = time.time()
for _ in range(ntimes):
    sc.fft.fftn(a, axes=(0, 1))
t2= time.time()
print("Average time taken (scipy) = ", (t2-t1) / ntimes, " s")


print("\n\n")
print("-------------------------------------------------")
print("Datatype: complex 128-bit tests")
print("-------------------------------------------------")
print("\n")

precision = np.complex128

a = np.zeros(shape=(n, n), dtype=precision)

t1 = time.time()
for _ in range(ntimes):
    np.fft.fftn(a, axes=(0, 1))
t2= time.time()
print("Average time taken (numpy) = ", (t2-t1) / ntimes, " s")

t1 = time.time()
for _ in range(ntimes):
    sc.fft.fftn(a, axes=(0, 1))
t2= time.time()
print("Average time taken (scipy) = ", (t2-t1) / ntimes, " s")
