import numba
import numpy as np
import time
from ..Solver import SpecialFunc

@numba.jit(nopython=True, parallel=True)
def func(a):
    for i in numba.prange(a.shape[0]):
    # for i in range(a.shape[0]):
        print("i = ", i)
        for j in range(1000):
            a[i, :] = SpecialFunc.gamma(a[i, :])


_a = np.zeros(shape=(20, 100000)) + 1.0

start_t_ = time.time()
_b = func(_a.astype(np.complex128))
end_t_ = time.time()
print("Total time: ", "{:4.2f}".format(end_t_ - start_t_), " s \n")
