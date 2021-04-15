import numpy as np
import scipy.special as sp
import numba as nb


@nb.njit
def bessel0(x):
    return sp.j0(x)


x = np.zeros((10, 1))
x += np.random.uniform(0, 1)
y = bessel0(x)
print(y)