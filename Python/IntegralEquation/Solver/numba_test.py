import numpy as np
import scipy.special as sp
import numba as nb


x = np.asarray([[1,2,3],[4,5,6],[7,8,9]])
y = np.asarray([1,2,3,4,5])
y = np.reshape(y, newshape=(5,1,1))

z = y * x
print(x)
print(y)
print(z)