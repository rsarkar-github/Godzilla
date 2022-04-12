from scipy.sparse.linalg import lsqr, lsmr, LinearOperator
import numpy as np

def func_matvec(v):
    return v * 2.0

n = 2
B = LinearOperator(shape=(n, n), matvec=func_matvec, rmatvec=func_matvec)
b = np.array([1., 0.], dtype=float)

# x, istop, itn, r1norm = lsqr(B, b)[:4]
# print(istop)
# print(x)
#
# x, istop, itn, r1norm = lsmr(B, b)[:4]
# print(istop)
# print(x)

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import gmres

A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
b = np.array([2, 4, -1], dtype=float)

# Callback generator
def make_callback():
    closure_variables = dict(counter=0, residuals=[])

    def callback(residuals):
        closure_variables["counter"] += 1
        closure_variables["residuals"].append(residuals)
        print(closure_variables["counter"], residuals)
    return callback

x, exitCode = gmres(A, b, maxiter=2, restart=2)
print(exitCode)
print(x)
