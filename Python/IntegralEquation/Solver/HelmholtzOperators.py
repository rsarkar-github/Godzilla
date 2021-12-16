from scipy.sparse import csc_matrix
import numpy as np
from . import TypeChecker


def create_helmholtz2d_matrix(vel, omega, n, h, pad, precision, adj=False):
    """
    The domain of the Helmholtz equation is the box [0,1] x [0,1]

    :param vel: 3d numpy array real valued (must be nz x nx dimensions, where n = [nz, nx])
    :param omega: Frequency. The Helmholtz equation reads (lap + omega^2 / vel^2)u = f.
    :param n: 1d numpy array (must be of dimensions (2,), type np.int32).
        Contains the number of cells [nz, nx] in each dimension.
    :param h: 1d numpy array (must be of dimensions (2,), type np.float32).
        Contains the grid spacing [hz, hx] in each dimension.
    :param pad: 1d numpy array (must be of dimensions (2,), type np.float32).
        Contains the number of pad cells [padz, padx] in each dimension.
    :param precision: np.complex64 or np.complex128
    :param adj: boolean flag (whether to compute the adjoint)

    :return: Sparse Helmholtz matrix
    """

    # Check type, shape, values of n
    __check_n(n=n, dim=2)

    if vel.dtype != precision:
        raise TypeError("Type of 'vel' must match that of 'precision': ", precision)

    # Check dimensions
    if vel.shape != (n, self._n, self._n) or output.shape != (self._n, self._n, self._n):
        raise ValueError("Shapes of 'u' and 'output' must be (n, n, n) with n = ", self._n)


"""
Following functions are needed for Typechecking capabilities
"""
def __check_n(n, dim):
    """
    Check if input is np.ndarray, of type np.int32, of length dim, and contains values >= 1
    :param n: Input np.ndarray
    :return:
    """
    def f(x):

        if x.dtype != np.int32:
            msg = "Type of 'n' must match " + str(np.int32)
            return False, False, msg

        if x.shape != (dim,):
            msg = "Shape of 'n' must match " + str((dim, ))
            return False, False, msg

        for i in range(dim):
            if x[i] <= 0:
                msg = ""
                return True, False, msg

    TypeChecker.check(x=n, expected_type=np.ndarray, f=f)

