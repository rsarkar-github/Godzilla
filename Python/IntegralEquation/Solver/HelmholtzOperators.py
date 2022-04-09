from scipy.sparse import csc_matrix
import numpy as np
from . import TypeChecker


def create_helmholtz2d_matrix(a1, a2, n1, n2, pad1, pad2, omega, precision, vel, adj=False):
    """
    :param a1: The domain on which Helmholtz equation is solved is [0,a1] x [0,a2]
    :param a2: The domain on which Helmholtz equation is solved is [0,a1] x [0,a2]
    :param n1: Number of uniformly spaced points along x1 direction.
    :param n2: Number of uniformly soaced points along x2 direction.
    :param pad1: Number of pad points along x1 direction.
    :param pad2: Number of pad points along x2 direction.
    :param omega:  Angular frequency. The Helmholtz equation reads (lap + omega^2 / vel^2)u = f.
    :param precision: np.complex64 or np.complex128
    :param vel: 2d numpy array real valued (must be n1 x n2 dimensions
    :param adj: boolean flag (whether to compute the adjoint)

    :return: Sparse Helmholtz matrix
    """

    # Check inputs
    TypeChecker.check_float_positive(x=a1)
    TypeChecker.check_float_positive(x=a2)
    TypeChecker.check_int_lower_bound(x=n1, lb=3)
    TypeChecker.check_int_lower_bound(x=n2, lb=3)
    TypeChecker.check_int_bounds(x=pad1, lb=1, ub=int(n1 / 2))
    TypeChecker.check_int_bounds(x=pad2, lb=1, ub=int(n2 / 2))
    TypeChecker.check(x=precision, expected_type=(np.complex64, np.complex128))
    TypeChecker.check(x=adj, expected_type=(bool,))

    # Check vel for type, make copy consistent with type of precision, and check if values > 0
    TypeChecker.check(x=vel, expected_type=(np.ndarray,))
    if vel.shape != (n1, n2):
        raise ValueError("Shape of 'vel' must match (" + str(n1) + ", " + str(n2) + ")")
    if vel.dtype not in (np.float32, np.float64):
        raise TypeError("dtype of 'vel' must be np.float32 or np.float64")

    vel1 = np.copy(vel)
    if precision is np.complex64:
        vel1.astype(np.float32)
    if precision is np.complex128:
        vel1.astype(np.float64)

    if np.any(vel1 <= 0.0):
        raise ValueError("Non-positive velocities detected after casting type")


