from scipy.sparse import csc_matrix
import numpy as np
from . import TypeChecker


def create_helmholtz2d_matrix(
        a1,
        a2,
        n1,
        n2,
        pad1,
        pad2,
        omega,
        precision,
        vel,
        pml_damping=1.0,
        adj=False,
        warnings=True
):
    """
    :param a1: The domain on which Helmholtz equation is solved is [0,a1] x [0,a2]
    :param a2: The domain on which Helmholtz equation is solved is [0,a1] x [0,a2]
    :param n1: Number of uniformly spaced points along x1 direction.
    :param n2: Number of uniformly spaced points along x2 direction.
    :param pad1: Number of pad points along x1 direction.
    :param pad2: Number of pad points along x2 direction.
    :param omega:  Angular frequency. The Helmholtz equation reads (lap + omega^2 / vel^2)u = f.
    :param precision: np.complex64 or np.complex128
    :param vel: 2d numpy array real valued (must be n1 x n2 dimensions)
    :param pml_damping: Positive float. Damping parameter for pml
    :param adj: Boolean flag (whether to compute the adjoint)
    :param warnings: Boolean flag (whether to print warnings due to grid param check)

    :return: Sparse Helmholtz matrix
    """

    # Check inputs
    TypeChecker.check_float_positive(x=a1)
    TypeChecker.check_float_positive(x=a2)
    TypeChecker.check_int_lower_bound(x=n1, lb=4)
    TypeChecker.check_int_lower_bound(x=n2, lb=4)
    TypeChecker.check_int_bounds(x=pad1, lb=2, ub=int(n1 / 2))
    TypeChecker.check_int_bounds(x=pad2, lb=2, ub=int(n2 / 2))
    TypeChecker.check(x=precision, expected_type=(np.complex64, np.complex128))
    TypeChecker.check_float_positive(x=pml_damping)
    TypeChecker.check(x=adj, expected_type=(bool,))
    TypeChecker.check(x=warnings, expected_type=(bool,))

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

    # Check omega
    TypeChecker.check_float_positive(x=omega)

    # Print grid size conditions (recommended)
    d1 = a1 / (n1 - 1)
    d2 = a2 / (n2 - 1)
    vel_max = np.max(vel1)
    vel_min = np.min(vel1)
    lambda_min = 2 * np.pi * vel_min / omega
    lambda_max = 2 * np.pi * vel_max / omega
    pml_width1 = d1 * (pad1 - 1)
    pml_width2 = d2 * (pad2 - 1)
    dmin = lambda_min / 10.0

    if warnings:
        print("\n\n")

        if d1 < dmin:
            print("Warning: Required dmin = ", "{:.2e}".format(dmin), ", d1 = ", "{:.2e}".format(d1))

        if d2 < dmin:
            print("Warning: Required dmin = ", "{:.2e}".format(dmin), ", d2 = ", "{:.2e}".format(d2))

        if pml_width1 < lambda_max:
            print("Warning: Required minimum pml width = ", "{:.2e}".format(lambda_max),
                  ", pml_width1 = ", "{:.2e}".format(pml_width1))

        if pml_width2 < lambda_max:
            print("Warning: Required minimum pml width = ", "{:.2e}".format(lambda_max),
                  ", pml_width2 = ", "{:.2e}".format(pml_width2))

    # Calculate s1 and s2 arrays
    def s1_array():

        pml_width = self.__vel2D.geometry2D.ncellsX_pad * self.__vel2D.geometry2D.dx
        dx = self.__vel2D.geometry2D.dx / 2.0
        end_x = self.__vel2D.geometry2D.dimX + 2 * pml_width
        sx = np.zeros(shape=(2 * self.__vel2D.geometry2D.gridpointsX - 1,), dtype=np.complex64)

        for i1 in range(2 * self.__vel2D.geometry2D.ncellsX_pad + 1):
            sx[i1] = (1.0 - i1 * dx / pml_width) ** 2

        for i1 in range(2 * (self.__vel2D.geometry2D.ncellsX + self.__vel2D.geometry2D.ncellsX_pad),
                        2 * self.__vel2D.geometry2D.gridpointsX - 1):
            sx[i1] = (1.0 - (end_x - i1 * dx) / pml_width) ** 2

        sx = (self.__pml_damping / pml_width) * sx
        sx = 1 + Common.i * sx / omega
        return 1.0 / sx